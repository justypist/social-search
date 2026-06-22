from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any, Protocol

from .audio import FfmpegAudioExtractor
from .errors import ExtractionError
from .formats import (
    subtitle_text_to_transcript,
    write_json,
    write_paragraph_srt,
    write_srt,
    write_transcript_json,
    write_transcript_text,
)
from .models import ExtractConfig, ExtractionResult, ExtractionState, SubtitleRef, Transcript
from .paths import prepare_output_dir, relative_or_name
from .progress import ProgressCallback, StageProgressCallback, stage_progress_callback
from .subtitles import select_subtitle
from .transcriber import FasterWhisperTranscriber, Transcriber
from .yt_dlp_client import YtDlpClient


class MediaClient(Protocol):
    def probe(self, url: str) -> dict[str, Any]:
        ...

    def download_subtitle_text(self, subtitle: SubtitleRef) -> str:
        ...

    def download_audio(
        self,
        url: str,
        output_dir: Path,
        *,
        progress_callback: StageProgressCallback | None = None,
    ) -> Path:
        ...

    def download_video(
        self,
        url: str,
        output_dir: Path,
        *,
        progress_callback: StageProgressCallback | None = None,
    ) -> Path:
        ...


class AudioExtractor(Protocol):
    def extract(self, video_path: Path, output_dir: Path) -> Path:
        ...


class Extractor:
    def __init__(
        self,
        *,
        media_client: MediaClient | None = None,
        transcriber: Transcriber | None = None,
        audio_extractor: AudioExtractor | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._media_client = media_client
        self._transcriber = transcriber or FasterWhisperTranscriber()
        self._audio_extractor = audio_extractor or FfmpegAudioExtractor()
        self._progress_callback = progress_callback

    def extract(self, url: str, config: ExtractConfig) -> ExtractionResult:
        started = time.monotonic()
        media_client = self._media_client or YtDlpClient(
            http_headers=config.http_headers,
            cookie_files=config.configured_cookie_files,
            cookies_from_browser=config.cookies_from_browser,
        )
        self._emit_progress("probe", "正在读取视频信息", 0.05)
        info = media_client.probe(url)
        self._emit_progress("prepare", "正在准备输出目录", 0.12)
        output_dir = prepare_output_dir(config.output_root, info, url, config.overwrite)
        state = ExtractionState()

        transcript = self._try_downloaded_subtitle(media_client, info, config, state)
        if transcript is None:
            transcript = self._transcribe_media(media_client, url, output_dir, config, state)

        subtitle_path = output_dir / "subtitle.srt"
        paragraph_subtitle_path = output_dir / "subtitle.paragraph.srt"
        transcript_text_path = output_dir / "transcript.txt"
        transcript_json_path = output_dir / "transcript.json"
        meta_path = output_dir / "meta.json"

        self._emit_progress("write", "正在写入字幕和转写文件", 0.92)
        write_srt(transcript, subtitle_path)
        write_paragraph_srt(transcript, paragraph_subtitle_path)
        write_transcript_text(transcript, transcript_text_path)
        write_transcript_json(transcript, transcript_json_path)

        if not config.keep_media:
            self._emit_progress("cleanup", "正在清理媒体文件", 0.95)
            self._remove_media(state)

        meta = self._build_meta(
            url=url,
            info=info,
            config=config,
            state=state,
            output_dir=output_dir,
            transcript=transcript,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(meta, meta_path)
        self._emit_progress("done", "提取完成", 1.0)

        return ExtractionResult(
            output_dir=output_dir,
            source=state.source,
            transcript=transcript,
            meta=meta,
            subtitle_path=subtitle_path,
            paragraph_subtitle_path=paragraph_subtitle_path,
            transcript_text_path=transcript_text_path,
            transcript_json_path=transcript_json_path,
            meta_path=meta_path,
            audio_path=state.audio_path if config.keep_media else None,
            video_path=state.video_path if config.keep_media else None,
        )

    def _try_downloaded_subtitle(
        self,
        media_client: MediaClient,
        info: dict[str, Any],
        config: ExtractConfig,
        state: ExtractionState,
    ) -> Transcript | None:
        self._emit_progress("subtitle", "正在查找可用字幕", 0.18)
        subtitle = select_subtitle(info, config.language)
        if subtitle is None:
            self._emit_progress("subtitle", "未找到可用字幕，准备转写媒体", 0.24)
            return None

        try:
            self._emit_progress("subtitle", "正在下载字幕", 0.3)
            text = media_client.download_subtitle_text(subtitle)
            transcript = subtitle_text_to_transcript(text, subtitle.ext, subtitle.language)
        except Exception as exc:
            state.notes.append(f"Downloaded subtitle could not be used: {exc}")
            self._emit_progress("subtitle", "字幕不可用，准备转写媒体", 0.34)
            return None

        if not transcript.segments:
            state.notes.append("Downloaded subtitle contained no usable segments")
            self._emit_progress("subtitle", "字幕为空，准备转写媒体", 0.34)
            return None

        state.source = "downloaded_subtitle"
        self._emit_progress("subtitle", "已使用视频自带字幕", 0.84)
        return transcript

    def _transcribe_media(
        self,
        media_client: MediaClient,
        url: str,
        output_dir: Path,
        config: ExtractConfig,
        state: ExtractionState,
    ) -> Transcript:
        try:
            self._emit_progress("download_audio", "正在下载音频", 0.38)
            state.audio_path = media_client.download_audio(
                url,
                output_dir,
                progress_callback=self._stage_progress_callback("download_audio", 0.38, 0.64),
            )
            state.source = "audio_transcribe"
        except ExtractionError as exc:
            state.notes.append(f"Audio download failed: {exc}")
            self._emit_progress("download_video", "音频下载失败，正在下载视频", 0.48)
            state.video_path = media_client.download_video(
                url,
                output_dir,
                progress_callback=self._stage_progress_callback("download_video", 0.48, 0.58),
            )
            self._emit_progress("extract_audio", "正在从视频提取音频", 0.58)
            state.audio_path = self._audio_extractor.extract(state.video_path, output_dir)
            state.source = "video_audio_transcribe"

        self._emit_progress("transcribe", "正在本地转写音频", 0.68)
        state.whisper = self._transcriber.transcribe(
            state.audio_path,
            language=config.language,
            model_name=config.model,
            device=config.device,
            compute_type=config.compute_type,
            vad_filter=config.vad_filter,
            progress_callback=self._stage_progress_callback("transcribe", 0.68, 0.88),
        )
        self._emit_progress("transcribe", "转写完成", 0.88)
        return state.whisper.transcript

    def _build_meta(
        self,
        *,
        url: str,
        info: dict[str, Any],
        config: ExtractConfig,
        state: ExtractionState,
        output_dir: Path,
        transcript: Transcript,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        whisper = state.whisper
        return {
            "url": url,
            "resolved_url": info.get("webpage_url") or info.get("original_url"),
            "extracted_at": dt.datetime.now(dt.UTC).isoformat(),
            "source": state.source,
            "language": transcript.language,
            "requested_language": config.language,
            "vad_filter": config.vad_filter,
            "request_headers": sorted(config.http_headers),
            "cookie_file": bool(config.configured_cookie_files),
            "cookies_from_browser": bool(config.cookies_from_browser),
            "video": {
                "id": info.get("id"),
                "title": info.get("title"),
                "extractor": info.get("extractor"),
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "channel": info.get("channel"),
                "upload_date": info.get("upload_date"),
            },
            "files": {
                "subtitle_srt": "subtitle.srt",
                "paragraph_srt": "subtitle.paragraph.srt",
                "transcript_txt": "transcript.txt",
                "transcript_json": "transcript.json",
                "audio": relative_or_name(state.audio_path, output_dir),
                "video": relative_or_name(state.video_path, output_dir),
            },
            "whisper": None
            if whisper is None
            else {
                "model": whisper.model,
                "device": whisper.device,
                "compute_type": whisper.compute_type,
                "elapsed_seconds": round(whisper.elapsed_seconds, 3),
            },
            "segment_count": len(transcript.segments),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "notes": state.notes,
        }

    @staticmethod
    def _remove_media(state: ExtractionState) -> None:
        for path in (state.audio_path, state.video_path):
            if path is not None and path.exists():
                path.unlink()
        state.audio_path = None
        state.video_path = None

    def _emit_progress(self, stage: str, message: str, progress: float | None = None) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(stage, message, progress)

    def _stage_progress_callback(self, stage: str, start: float, end: float) -> StageProgressCallback:
        return stage_progress_callback(self._progress_callback, stage, start, end)
