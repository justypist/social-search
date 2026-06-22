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
    write_srt,
    write_transcript_json,
    write_transcript_text,
)
from .models import ExtractConfig, ExtractionResult, ExtractionState, SubtitleRef, Transcript
from .paths import prepare_output_dir, relative_or_name
from .subtitles import select_subtitle
from .transcriber import FasterWhisperTranscriber, Transcriber
from .yt_dlp_client import YtDlpClient


class MediaClient(Protocol):
    def probe(self, url: str) -> dict[str, Any]:
        ...

    def download_subtitle_text(self, subtitle: SubtitleRef) -> str:
        ...

    def download_audio(self, url: str, output_dir: Path) -> Path:
        ...

    def download_video(self, url: str, output_dir: Path) -> Path:
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
    ) -> None:
        self._media_client = media_client
        self._transcriber = transcriber or FasterWhisperTranscriber()
        self._audio_extractor = audio_extractor or FfmpegAudioExtractor()

    def extract(self, url: str, config: ExtractConfig) -> ExtractionResult:
        started = time.monotonic()
        media_client = self._media_client or YtDlpClient(http_headers=config.http_headers)
        info = media_client.probe(url)
        output_dir = prepare_output_dir(config.output_root, info, url, config.overwrite)
        state = ExtractionState()

        transcript = self._try_downloaded_subtitle(media_client, info, config, state)
        if transcript is None:
            transcript = self._transcribe_media(media_client, url, output_dir, config, state)

        subtitle_path = output_dir / "subtitle.srt"
        transcript_text_path = output_dir / "transcript.txt"
        transcript_json_path = output_dir / "transcript.json"
        meta_path = output_dir / "meta.json"

        write_srt(transcript, subtitle_path)
        write_transcript_text(transcript, transcript_text_path)
        write_transcript_json(transcript, transcript_json_path)

        if not config.keep_media:
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

        return ExtractionResult(
            output_dir=output_dir,
            source=state.source,
            transcript=transcript,
            meta=meta,
            subtitle_path=subtitle_path,
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
        subtitle = select_subtitle(info, config.language)
        if subtitle is None:
            return None

        try:
            text = media_client.download_subtitle_text(subtitle)
            transcript = subtitle_text_to_transcript(text, subtitle.ext, subtitle.language)
        except Exception as exc:
            state.notes.append(f"Downloaded subtitle could not be used: {exc}")
            return None

        if not transcript.segments:
            state.notes.append("Downloaded subtitle contained no usable segments")
            return None

        state.source = "downloaded_subtitle"
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
            state.audio_path = media_client.download_audio(url, output_dir)
            state.source = "audio_transcribe"
        except ExtractionError as exc:
            state.notes.append(f"Audio download failed: {exc}")
            state.video_path = media_client.download_video(url, output_dir)
            state.audio_path = self._audio_extractor.extract(state.video_path, output_dir)
            state.source = "video_audio_transcribe"

        state.whisper = self._transcriber.transcribe(
            state.audio_path,
            language=config.language,
            model_name=config.model,
            device=config.device,
            compute_type=config.compute_type,
            vad_filter=config.vad_filter,
        )
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
