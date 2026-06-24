from __future__ import annotations

from pathlib import Path

import pytest

from social_extract.errors import ExtractionError
from social_extract.extractor import Extractor
from social_extract.formats import write_json
from social_extract.models import ExtractConfig, Segment, SubtitleRef, TranscriptionResult, Transcript
from social_extract.visual import VisualExtractionResult


class FakeMediaClient:
    def __init__(self, info: dict, *, audio_fails: bool = False) -> None:
        self.info = info
        self.audio_fails = audio_fails
        self.downloaded_audio = False
        self.downloaded_video = False

    def probe(self, url: str) -> dict:
        return self.info

    def download_subtitle_text(self, subtitle: SubtitleRef) -> str:
        return subtitle.data or ""

    def download_audio(self, url: str, output_dir: Path, *, progress_callback=None) -> Path:
        if self.audio_fails:
            raise ExtractionError("no audio")
        if progress_callback is not None:
            progress_callback(50.0, "下载中 50.0%")
        self.downloaded_audio = True
        path = output_dir / "audio.m4a"
        path.write_bytes(b"audio")
        return path

    def download_video(self, url: str, output_dir: Path, *, progress_callback=None) -> Path:
        if progress_callback is not None:
            progress_callback(50.0, "下载中 50.0%")
        self.downloaded_video = True
        path = output_dir / "video.mp4"
        path.write_bytes(b"video")
        return path


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str,
        model_name: str,
        device: str,
        compute_type: str,
        vad_filter: bool,
        progress_callback=None,
    ) -> TranscriptionResult:
        self.calls.append(media_path)
        if progress_callback is not None:
            progress_callback(25.0, "转写中 25.0%")
        return TranscriptionResult(
            transcript=Transcript(language="en", segments=[Segment(0.0, 1.0, "generated text")]),
            model=model_name,
            device="cpu",
            compute_type="int8",
            elapsed_seconds=0.1,
        )


class FakeAudioExtractor:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def extract(self, video_path: Path, output_dir: Path) -> Path:
        self.calls.append(video_path)
        path = output_dir / "audio.wav"
        path.write_bytes(b"wav")
        return path


class FakeVisualExtractor:
    def __init__(self, *, pages: list[dict] | None = None, error: Exception | None = None) -> None:
        self.pages = [] if pages is None else pages
        self.error = error
        self.calls: list[Path] = []
        self.configs: list[ExtractConfig] = []

    def extract(self, video_path: Path, output_dir: Path, config: ExtractConfig, *, progress_callback=None) -> VisualExtractionResult:
        self.configs.append(config)
        self.calls.append(video_path)
        if self.error is not None:
            raise self.error
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, _page in enumerate(self.pages):
            (frames_dir / f"page_{index:04d}.jpg").write_bytes(b"frame")
        pages_json_path = output_dir / "pages.json"
        payload = {
            "frame_fps": 1.0,
            "pages": self.pages,
            "non_text_segments": [],
            "stats": {
                "sampled_frames": 1,
                "text_frames": 1,
                "ocr_frames": len(self.pages),
                "pages": len(self.pages),
            },
        }
        write_json(payload, pages_json_path)
        if progress_callback is not None:
            progress_callback("visual_write", "写入画面文字", 0.9)
        return VisualExtractionResult(pages_json_path=pages_json_path, frames_dir=frames_dir, payload=payload)


def test_downloaded_subtitle_is_used_before_audio(tmp_path: Path) -> None:
    client = FakeMediaClient(
        {
            "id": "abc123",
            "title": "Example",
            "subtitles": {
                "en": [
                    {
                        "ext": "vtt",
                        "data": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n",
                    }
                ]
            },
        }
    )
    transcriber = FakeTranscriber()
    result = Extractor(media_client=client, transcriber=transcriber).extract(
        "https://example.test/video",
        ExtractConfig(output_root=tmp_path, language="en"),
    )

    assert result.source == "downloaded_subtitle"
    assert result.subtitle_path.read_text(encoding="utf-8").startswith("1\n00:00:00,000")
    assert result.paragraph_subtitle_path.name == "subtitle.paragraph.srt"
    assert result.paragraph_subtitle_path.read_text(encoding="utf-8").startswith("1\n00:00:00,000")
    assert result.transcript_text_path.read_text(encoding="utf-8") == "hello\n"
    assert result.meta["files"]["paragraph_srt"] == "subtitle.paragraph.srt"
    assert not client.downloaded_audio
    assert not client.downloaded_video
    assert "pages_json" not in result.meta["files"]
    assert transcriber.calls == []


def test_audio_fallback_transcribes_and_keeps_audio(tmp_path: Path) -> None:
    client = FakeMediaClient({"id": "no-subs", "title": "No Subs"})
    transcriber = FakeTranscriber()
    result = Extractor(media_client=client, transcriber=transcriber).extract(
        "https://example.test/video",
        ExtractConfig(output_root=tmp_path),
    )

    assert result.source == "audio_transcribe"
    assert result.audio_path is not None
    assert result.audio_path.name == "audio.m4a"
    assert transcriber.calls == [result.audio_path]
    assert result.meta["files"]["audio"] == "audio.m4a"


def test_download_and_transcription_progress_is_forwarded(tmp_path: Path) -> None:
    events: list[tuple[str, str, float | None]] = []
    client = FakeMediaClient({"id": "progress", "title": "Progress"})
    extractor = Extractor(
        media_client=client,
        transcriber=FakeTranscriber(),
        progress_callback=lambda stage, message, progress: events.append((stage, message, progress)),
    )

    extractor.extract("https://example.test/video", ExtractConfig(output_root=tmp_path))

    download_event = next(event for event in events if event[:2] == ("download_audio", "下载中 50.0%"))
    transcribe_event = next(event for event in events if event[:2] == ("transcribe", "转写中 25.0%"))
    assert download_event[2] == pytest.approx(0.51)
    assert transcribe_event[2] == pytest.approx(0.73)


def test_video_fallback_when_audio_download_fails(tmp_path: Path) -> None:
    client = FakeMediaClient({"id": "video-fallback"}, audio_fails=True)
    transcriber = FakeTranscriber()
    result = Extractor(
        media_client=client,
        transcriber=transcriber,
        audio_extractor=FakeAudioExtractor(),
    ).extract(
        "https://example.test/video",
        ExtractConfig(output_root=tmp_path),
    )

    assert result.source == "video_audio_transcribe"
    assert result.video_path is not None
    assert result.video_path.name == "video.mp4"
    assert result.audio_path is not None
    assert result.audio_path.name == "audio.wav"
    assert result.meta["files"]["video"] == "video.mp4"


def test_extract_visual_with_downloaded_subtitle_still_downloads_video_and_pages(tmp_path: Path) -> None:
    client = FakeMediaClient(
        {
            "id": "visual-subtitle",
            "title": "Visual Subtitle",
            "subtitles": {
                "en": [
                    {
                        "ext": "vtt",
                        "data": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n",
                    }
                ]
            },
        }
    )
    visual = FakeVisualExtractor(
        pages=[
            {
                "page_index": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Slide title",
                "frame_path": "frames/page_0000.jpg",
                "confidence": 0.9,
            }
        ]
    )
    transcriber = FakeTranscriber()

    result = Extractor(media_client=client, transcriber=transcriber, visual_extractor=visual).extract(
        "https://example.test/video",
        ExtractConfig(output_root=tmp_path, language="en", extract_visual=True),
    )

    assert result.source == "downloaded_subtitle"
    assert client.downloaded_video
    assert not client.downloaded_audio
    assert transcriber.calls == []
    assert visual.calls == [result.video_path]
    assert result.pages_json_path is not None
    assert result.pages_json_path.name == "pages.json"
    assert result.frames_dir is not None
    assert result.meta["files"]["pages_json"] == "pages.json"
    assert result.meta["files"]["visual_frames"] == "frames"


def test_extract_visual_reuses_video_for_transcription_without_audio_download(tmp_path: Path) -> None:
    client = FakeMediaClient({"id": "visual-no-subs", "title": "Visual No Subs"})
    transcriber = FakeTranscriber()
    audio_extractor = FakeAudioExtractor()
    visual = FakeVisualExtractor()

    result = Extractor(
        media_client=client,
        transcriber=transcriber,
        audio_extractor=audio_extractor,
        visual_extractor=visual,
    ).extract(
        "https://example.test/video",
        ExtractConfig(output_root=tmp_path, extract_visual=True),
    )

    assert result.source == "video_audio_transcribe"
    assert client.downloaded_video
    assert not client.downloaded_audio
    assert result.video_path is not None
    assert audio_extractor.calls == [result.video_path]
    assert result.audio_path is not None
    assert transcriber.calls == [result.audio_path]
    assert visual.calls == [result.video_path]


def test_keep_media_false_removes_audio_video_but_keeps_visual_outputs(tmp_path: Path) -> None:
    client = FakeMediaClient({"id": "visual-cleanup"})
    result = Extractor(
        media_client=client,
        transcriber=FakeTranscriber(),
        audio_extractor=FakeAudioExtractor(),
        visual_extractor=FakeVisualExtractor(
            pages=[
                {
                    "page_index": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Slide title",
                    "frame_path": "frames/page_0000.jpg",
                    "confidence": 0.9,
                }
            ]
        ),
    ).extract(
        "https://example.test/video",
        ExtractConfig(output_root=tmp_path, extract_visual=True, keep_media=False),
    )

    assert result.audio_path is None
    assert result.video_path is None
    assert not (result.output_dir / "audio.wav").exists()
    assert not (result.output_dir / "video.mp4").exists()
    assert result.pages_json_path is not None
    assert result.pages_json_path.exists()
    assert result.frames_dir is not None
    assert (result.frames_dir / "page_0000.jpg").exists()
    assert result.meta["files"]["pages_json"] == "pages.json"
    assert result.meta["files"]["visual_frames"] == "frames"


def test_extract_visual_empty_pages_is_success(tmp_path: Path) -> None:
    result = Extractor(
        media_client=FakeMediaClient({"id": "visual-empty"}),
        transcriber=FakeTranscriber(),
        audio_extractor=FakeAudioExtractor(),
        visual_extractor=FakeVisualExtractor(pages=[]),
    ).extract(
        "https://example.test/video",
        ExtractConfig(output_root=tmp_path, extract_visual=True),
    )

    assert result.pages_json_path is not None
    assert result.pages_json_path.exists()
    assert result.meta["files"]["pages_json"] == "pages.json"


def test_describe_visual_implies_visual_extraction(tmp_path: Path) -> None:
    client = FakeMediaClient({"id": "describe-visual", "title": "Describe Visual"})
    visual = FakeVisualExtractor(
        pages=[
            {
                "page_index": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Slide title",
                "frame_path": "frames/page_0000.jpg",
                "confidence": 0.9,
                "visual_summary": "页面包含图表。",
                "visual_keywords": ["图表"],
            }
        ]
    )

    result = Extractor(
        media_client=client,
        transcriber=FakeTranscriber(),
        audio_extractor=FakeAudioExtractor(),
        visual_extractor=visual,
    ).extract(
        "https://example.test/video",
        ExtractConfig(output_root=tmp_path, describe_visual=True),
    )

    assert client.downloaded_video
    assert not client.downloaded_audio
    assert visual.calls == [result.video_path]
    assert visual.configs[0].describe_visual is True
    assert result.meta["extract_visual"] is True
    assert result.meta["describe_visual"] is True
    assert result.meta["files"]["pages_json"] == "pages.json"


def test_visual_extraction_error_fails_task(tmp_path: Path) -> None:
    extractor = Extractor(
        media_client=FakeMediaClient({"id": "visual-error"}),
        transcriber=FakeTranscriber(),
        audio_extractor=FakeAudioExtractor(),
        visual_extractor=FakeVisualExtractor(error=ExtractionError("ocr failed")),
    )

    with pytest.raises(ExtractionError, match="ocr failed"):
        extractor.extract(
            "https://example.test/video",
            ExtractConfig(output_root=tmp_path, extract_visual=True),
        )


def test_existing_output_directory_requires_overwrite(tmp_path: Path) -> None:
    (tmp_path / "abc").mkdir()
    extractor = Extractor(media_client=FakeMediaClient({"id": "abc"}), transcriber=FakeTranscriber())

    with pytest.raises(ExtractionError, match="already exists"):
        extractor.extract("https://example.test/video", ExtractConfig(output_root=tmp_path))
