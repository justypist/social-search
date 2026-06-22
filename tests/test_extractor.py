from __future__ import annotations

from pathlib import Path

import pytest

from social_extract.errors import ExtractionError
from social_extract.extractor import Extractor
from social_extract.models import ExtractConfig, Segment, SubtitleRef, TranscriptionResult, Transcript


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

    def download_audio(self, url: str, output_dir: Path) -> Path:
        if self.audio_fails:
            raise ExtractionError("no audio")
        self.downloaded_audio = True
        path = output_dir / "audio.m4a"
        path.write_bytes(b"audio")
        return path

    def download_video(self, url: str, output_dir: Path) -> Path:
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
    ) -> TranscriptionResult:
        self.calls.append(media_path)
        return TranscriptionResult(
            transcript=Transcript(language="en", segments=[Segment(0.0, 1.0, "generated text")]),
            model=model_name,
            device="cpu",
            compute_type="int8",
            elapsed_seconds=0.1,
        )


class FakeAudioExtractor:
    def extract(self, video_path: Path, output_dir: Path) -> Path:
        path = output_dir / "audio.wav"
        path.write_bytes(b"wav")
        return path


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
    assert result.transcript_text_path.read_text(encoding="utf-8") == "hello\n"
    assert not client.downloaded_audio
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


def test_existing_output_directory_requires_overwrite(tmp_path: Path) -> None:
    (tmp_path / "abc").mkdir()
    extractor = Extractor(media_client=FakeMediaClient({"id": "abc"}), transcriber=FakeTranscriber())

    with pytest.raises(ExtractionError, match="already exists"):
        extractor.extract("https://example.test/video", ExtractConfig(output_root=tmp_path))
