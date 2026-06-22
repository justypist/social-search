from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_extract.transcriber import AUTO_LANGUAGE_DETECTION_SEGMENTS, VAD_PARAMETERS, FasterWhisperTranscriber


@dataclass(frozen=True)
class RawSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Info:
    language: str = "zh"
    duration: float = 2.0


def test_auto_language_uses_vad_multi_segment_detection_and_isolated_context(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls = _install_fake_whisper(monkeypatch)
    audio_path = tmp_path / "audio.m4a"
    audio_path.write_bytes(b"audio")

    result = FasterWhisperTranscriber()._transcribe_once(
        audio_path,
        language="auto",
        model_name="small",
        device="cpu",
        compute_type="int8",
        vad_filter=False,
    )

    transcribe_kwargs = calls[0]["transcribe_kwargs"]
    assert transcribe_kwargs["language"] is None
    assert transcribe_kwargs["vad_filter"] is True
    assert transcribe_kwargs["vad_parameters"] == VAD_PARAMETERS
    assert transcribe_kwargs["condition_on_previous_text"] is False
    assert transcribe_kwargs["language_detection_segments"] == AUTO_LANGUAGE_DETECTION_SEGMENTS
    assert result.transcript.language == "zh"
    assert result.transcript.segments[0].text == "hello"


def test_requested_language_is_forced_and_previous_text_is_not_reused(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls = _install_fake_whisper(monkeypatch)
    audio_path = tmp_path / "audio.m4a"
    audio_path.write_bytes(b"audio")

    result = FasterWhisperTranscriber()._transcribe_once(
        audio_path,
        language="en",
        model_name="small",
        device="cpu",
        compute_type="int8",
        vad_filter=False,
    )

    transcribe_kwargs = calls[0]["transcribe_kwargs"]
    assert transcribe_kwargs["language"] == "en"
    assert transcribe_kwargs["vad_filter"] is False
    assert transcribe_kwargs["vad_parameters"] is None
    assert transcribe_kwargs["condition_on_previous_text"] is False
    assert transcribe_kwargs["language_detection_segments"] == 1
    assert result.transcript.language == "en"


def _install_fake_whisper(monkeypatch: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class FakeWhisperModel:
        def __init__(self, model_name: str, *, device: str, compute_type: str) -> None:
            calls.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                }
            )

        def transcribe(self, audio: str, **kwargs: Any) -> tuple[list[RawSegment], Info]:
            calls[-1]["audio"] = audio
            calls[-1]["transcribe_kwargs"] = kwargs
            return [RawSegment(0.0, 1.0, " hello ")], Info()

    module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return calls
