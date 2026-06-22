from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .cuda_runtime import preload_nvidia_cuda_libraries
from .errors import ExtractionError
from .models import Segment, TranscriptionResult, Transcript


TranscriptionProgressCallback = Callable[[float | None, str], None]


class Transcriber(Protocol):
    def transcribe(
        self,
        media_path: Path,
        *,
        language: str,
        model_name: str,
        device: str,
        compute_type: str,
        vad_filter: bool,
        progress_callback: TranscriptionProgressCallback | None = None,
    ) -> TranscriptionResult:
        ...


class FasterWhisperTranscriber:
    def transcribe(
        self,
        media_path: Path,
        *,
        language: str,
        model_name: str,
        device: str,
        compute_type: str,
        vad_filter: bool,
        progress_callback: TranscriptionProgressCallback | None = None,
    ) -> TranscriptionResult:
        if not media_path.exists():
            raise ExtractionError(f"Media file does not exist: {media_path}")

        candidates = _device_candidates(device, compute_type)
        failures: list[str] = []
        for candidate_device, candidate_compute_type in candidates:
            try:
                return self._transcribe_once(
                    media_path,
                    language=language,
                    model_name=model_name,
                    device=candidate_device,
                    compute_type=candidate_compute_type,
                    vad_filter=vad_filter,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                failures.append(f"{candidate_device}/{candidate_compute_type}: {exc}")
                if device != "auto":
                    break

        joined = "; ".join(failures)
        raise ExtractionError(f"Whisper transcription failed: {joined}")

    def _transcribe_once(
        self,
        media_path: Path,
        *,
        language: str,
        model_name: str,
        device: str,
        compute_type: str,
        vad_filter: bool,
        progress_callback: TranscriptionProgressCallback | None = None,
    ) -> TranscriptionResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ExtractionError("faster-whisper is not installed") from exc

        if device == "cuda":
            preload_nvidia_cuda_libraries()

        start = time.monotonic()
        if progress_callback is not None:
            progress_callback(None, "正在加载 Whisper 模型")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        language_arg = None if language == "auto" else language
        if progress_callback is not None:
            progress_callback(0.0, "正在分析音频")
        raw_segments, info = model.transcribe(
            str(media_path),
            language=language_arg,
            vad_filter=vad_filter,
        )
        duration = _audio_duration(info)
        segments: list[Segment] = []
        last_percent: float | None = None
        last_emit = 0.0
        for raw_segment in raw_segments:
            text = raw_segment.text.strip()
            if text:
                segments.append(Segment(start=float(raw_segment.start), end=float(raw_segment.end), text=text))
            if progress_callback is not None:
                percent = _segment_percent(raw_segment, duration)
                now = time.monotonic()
                if _should_emit_transcription_progress(percent, last_percent, now, last_emit):
                    progress_callback(percent, _format_transcription_progress(raw_segment, duration, percent))
                    last_percent = percent
                    last_emit = now

        detected_language = language_arg or getattr(info, "language", "unknown") or "unknown"
        return TranscriptionResult(
            transcript=Transcript(language=detected_language, segments=segments),
            model=model_name,
            device=device,
            compute_type=compute_type,
            elapsed_seconds=time.monotonic() - start,
        )


def _device_candidates(device: str, compute_type: str) -> list[tuple[str, str]]:
    if device == "cpu":
        return [("cpu", "int8" if compute_type == "auto" else compute_type)]
    if device == "cuda":
        return [("cuda", "float16" if compute_type == "auto" else compute_type)]
    return [
        ("cuda", "float16" if compute_type == "auto" else compute_type),
        ("cpu", "int8"),
    ]


def _audio_duration(info: Any) -> float | None:
    for name in ("duration_after_vad", "duration"):
        value = getattr(info, name, None)
        if isinstance(value, int | float) and value > 0:
            return float(value)
    return None


def _segment_percent(segment: Any, duration: float | None) -> float | None:
    if duration is None:
        return None
    end = getattr(segment, "end", None)
    if not isinstance(end, int | float):
        return None
    return max(0.0, min(100.0, float(end) / duration * 100))


def _should_emit_transcription_progress(
    percent: float | None,
    last_percent: float | None,
    now: float,
    last_emit: float,
) -> bool:
    if last_emit == 0.0:
        return True
    if percent is None:
        return now - last_emit >= 1.0
    if last_percent is None:
        return True
    if percent - last_percent >= 1.0:
        return True
    return now - last_emit >= 1.5


def _format_transcription_progress(segment: Any, duration: float | None, percent: float | None) -> str:
    pieces = [f"转写中 {percent:.1f}%" if percent is not None else "转写中"]
    end = getattr(segment, "end", None)
    if isinstance(end, int | float) and duration is not None:
        pieces.append(f"{_format_duration(end)}/{_format_duration(duration)}")
    return " | ".join(pieces)


def _format_duration(value: float) -> str:
    seconds = max(0, int(value))
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    if hour:
        return f"{hour:d}:{minute:02d}:{second:02d}"
    return f"{minute:02d}:{second:02d}"
