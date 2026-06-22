from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from .cuda_runtime import preload_nvidia_cuda_libraries
from .errors import ExtractionError
from .models import Segment, TranscriptionResult, Transcript


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
    ) -> TranscriptionResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ExtractionError("faster-whisper is not installed") from exc

        if device == "cuda":
            preload_nvidia_cuda_libraries()

        start = time.monotonic()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        language_arg = None if language == "auto" else language
        raw_segments, info = model.transcribe(
            str(media_path),
            language=language_arg,
            vad_filter=vad_filter,
        )
        segments = [
            Segment(start=float(segment.start), end=float(segment.end), text=segment.text.strip())
            for segment in raw_segments
            if segment.text.strip()
        ]
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
