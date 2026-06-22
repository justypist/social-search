from __future__ import annotations

import subprocess
from pathlib import Path

from .errors import ExtractionError


class FfmpegAudioExtractor:
    def extract(self, video_path: Path, output_dir: Path) -> Path:
        audio_path = output_dir / "audio.wav"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise ExtractionError("ffmpeg is required for video fallback but was not found") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip().splitlines()
            detail = stderr[-1] if stderr else "unknown ffmpeg error"
            raise ExtractionError(f"ffmpeg failed to extract audio: {detail}")
        return audio_path
