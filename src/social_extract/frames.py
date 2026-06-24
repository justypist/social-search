from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .errors import ExtractionError
from .formats import write_json
from .paths import relative_or_name


@dataclass(frozen=True)
class FrameRef:
    index: int
    timestamp: float
    path: Path


@dataclass(frozen=True)
class FrameExtractionResult:
    fps: float
    frames_dir: Path
    frames_json_path: Path
    frames: list[FrameRef]


class FrameExtractor(Protocol):
    def extract(self, video_path: Path, output_dir: Path, *, fps: float) -> FrameExtractionResult:
        ...


class FfmpegFrameExtractor:
    def extract(self, video_path: Path, output_dir: Path, *, fps: float) -> FrameExtractionResult:
        if fps <= 0:
            raise ExtractionError("frame_fps must be greater than 0")

        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames_json_path = output_dir / "frames.json"
        output_pattern = frames_dir / "%06d.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps:g}",
            "-q:v",
            "2",
            str(output_pattern),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise ExtractionError("ffmpeg is required for visual extraction but was not found") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip().splitlines()
            detail = stderr[-1] if stderr else "unknown ffmpeg error"
            raise ExtractionError(f"ffmpeg failed to extract frames: {detail}")

        frame_paths = sorted(path for path in frames_dir.glob("*.jpg") if path.is_file())
        frames = [
            FrameRef(index=index, timestamp=round(index / fps, 3), path=path)
            for index, path in enumerate(frame_paths)
        ]
        write_json(
            {
                "fps": fps,
                "frames": [
                    {
                        "index": frame.index,
                        "timestamp": frame.timestamp,
                        "path": relative_or_name(frame.path, output_dir),
                    }
                    for frame in frames
                ],
            },
            frames_json_path,
        )
        return FrameExtractionResult(
            fps=fps,
            frames_dir=frames_dir,
            frames_json_path=frames_json_path,
            frames=frames,
        )
