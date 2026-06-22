from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Language = Literal["auto", "zh", "en"]
Device = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    language: str
    segments: list[Segment]


@dataclass(frozen=True)
class SubtitleRef:
    language: str
    ext: str
    url: str | None = None
    data: str | None = None
    source: Literal["manual", "automatic"] = "manual"


@dataclass(frozen=True)
class ExtractConfig:
    output_root: Path
    language: Language = "auto"
    model: str = "medium"
    device: Device = "auto"
    compute_type: str = "auto"
    vad_filter: bool = False
    keep_media: bool = True
    overwrite: bool = False
    http_headers: dict[str, str] = field(default_factory=dict)
    cookie_file: Path | None = None
    cookie_files: tuple[Path, ...] = ()
    cookies_from_browser: str | None = None

    @property
    def configured_cookie_files(self) -> tuple[Path, ...]:
        if self.cookie_file is None:
            return self.cookie_files
        if self.cookie_file in self.cookie_files:
            return self.cookie_files
        return (self.cookie_file, *self.cookie_files)


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: Transcript
    model: str
    device: str
    compute_type: str
    elapsed_seconds: float


@dataclass(frozen=True)
class ExtractionResult:
    output_dir: Path
    source: str
    transcript: Transcript
    meta: dict[str, Any]
    subtitle_path: Path
    paragraph_subtitle_path: Path
    transcript_text_path: Path
    transcript_json_path: Path
    meta_path: Path
    audio_path: Path | None = None
    video_path: Path | None = None


@dataclass
class ExtractionState:
    source: str = ""
    audio_path: Path | None = None
    video_path: Path | None = None
    whisper: TranscriptionResult | None = None
    notes: list[str] = field(default_factory=list)
