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
    extract_visual: bool = False
    describe_visual: bool = False
    frame_fps: float = 1.0
    has_text_variance_min: float = 20.0
    has_text_skin_ratio_max: float = 0.15
    has_text_row_edge_std_min: float = 1.5
    has_text_min_consecutive_frames: int = 2
    page_change_ssim_threshold: float = 0.85
    page_change_phash_threshold: int = 20
    page_change_phash_delta: int = 12
    stable_frame_count: int = 2
    stable_ssim_threshold: float = 0.95
    visual_text_min_chars: int = 10
    text_dedup_jaccard_threshold: float = 0.9
    text_dedup_containment_threshold: float = 0.85
    visual_description_optional: bool = True
    max_visual_describe_pages: int = 120
    visual_description_provider: str | None = "openai"
    visual_description_model: str | None = None

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
    pages_json_path: Path | None = None
    frames_dir: Path | None = None


@dataclass
class ExtractionState:
    source: str = ""
    audio_path: Path | None = None
    video_path: Path | None = None
    pages_json_path: Path | None = None
    frames_dir: Path | None = None
    visual_description: dict[str, Any] | None = None
    whisper: TranscriptionResult | None = None
    notes: list[str] = field(default_factory=list)
