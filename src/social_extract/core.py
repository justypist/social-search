from __future__ import annotations

from .extractor import Extractor
from .models import ExtractConfig, ExtractionResult
from .progress import ProgressCallback


def extract_subtitles(
    url: str,
    config: ExtractConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> ExtractionResult:
    return Extractor(progress_callback=progress_callback).extract(url, config)
