"""Subtitle and transcript extraction for social video URLs."""

from .core import extract_subtitles
from .models import ExtractConfig, ExtractionResult

__all__ = ["__version__", "ExtractConfig", "ExtractionResult", "extract_subtitles"]

__version__ = "0.1.0"
