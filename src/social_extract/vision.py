from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from .errors import ExtractionError
from .models import ExtractConfig


class HasTextDetector(Protocol):
    def has_text(self, frame_path: Path, config: ExtractConfig) -> bool:
        ...


class PageChangeDetector(Protocol):
    def is_page_change(self, previous_frame: Path, current_frame: Path, config: ExtractConfig) -> bool:
        ...

    def is_stable(self, previous_frame: Path, current_frame: Path, config: ExtractConfig) -> bool:
        ...


class OpenCvHasTextDetector:
    def has_text(self, frame_path: Path, config: ExtractConfig) -> bool:
        frame = _read_bgr(frame_path)
        return has_text(frame, config)


class SsimPageChangeDetector:
    def is_page_change(self, previous_frame: Path, current_frame: Path, config: ExtractConfig) -> bool:
        score = compare_ssim(previous_frame, current_frame)
        return score < config.page_change_ssim_threshold

    def is_stable(self, previous_frame: Path, current_frame: Path, config: ExtractConfig) -> bool:
        score = compare_ssim(previous_frame, current_frame)
        return score > config.stable_ssim_threshold


class TextDedup:
    def should_merge(self, previous_text: str, current_text: str, config: ExtractConfig) -> bool:
        shorter, longer = sorted(
            (previous_text, current_text),
            key=lambda text: len(normalize_text(text)),
        )
        if not normalize_text(shorter) or not normalize_text(longer):
            return False
        return (
            jaccard_similarity(shorter, longer) > config.text_dedup_jaccard_threshold
            or containment_ratio(shorter, longer) > config.text_dedup_containment_threshold
        )


def has_text(frame: Any, config: ExtractConfig) -> bool:
    cv2 = _cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray.var() < config.has_text_variance_min:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))
    skin_ratio = float((skin > 0).mean())
    if skin_ratio > config.has_text_skin_ratio_max:
        return False

    edges = cv2.Canny(gray, 50, 150)
    row_profile = edges.mean(axis=1)
    if row_profile.std() < config.has_text_row_edge_std_min:
        return False

    return True


def compare_ssim(previous_frame: Path, current_frame: Path) -> float:
    cv2 = _cv2()
    ssim = _ssim()
    previous_gray = _read_gray(previous_frame)
    current_gray = _read_gray(current_frame)
    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return float(ssim(previous_gray, current_gray))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(_text_tokens(left))
    right_tokens = set(_text_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def containment_ratio(shorter_text: str, longer_text: str) -> float:
    shorter = normalize_text(shorter_text)
    longer = normalize_text(longer_text)
    if not shorter:
        return 0.0
    if shorter in longer:
        return 1.0
    shorter_tokens = set(_text_tokens(shorter))
    longer_tokens = set(_text_tokens(longer))
    if not shorter_tokens:
        return 0.0
    return len(shorter_tokens & longer_tokens) / len(shorter_tokens)


def _text_tokens(value: str) -> list[str]:
    folded = value.casefold()
    words = re.findall(r"[a-z0-9]+", folded)
    cjk_chars = [char for char in folded if "\u4e00" <= char <= "\u9fff"]
    if words or cjk_chars:
        return words + cjk_chars
    return list(normalize_text(value))


def _read_bgr(frame_path: Path) -> Any:
    cv2 = _cv2()
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise ExtractionError(f"Could not read frame image: {frame_path}")
    return frame


def _read_gray(frame_path: Path) -> Any:
    cv2 = _cv2()
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ExtractionError(f"Could not read frame image: {frame_path}")
    return frame


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ExtractionError("opencv-python-headless is required for visual extraction") from exc
    return cv2


def _ssim() -> Any:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ExtractionError("scikit-image is required for visual page change detection") from exc
    return structural_similarity
