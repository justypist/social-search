from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .formats import write_json
from .frames import FfmpegFrameExtractor, FrameExtractionResult, FrameExtractor, FrameRef
from .models import ExtractConfig
from .ocr import OcrRecognizer, RapidOcrRecognizer
from .paths import relative_or_name
from .progress import ProgressCallback
from .vision import (
    HasTextDetector,
    OpenCvHasTextDetector,
    PageChangeDetector,
    SsimPageChangeDetector,
    TextDedup,
    normalize_text,
)


@dataclass(frozen=True)
class VisualExtractionResult:
    pages_json_path: Path
    frames_dir: Path
    payload: dict


@dataclass(frozen=True)
class _RecognizedPage:
    start: float
    end: float
    text: str
    source_frame_path: Path
    confidence: float


@dataclass(frozen=True)
class _PageCandidate:
    start: float
    end: float
    frame: FrameRef


class VisualExtractor:
    def __init__(
        self,
        *,
        frame_extractor: FrameExtractor | None = None,
        ocr_recognizer: OcrRecognizer | None = None,
        has_text_detector: HasTextDetector | None = None,
        page_change_detector: PageChangeDetector | None = None,
        text_dedup: TextDedup | None = None,
    ) -> None:
        self._frame_extractor = frame_extractor or FfmpegFrameExtractor()
        self._ocr_recognizer = ocr_recognizer or RapidOcrRecognizer()
        self._has_text_detector = has_text_detector or OpenCvHasTextDetector()
        self._page_change_detector = page_change_detector or SsimPageChangeDetector()
        self._text_dedup = text_dedup or TextDedup()

    def extract(
        self,
        video_path: Path,
        output_dir: Path,
        config: ExtractConfig,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> VisualExtractionResult:
        _emit(progress_callback, "visual_prepare", "准备视觉提取", 0.72)
        _emit(progress_callback, "visual_frames", "抽取视频帧", 0.74)
        frame_result = self._frame_extractor.extract(video_path, output_dir, fps=config.frame_fps)

        _emit(progress_callback, "visual_detect", "检测文字画面 0.0%", 0.8)
        text_flags = self._detect_text_frames(frame_result.frames, config, progress_callback)
        text_flags = _apply_min_consecutive_text_frames(text_flags, config.has_text_min_consecutive_frames)
        non_text_segments = _non_text_segments(frame_result.frames, text_flags, frame_result.fps)
        candidates = self._select_representative_frames(frame_result.frames, text_flags, frame_result.fps, config)

        _emit(progress_callback, "visual_ocr", "识别画面文字 0.0%", 0.86)
        pages = self._recognize_pages(candidates, config, progress_callback)
        pages = self._deduplicate_pages(pages, config)

        _emit(progress_callback, "visual_write", "写入画面文字", 0.9)
        payload_pages = self._materialize_page_frames(pages, frame_result.frames_dir, output_dir)
        payload = {
            "frame_fps": frame_result.fps,
            "pages": payload_pages,
            "non_text_segments": non_text_segments,
            "stats": {
                "sampled_frames": len(frame_result.frames),
                "text_frames": sum(1 for flag in text_flags if flag),
                "ocr_frames": len(candidates),
                "pages": len(payload_pages),
            },
        }
        pages_json_path = output_dir / "pages.json"
        write_json(payload, pages_json_path)
        return VisualExtractionResult(
            pages_json_path=pages_json_path,
            frames_dir=frame_result.frames_dir,
            payload=payload,
        )

    def _detect_text_frames(
        self,
        frames: list[FrameRef],
        config: ExtractConfig,
        progress_callback: ProgressCallback | None,
    ) -> list[bool]:
        flags: list[bool] = []
        total = len(frames)
        if total == 0:
            _emit(progress_callback, "visual_detect", "检测文字画面 100.0%", 0.84)
            return flags
        for index, frame in enumerate(frames):
            flags.append(self._has_text_detector.has_text(frame.path, config))
            if total and (index + 1 == total or (index + 1) % 25 == 0):
                percent = (index + 1) / total * 100
                progress = 0.8 + 0.04 * (index + 1) / total
                _emit(progress_callback, "visual_detect", f"检测文字画面 {percent:.1f}%", progress)
        return flags

    def _select_representative_frames(
        self,
        frames: list[FrameRef],
        text_flags: list[bool],
        fps: float,
        config: ExtractConfig,
    ) -> list[_PageCandidate]:
        candidates: list[_PageCandidate] = []
        duration = _frame_duration(fps)
        for start, end in _true_runs(text_flags):
            run = frames[start:end]
            if not run:
                continue
            run_candidates: list[int] = [0]
            index = 1
            while index < len(run):
                if self._page_change_detector.is_page_change(run[index - 1].path, run[index].path, config):
                    stable_index = self._find_stable_index(run, index, config)
                    if stable_index != run_candidates[-1]:
                        run_candidates.append(stable_index)
                    index = max(index + 1, stable_index + 1)
                    continue
                index += 1
            for candidate_index, run_index in enumerate(run_candidates):
                next_index = run_candidates[candidate_index + 1] if candidate_index + 1 < len(run_candidates) else None
                page_start = run[0].timestamp if candidate_index == 0 else run[run_index].timestamp
                page_end = run[next_index].timestamp if next_index is not None else run[-1].timestamp + duration
                candidates.append(
                    _PageCandidate(
                        start=page_start,
                        end=max(page_start, page_end),
                        frame=run[run_index],
                    )
                )
        return candidates

    def _find_stable_index(self, run: list[FrameRef], start_index: int, config: ExtractConfig) -> int:
        required_frames = max(1, config.stable_frame_count)
        if required_frames == 1:
            return start_index
        last_start = len(run) - required_frames
        for index in range(start_index, max(start_index, last_start) + 1):
            window = run[index : index + required_frames]
            if len(window) < required_frames:
                break
            stable = all(
                self._page_change_detector.is_stable(window[offset - 1].path, window[offset].path, config)
                for offset in range(1, len(window))
            )
            if stable:
                return index + (required_frames - 1) // 2
        return start_index

    def _recognize_pages(
        self,
        candidates: list[_PageCandidate],
        config: ExtractConfig,
        progress_callback: ProgressCallback | None,
    ) -> list[_RecognizedPage]:
        pages: list[_RecognizedPage] = []
        total = len(candidates)
        if total == 0:
            _emit(progress_callback, "visual_ocr", "识别画面文字 100.0%", 0.89)
            return pages
        for index, candidate in enumerate(candidates):
            ocr_result = self._ocr_recognizer.recognize(candidate.frame.path)
            text = ocr_result.text.strip()
            if len(normalize_text(text)) >= config.visual_text_min_chars:
                pages.append(
                    _RecognizedPage(
                        start=candidate.start,
                        end=candidate.end,
                        text=text,
                        source_frame_path=candidate.frame.path,
                        confidence=ocr_result.confidence,
                    )
                )
            if total:
                percent = (index + 1) / total * 100
                progress = 0.86 + 0.03 * (index + 1) / total
                _emit(progress_callback, "visual_ocr", f"识别画面文字 {percent:.1f}%", progress)
        return pages

    def _deduplicate_pages(self, pages: list[_RecognizedPage], config: ExtractConfig) -> list[_RecognizedPage]:
        merged: list[_RecognizedPage] = []
        for page in pages:
            if not merged or not self._text_dedup.should_merge(merged[-1].text, page.text, config):
                merged.append(page)
                continue
            merged[-1] = _merge_pages(merged[-1], page)
        return merged

    @staticmethod
    def _materialize_page_frames(
        pages: list[_RecognizedPage],
        frames_dir: Path,
        output_dir: Path,
    ) -> list[dict]:
        payload: list[dict] = []
        for index, page in enumerate(pages):
            suffix = page.source_frame_path.suffix or ".jpg"
            target = frames_dir / f"page_{index:04d}{suffix}"
            if page.source_frame_path.resolve() != target.resolve():
                shutil.copyfile(page.source_frame_path, target)
            payload.append(
                {
                    "page_index": index,
                    "start": round(page.start, 3),
                    "end": round(page.end, 3),
                    "text": page.text,
                    "frame_path": relative_or_name(target, output_dir),
                    "confidence": round(page.confidence, 3),
                }
            )
        return payload


def _merge_pages(previous: _RecognizedPage, current: _RecognizedPage) -> _RecognizedPage:
    previous_score = (previous.confidence, len(normalize_text(previous.text)))
    current_score = (current.confidence, len(normalize_text(current.text)))
    selected = current if current_score >= previous_score else previous
    return _RecognizedPage(
        start=previous.start,
        end=max(previous.end, current.end),
        text=selected.text,
        source_frame_path=selected.source_frame_path,
        confidence=selected.confidence,
    )


def _non_text_segments(frames: list[FrameRef], text_flags: list[bool], fps: float) -> list[dict]:
    segments: list[dict] = []
    duration = _frame_duration(fps)
    start: float | None = None
    end: float | None = None
    for frame, has_text in zip(frames, text_flags, strict=True):
        if not has_text:
            start = frame.timestamp if start is None else start
            end = frame.timestamp + duration
            continue
        if start is not None and end is not None:
            segments.append({"start": round(start, 3), "end": round(end, 3), "reason": "no_text"})
        start = None
        end = None
    if start is not None and end is not None:
        segments.append({"start": round(start, 3), "end": round(end, 3), "reason": "no_text"})
    return segments


def _apply_min_consecutive_text_frames(flags: list[bool], minimum: int) -> list[bool]:
    required = max(1, minimum)
    confirmed = [False] * len(flags)
    for start, end in _true_runs(flags):
        if end - start < required:
            continue
        for index in range(start, end):
            confirmed[index] = True
    return confirmed


def _true_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
            continue
        if not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _frame_duration(fps: float) -> float:
    return 1 / fps if fps > 0 else 0.0


def _emit(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
    progress: float | None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, message, progress)
