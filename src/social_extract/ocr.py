from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import ExtractionError


@dataclass(frozen=True)
class OcrResult:
    texts: list[str]
    scores: list[float]
    boxes: list[Any]

    @property
    def text(self) -> str:
        return "\n".join(text.strip() for text in self.texts if text.strip())

    @property
    def confidence(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


class OcrRecognizer(Protocol):
    def recognize(self, frame_path: Path) -> OcrResult:
        ...


class RapidOcrRecognizer:
    def __init__(self) -> None:
        self._engine: Any | None = None

    def recognize(self, frame_path: Path) -> OcrResult:
        engine = self._get_engine()
        try:
            with _suppress_rapidocr_output():
                raw_result = engine(str(frame_path))
        except Exception as exc:
            raise ExtractionError(f"OCR failed for {frame_path.name}: {exc}") from exc
        return _coerce_rapidocr_result(raw_result)

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        _quiet_rapidocr_logging()
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise ExtractionError(
                "rapidocr and an inference engine such as onnxruntime are required for visual extraction"
            ) from exc
        _quiet_rapidocr_logging()
        with _suppress_rapidocr_output():
            self._engine = RapidOCR()
        return self._engine


def _coerce_rapidocr_result(raw_result: Any) -> OcrResult:
    result = raw_result[0] if isinstance(raw_result, tuple) and raw_result else raw_result

    texts = _as_list(getattr(result, "txts", None))
    scores = [_coerce_float(score) for score in _as_list(getattr(result, "scores", None))]
    boxes = _as_list(getattr(result, "boxes", None))
    if texts or scores or boxes:
        return OcrResult(texts=[str(text) for text in texts], scores=scores, boxes=boxes)

    if isinstance(result, list):
        parsed_texts: list[str] = []
        parsed_scores: list[float] = []
        parsed_boxes: list[Any] = []
        for item in result:
            if not isinstance(item, list | tuple) or len(item) < 2:
                continue
            parsed_boxes.append(item[0])
            text_payload = item[1]
            if isinstance(text_payload, list | tuple) and text_payload:
                parsed_texts.append(str(text_payload[0]))
                if len(text_payload) > 1:
                    parsed_scores.append(_coerce_float(text_payload[1]))
            else:
                parsed_texts.append(str(text_payload))
        return OcrResult(texts=parsed_texts, scores=parsed_scores, boxes=parsed_boxes)

    return OcrResult(texts=[], scores=[], boxes=[])


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if converted is None:
            return []
        if isinstance(converted, list):
            return converted
        if isinstance(converted, tuple):
            return list(converted)
        return [converted]
    try:
        return list(value)
    except TypeError:
        return [value]


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _quiet_rapidocr_logging() -> None:
    for logger_name in ("RapidOCR", "rapidocr"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)


@contextlib.contextmanager
def _suppress_rapidocr_output() -> Iterator[None]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield
