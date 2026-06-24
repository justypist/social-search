from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import ExtractionError
from .gemini_client import GeminiConfigurationError, create_gemini_client
from .models import ExtractConfig
from .progress import ProgressCallback


VISUAL_DESCRIPTION_PROMPT_VERSION = "visual-description-v1"
VISUAL_DESCRIPTION_PROMPT_TEMPLATE = """You describe visual content in one representative video frame.
Focus on non-OCR visual information: charts, diagrams, screenshots, photos, flows, architecture relations, and important visual structure.
Do not repeat ordinary OCR text unless it is needed to explain the image, chart, or diagram.

Page time range: {start:.3f}s to {end:.3f}s
OCR text:
{ocr_text}

Return exactly one valid JSON object with this schema:
{{
  "visual_summary": "short searchable description",
  "visual_keywords": ["5 to 12 concise keywords"],
  "visual_content_type": "none|photo|chart|table|flowchart|architecture_diagram|screenshot|illustration|mixed|unknown",
  "visual_confidence": 0.0
}}

If the page is plain text with no meaningful visual content, use visual_content_type "none", an empty summary, empty keywords, and low confidence.
If details are unclear, lower confidence instead of inventing facts.
"""

VISUAL_CONTENT_TYPES = {
    "none",
    "photo",
    "chart",
    "table",
    "flowchart",
    "architecture_diagram",
    "screenshot",
    "illustration",
    "mixed",
    "unknown",
}


@dataclass(frozen=True)
class VisualDescriptionResult:
    pages: list[dict[str, Any]]
    meta: dict[str, Any]


class VisualDescriber(Protocol):
    def describe_pages(
        self,
        pages: list[dict[str, Any]],
        output_dir: Path,
        config: ExtractConfig,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> VisualDescriptionResult:
        ...


class GeminiVisualDescriber:
    provider = "gemini"

    def __init__(
        self,
        *,
        cookie_files: tuple[Path, ...] | None = None,
        client: Any | None = None,
        model: str | None = None,
        prompt_template: str = VISUAL_DESCRIPTION_PROMPT_TEMPLATE,
        generate_kwargs: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._cookie_files = cookie_files
        self._client = client
        self._model = model
        self._prompt_template = prompt_template
        self._generate_kwargs = {"temporary": True, **(generate_kwargs or {})}
        self._init_kwargs = {"auto_refresh": False, **(init_kwargs or {})}
        self._initialized = False
        self._runner: asyncio.Runner | None = None

    def describe_pages(
        self,
        pages: list[dict[str, Any]],
        output_dir: Path,
        config: ExtractConfig,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> VisualDescriptionResult:
        described_pages = [dict(page) for page in pages]
        limit = max(0, config.max_visual_describe_pages)
        model = self._model or config.visual_description_model
        model_label = model or "default"
        cache_path = output_dir / "visual_description_cache.json"
        cache = _load_cache(cache_path)
        meta = {
            "enabled": True,
            "provider": self.provider,
            "model": model_label,
            "candidate_pages": len(described_pages),
            "described_pages": 0,
            "cache_hits": 0,
            "skipped_pages": 0,
            "failed_pages": 0,
        }

        total = len(described_pages)
        if total == 0:
            _emit(progress_callback, "visual_describe", "总结关键帧 100.0%", 0.92)
            _save_cache(cache_path, cache)
            return VisualDescriptionResult(pages=described_pages, meta=meta)

        for index, page in enumerate(described_pages):
            if index >= limit:
                meta["skipped_pages"] += 1
                _mark_skipped(page, self.provider, model_label, "max_visual_describe_pages")
                _emit_page_progress(progress_callback, index, total)
                continue

            try:
                frame_path = _resolve_page_frame(output_dir, page)
                image_hash = _sha256_file(frame_path)
                cache_key = _cache_key(model_label, image_hash)
                cached = cache.get(cache_key)
                if isinstance(cached, dict):
                    description = _coerce_description(cached)
                    page.update(description)
                    page["visual_cache_hit"] = True
                    meta["cache_hits"] += 1
                else:
                    prompt = self._prompt(page)
                    description = self._run(self._describe_async(frame_path, prompt, model))
                    cache[cache_key] = description
                    page.update(description)
                    page["visual_cache_hit"] = False

                page["visual_provider"] = self.provider
                page["visual_model"] = model_label
                meta["described_pages"] += 1
            except Exception as exc:
                meta["failed_pages"] += 1
                if not config.visual_description_optional:
                    raise ExtractionError(f"Visual description failed for page {index}: {exc}") from exc
                _mark_failed(page, self.provider, model_label, str(exc))

            _emit_page_progress(progress_callback, index, total)

        _save_cache(cache_path, cache)
        return VisualDescriptionResult(pages=described_pages, meta=meta)

    def close(self) -> None:
        if self._runner is None:
            return
        try:
            self._runner.run(self._close_async())
        finally:
            self._runner.close()
            self._runner = None
            self._initialized = False

    def _prompt(self, page: dict[str, Any]) -> str:
        return self._prompt_template.format(
            start=_coerce_float(page.get("start")),
            end=_coerce_float(page.get("end")),
            ocr_text=str(page.get("text") or "").strip() or "(empty)",
        )

    def _run(self, coroutine: Any) -> Any:
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(coroutine)

    async def _describe_async(self, frame_path: Path, prompt: str, model: str | None) -> dict[str, Any]:
        client = await self._get_client()
        kwargs = dict(self._generate_kwargs)
        if model:
            kwargs["model"] = model
        output = client.generate_content(prompt, files=[frame_path], **kwargs)
        if hasattr(output, "__await__"):
            output = await output
        output_text = getattr(output, "text", None)
        return _coerce_description(str(output_text) if output_text is not None else str(output))

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                self._client = create_gemini_client(cookie_files=self._cookie_files)
            except GeminiConfigurationError as exc:
                raise ExtractionError(f"Gemini credentials are required for keyframe summaries: {exc}") from exc

        if self._initialized:
            return self._client

        init = getattr(self._client, "init", None)
        if callable(init):
            initialized = init(**self._init_kwargs)
            if hasattr(initialized, "__await__"):
                await initialized
        self._initialized = True
        return self._client

    async def _close_async(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            closed = close()
            if hasattr(closed, "__await__"):
                await closed


def _coerce_description(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        data = raw
    else:
        raw_text = str(raw).strip()
        data = _parse_json_object(raw_text)
        if not isinstance(data, dict):
            data = {"visual_summary": _strip_markdown_fence(raw_text)}

    summary = str(data.get("visual_summary") or data.get("summary") or "").strip()
    keywords = _coerce_keywords(data.get("visual_keywords") or data.get("keywords") or [])
    content_type = str(data.get("visual_content_type") or data.get("content_type") or "unknown").strip()
    if content_type not in VISUAL_CONTENT_TYPES:
        content_type = "unknown"
    confidence_value = data["visual_confidence"] if "visual_confidence" in data else data.get("confidence")

    return {
        "visual_summary": summary,
        "visual_keywords": keywords,
        "visual_content_type": content_type,
        "visual_confidence": _bounded_float(confidence_value),
    }


def _coerce_keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = value.replace("，", ",").split(",")
    elif isinstance(value, list | tuple):
        raw_items = value
    else:
        raw_items = []
    keywords: list[str] = []
    for item in raw_items:
        keyword = str(item).strip()
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    return keywords[:12]


def _parse_json_object(value: str) -> Any:
    text = _strip_markdown_fence(value)
    for candidate in (text, _json_object_slice(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_object_slice(value: str) -> str | None:
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return None
    return value[start : end + 1]


def _strip_markdown_fence(value: str) -> str:
    text = value.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _resolve_page_frame(output_dir: Path, page: dict[str, Any]) -> Path:
    frame_path = Path(str(page.get("frame_path") or ""))
    if not frame_path:
        raise ExtractionError("page has no frame_path")
    if not frame_path.is_absolute():
        frame_path = output_dir / frame_path
    if not frame_path.is_file():
        raise ExtractionError(f"page frame does not exist: {frame_path}")
    return frame_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(model_label: str, image_hash: str) -> str:
    return f"{VISUAL_DESCRIPTION_PROMPT_VERSION}:{model_label}:{image_hash}"


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _mark_skipped(page: dict[str, Any], provider: str, model: str, reason: str) -> None:
    page.update(
        {
            "visual_summary": "",
            "visual_keywords": [],
            "visual_content_type": "none",
            "visual_confidence": 0.0,
            "visual_provider": provider,
            "visual_model": model,
            "visual_cache_hit": False,
            "visual_skipped_reason": reason,
        }
    )


def _mark_failed(page: dict[str, Any], provider: str, model: str, error: str) -> None:
    page.update(
        {
            "visual_summary": "",
            "visual_keywords": [],
            "visual_content_type": "unknown",
            "visual_confidence": 0.0,
            "visual_provider": provider,
            "visual_model": model,
            "visual_cache_hit": False,
            "visual_error": error,
        }
    )


def _bounded_float(value: Any) -> float:
    return max(0.0, min(1.0, _coerce_float(value)))


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _emit(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
    progress: float | None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, message, progress)


def _emit_page_progress(
    progress_callback: ProgressCallback | None,
    index: int,
    total: int,
) -> None:
    if total and (index + 1 == total or (index + 1) % 10 == 0):
        percent = (index + 1) / total * 100
        progress = 0.89 + 0.03 * (index + 1) / total
        _emit(progress_callback, "visual_describe", f"总结关键帧 {percent:.1f}%", progress)
