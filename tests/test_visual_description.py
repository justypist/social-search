from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from social_extract.models import ExtractConfig
from social_extract.visual_description import GeminiVisualDescriber, _coerce_description


class StubGeminiVisionClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.init_calls: list[dict] = []
        self.generate_calls: list[dict] = []
        self.close_calls = 0

    async def init(self, **kwargs) -> None:
        self.init_calls.append(kwargs)

    async def generate_content(self, prompt: str, *, files: list[Path], **kwargs):
        self.generate_calls.append({"prompt": prompt, "files": files, "kwargs": kwargs})
        response = self.responses.pop(0)
        if response == "__raise__":
            raise RuntimeError("api failed")
        return SimpleNamespace(text=response)

    async def close(self) -> None:
        self.close_calls += 1


def test_gemini_visual_describer_uses_keyframe_file_and_ocr_context(tmp_path: Path) -> None:
    frame_path = _write_frame(tmp_path, "frames/page_0000.jpg", b"frame")
    client = StubGeminiVisionClient(
        [
            '{"visual_summary": "页面包含系统架构图。", "visual_keywords": ["系统架构", "API"], '
            '"visual_content_type": "architecture_diagram", "visual_confidence": 0.86}'
        ]
    )
    describer = GeminiVisualDescriber(client=client, model="gemini-test")

    try:
        result = describer.describe_pages(
            [_page(frame_path, "OCR text")],
            tmp_path,
            ExtractConfig(output_root=tmp_path, describe_visual=True),
        )
    finally:
        describer.close()

    page = result.pages[0]
    assert page["visual_summary"] == "页面包含系统架构图。"
    assert page["visual_keywords"] == ["系统架构", "API"]
    assert page["visual_content_type"] == "architecture_diagram"
    assert page["visual_confidence"] == pytest.approx(0.86)
    assert page["visual_provider"] == "gemini"
    assert page["visual_model"] == "gemini-test"
    assert page["visual_cache_hit"] is False
    assert result.meta["described_pages"] == 1
    assert client.init_calls == [{"auto_refresh": False}]
    assert client.generate_calls[0]["files"] == [frame_path]
    assert client.generate_calls[0]["kwargs"] == {"temporary": True, "model": "gemini-test"}
    assert "OCR text" in client.generate_calls[0]["prompt"]
    assert client.close_calls == 1


def test_gemini_visual_describer_reuses_cache_for_same_frame(tmp_path: Path) -> None:
    frame_path = _write_frame(tmp_path, "frames/page_0000.jpg", b"same-frame")
    client = StubGeminiVisionClient(
        [
            '{"visual_summary": "缓存描述", "visual_keywords": ["缓存"], '
            '"visual_content_type": "screenshot", "visual_confidence": 0.7}'
        ]
    )
    describer = GeminiVisualDescriber(client=client)

    try:
        result = describer.describe_pages(
            [_page(frame_path, "first"), _page(frame_path, "second")],
            tmp_path,
            ExtractConfig(output_root=tmp_path, describe_visual=True),
        )
    finally:
        describer.close()

    assert len(client.generate_calls) == 1
    assert result.pages[0]["visual_cache_hit"] is False
    assert result.pages[1]["visual_cache_hit"] is True
    assert result.pages[1]["visual_summary"] == "缓存描述"
    assert result.meta["cache_hits"] == 1


def test_gemini_visual_describer_respects_page_limit(tmp_path: Path) -> None:
    first = _write_frame(tmp_path, "frames/page_0000.jpg", b"first")
    second = _write_frame(tmp_path, "frames/page_0001.jpg", b"second")
    client = StubGeminiVisionClient(
        [
            '{"visual_summary": "描述", "visual_keywords": [], '
            '"visual_content_type": "mixed", "visual_confidence": 0.6}'
        ]
    )

    result = GeminiVisualDescriber(client=client).describe_pages(
        [_page(first, "first"), _page(second, "second")],
        tmp_path,
        ExtractConfig(output_root=tmp_path, describe_visual=True, max_visual_describe_pages=1),
    )

    assert len(client.generate_calls) == 1
    assert result.pages[1]["visual_skipped_reason"] == "max_visual_describe_pages"
    assert result.meta["skipped_pages"] == 1


def test_gemini_visual_describer_optional_failure_keeps_page(tmp_path: Path) -> None:
    frame_path = _write_frame(tmp_path, "frames/page_0000.jpg", b"frame")
    client = StubGeminiVisionClient(["__raise__"])

    result = GeminiVisualDescriber(client=client).describe_pages(
        [_page(frame_path, "OCR")],
        tmp_path,
        ExtractConfig(output_root=tmp_path, describe_visual=True, visual_description_optional=True),
    )

    assert result.pages[0]["visual_summary"] == ""
    assert "api failed" in result.pages[0]["visual_error"]
    assert result.meta["failed_pages"] == 1


def test_gemini_visual_describer_required_failure_raises(tmp_path: Path) -> None:
    frame_path = _write_frame(tmp_path, "frames/page_0000.jpg", b"frame")
    client = StubGeminiVisionClient(["__raise__"])

    with pytest.raises(Exception, match="Visual description failed"):
        GeminiVisualDescriber(client=client).describe_pages(
            [_page(frame_path, "OCR")],
            tmp_path,
            ExtractConfig(output_root=tmp_path, describe_visual=True, visual_description_optional=False),
        )


def test_visual_description_json_markdown_response_is_coerced() -> None:
    result = _coerce_description(
        '```json\n{"visual_summary": "趋势上升", "visual_keywords": "趋势,增长", '
        '"visual_content_type": "chart", "visual_confidence": "0.8"}\n```'
    )

    assert result["visual_summary"] == "趋势上升"
    assert result["visual_keywords"] == ["趋势", "增长"]
    assert result["visual_content_type"] == "chart"
    assert result["visual_confidence"] == pytest.approx(0.8)


def _write_frame(tmp_path: Path, relative_path: str, data: bytes) -> Path:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _page(frame_path: Path, text: str) -> dict:
    return {
        "page_index": 0,
        "start": 1.0,
        "end": 2.0,
        "text": text,
        "frame_path": frame_path.relative_to(frame_path.parents[1]).as_posix(),
        "confidence": 0.9,
    }
