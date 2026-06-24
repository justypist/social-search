from __future__ import annotations

import json
from pathlib import Path

from social_extract.frames import FrameExtractionResult, FrameRef
from social_extract.models import ExtractConfig
from social_extract.ocr import OcrResult
from social_extract.visual import VisualExtractor
from social_extract.vision import TextDedup


class FakeFrameExtractor:
    def __init__(self, count: int) -> None:
        self.count = count

    def extract(self, video_path: Path, output_dir: Path, *, fps: float) -> FrameExtractionResult:
        del video_path
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames: list[FrameRef] = []
        for index in range(self.count):
            path = frames_dir / f"{index + 1:06d}.jpg"
            path.write_bytes(f"frame-{index}".encode())
            frames.append(FrameRef(index=index, timestamp=index / fps, path=path))
        frames_json_path = output_dir / "frames.json"
        frames_json_path.write_text("{}", encoding="utf-8")
        return FrameExtractionResult(fps=fps, frames_dir=frames_dir, frames_json_path=frames_json_path, frames=frames)


class SequenceHasTextDetector:
    def __init__(self, flags: list[bool]) -> None:
        self.flags = flags

    def has_text(self, frame_path: Path, config: ExtractConfig) -> bool:
        del config
        index = int(frame_path.stem) - 1
        return self.flags[index]


class StaticPageChangeDetector:
    def __init__(self, *, changed: bool) -> None:
        self.changed = changed

    def is_page_change(self, previous_frame: Path, current_frame: Path, config: ExtractConfig) -> bool:
        del previous_frame, current_frame, config
        return self.changed

    def is_stable(self, previous_frame: Path, current_frame: Path, config: ExtractConfig) -> bool:
        del previous_frame, current_frame, config
        return True


class MappingOcrRecognizer:
    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts
        self.calls: list[str] = []

    def recognize(self, frame_path: Path) -> OcrResult:
        self.calls.append(frame_path.name)
        text = self.texts.get(frame_path.name, "")
        return OcrResult(texts=[text] if text else [], scores=[0.9] if text else [], boxes=[])


def test_text_dedup_merges_contained_incremental_text() -> None:
    dedup = TextDedup()

    assert dedup.should_merge(
        "Agenda\nItem one",
        "Agenda\nItem one\nItem two",
        ExtractConfig(output_root=Path("out")),
    )


def test_visual_extractor_writes_empty_pages_when_ocr_has_no_text(tmp_path: Path) -> None:
    recognizer = MappingOcrRecognizer({})
    extractor = VisualExtractor(
        frame_extractor=FakeFrameExtractor(2),
        ocr_recognizer=recognizer,
        has_text_detector=SequenceHasTextDetector([True, True]),
        page_change_detector=StaticPageChangeDetector(changed=False),
    )

    result = extractor.extract(
        tmp_path / "video.mp4",
        tmp_path,
        ExtractConfig(output_root=tmp_path, visual_text_min_chars=1),
    )

    payload = json.loads(result.pages_json_path.read_text(encoding="utf-8"))
    assert payload["pages"] == []
    assert payload["stats"] == {"sampled_frames": 2, "text_frames": 2, "ocr_frames": 1, "pages": 0}
    assert recognizer.calls == ["000001.jpg"]


def test_visual_extractor_records_non_text_segments(tmp_path: Path) -> None:
    extractor = VisualExtractor(
        frame_extractor=FakeFrameExtractor(4),
        ocr_recognizer=MappingOcrRecognizer({"000003.jpg": "Slide title"}),
        has_text_detector=SequenceHasTextDetector([False, False, True, True]),
        page_change_detector=StaticPageChangeDetector(changed=False),
    )

    result = extractor.extract(
        tmp_path / "video.mp4",
        tmp_path,
        ExtractConfig(output_root=tmp_path, frame_fps=1.0, visual_text_min_chars=1),
    )

    payload = json.loads(result.pages_json_path.read_text(encoding="utf-8"))
    assert payload["non_text_segments"] == [{"start": 0.0, "end": 2.0, "reason": "no_text"}]
    assert payload["pages"][0]["start"] == 2.0
    assert payload["pages"][0]["end"] == 4.0
    assert payload["pages"][0]["frame_path"] == "frames/page_0000.jpg"
    assert (result.frames_dir / "page_0000.jpg").exists()


def test_visual_extractor_filters_single_frame_text_blips(tmp_path: Path) -> None:
    recognizer = MappingOcrRecognizer({"000002.jpg": "Transient title"})
    extractor = VisualExtractor(
        frame_extractor=FakeFrameExtractor(3),
        ocr_recognizer=recognizer,
        has_text_detector=SequenceHasTextDetector([False, True, False]),
        page_change_detector=StaticPageChangeDetector(changed=False),
    )

    result = extractor.extract(
        tmp_path / "video.mp4",
        tmp_path,
        ExtractConfig(output_root=tmp_path, frame_fps=1.0, visual_text_min_chars=1),
    )

    payload = json.loads(result.pages_json_path.read_text(encoding="utf-8"))
    assert payload["pages"] == []
    assert payload["non_text_segments"] == [{"start": 0.0, "end": 3.0, "reason": "no_text"}]
    assert recognizer.calls == []


def test_visual_extractor_deduplicates_incremental_pages(tmp_path: Path) -> None:
    extractor = VisualExtractor(
        frame_extractor=FakeFrameExtractor(2),
        ocr_recognizer=MappingOcrRecognizer(
            {
                "000001.jpg": "Agenda\nItem one",
                "000002.jpg": "Agenda\nItem one\nItem two",
            }
        ),
        has_text_detector=SequenceHasTextDetector([True, True]),
        page_change_detector=StaticPageChangeDetector(changed=True),
    )

    result = extractor.extract(
        tmp_path / "video.mp4",
        tmp_path,
        ExtractConfig(output_root=tmp_path, frame_fps=1.0, visual_text_min_chars=1),
    )

    payload = json.loads(result.pages_json_path.read_text(encoding="utf-8"))
    assert len(payload["pages"]) == 1
    assert payload["pages"][0]["start"] == 0.0
    assert payload["pages"][0]["end"] == 2.0
    assert payload["pages"][0]["text"] == "Agenda\nItem one\nItem two"
