from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from .models import Segment, Transcript

PARAGRAPH_DURATION_SECONDS = 60.0

_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[\.,]\d{1,3})\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[\.,]\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def subtitle_text_to_transcript(text: str, ext: str, language: str) -> Transcript:
    ext = ext.lower().lstrip(".")
    if ext not in {"srt", "vtt"}:
        raise ValueError(f"Unsupported subtitle extension: {ext}")

    segments: list[Segment] = []
    block: list[str] = []

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip("\ufeff")
        if not line.strip():
            _consume_block(block, segments)
            block = []
            continue
        if ext == "vtt" and line.strip().upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        block.append(line)

    _consume_block(block, segments)
    return Transcript(language=language, segments=segments)


def write_srt(transcript: Transcript, path: Path) -> None:
    lines: list[str] = []
    for index, segment in enumerate(transcript.segments, start=1):
        lines.append(str(index))
        lines.append(f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}")
        lines.append(segment.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paragraph_srt(
    transcript: Transcript,
    path: Path,
    *,
    max_duration_seconds: float = PARAGRAPH_DURATION_SECONDS,
) -> None:
    write_srt(aggregate_transcript(transcript, max_duration_seconds=max_duration_seconds), path)


def aggregate_transcript(
    transcript: Transcript,
    *,
    max_duration_seconds: float = PARAGRAPH_DURATION_SECONDS,
) -> Transcript:
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be greater than 0")

    paragraphs: list[Segment] = []
    start: float | None = None
    end: float | None = None
    texts: list[str] = []

    def flush() -> None:
        nonlocal start, end, texts
        if start is not None and end is not None and texts:
            paragraphs.append(Segment(start=start, end=end, text=" ".join(texts)))
        start = None
        end = None
        texts = []

    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue

        if start is not None and segment.start - start >= max_duration_seconds:
            flush()

        if start is None:
            start = segment.start
            end = segment.end
        else:
            end = max(end or segment.end, segment.end)
        texts.append(text)

        if end - start >= max_duration_seconds:
            flush()

    flush()
    return Transcript(language=transcript.language, segments=paragraphs)


def write_transcript_text(transcript: Transcript, path: Path) -> None:
    text = "\n".join(segment.text for segment in transcript.segments if segment.text)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_transcript_json(transcript: Transcript, path: Path) -> None:
    payload = {
        "language": transcript.language,
        "segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in transcript.segments
        ],
    }
    write_json(payload, path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _consume_block(block: list[str], segments: list[Segment]) -> None:
    if not block:
        return

    timestamp_index = next((index for index, line in enumerate(block) if "-->" in line), None)
    if timestamp_index is None:
        return

    match = _TIMESTAMP_RE.search(block[timestamp_index])
    if match is None:
        return

    text_lines = block[timestamp_index + 1 :]
    text = " ".join(_clean_caption_text(line) for line in text_lines)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return

    segments.append(
        Segment(
            start=parse_timestamp(match.group("start")),
            end=parse_timestamp(match.group("end")),
            text=text,
        )
    )


def parse_timestamp(value: str) -> float:
    normalized = value.replace(",", ".")
    pieces = normalized.split(":")
    if len(pieces) == 2:
        hours = 0
        minutes = int(pieces[0])
        seconds = float(pieces[1])
    elif len(pieces) == 3:
        hours = int(pieces[0])
        minutes = int(pieces[1])
        seconds = float(pieces[2])
    else:
        raise ValueError(f"Invalid timestamp: {value}")
    return hours * 3600 + minutes * 60 + seconds


def _clean_caption_text(value: str) -> str:
    without_tags = _TAG_RE.sub("", value)
    return html.unescape(without_tags).strip()
