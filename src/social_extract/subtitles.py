from __future__ import annotations

from typing import Any

from .models import Language, SubtitleRef

_PREFERRED_FORMATS = ("srt", "vtt")
_AUTO_LANGUAGE_ORDER = ("zh", "zh-Hans", "zh-Hant", "zh-CN", "zh-TW", "en", "en-US", "en-GB")


def select_subtitle(info: dict[str, Any], language: Language) -> SubtitleRef | None:
    manual = _collect_caption_refs(info.get("subtitles") or {}, source="manual")
    automatic = _collect_caption_refs(info.get("automatic_captions") or {}, source="automatic")
    refs = manual + automatic
    if not refs:
        return None

    candidates = _filter_by_language(refs, language)
    if not candidates:
        return None

    return sorted(candidates, key=_subtitle_rank)[0]


def _collect_caption_refs(captions: dict[str, list[dict[str, Any]]], source: str) -> list[SubtitleRef]:
    refs: list[SubtitleRef] = []
    for language, entries in captions.items():
        for entry in entries:
            ext = str(entry.get("ext") or "").lower()
            if ext not in _PREFERRED_FORMATS:
                continue
            refs.append(
                SubtitleRef(
                    language=language,
                    ext=ext,
                    url=entry.get("url"),
                    data=entry.get("data"),
                    source="automatic" if source == "automatic" else "manual",
                )
            )
    return refs


def _filter_by_language(refs: list[SubtitleRef], language: Language) -> list[SubtitleRef]:
    if language == "auto":
        return refs
    return [ref for ref in refs if _language_matches(ref.language, language)]


def _language_matches(candidate: str, requested: str) -> bool:
    normalized = candidate.lower().replace("_", "-")
    requested = requested.lower()
    if requested == "zh":
        return normalized == "zh" or normalized.startswith("zh-")
    if requested == "en":
        return normalized == "en" or normalized.startswith("en-")
    return normalized == requested


def _subtitle_rank(ref: SubtitleRef) -> tuple[int, int, int]:
    source_rank = 0 if ref.source == "manual" else 1
    format_rank = _PREFERRED_FORMATS.index(ref.ext) if ref.ext in _PREFERRED_FORMATS else 99
    language_rank = _AUTO_LANGUAGE_ORDER.index(ref.language) if ref.language in _AUTO_LANGUAGE_ORDER else 99
    return source_rank, format_rank, language_rank
