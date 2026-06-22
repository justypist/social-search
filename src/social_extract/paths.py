from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

from .errors import ExtractionError

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def prepare_output_dir(output_root: Path, info: dict[str, Any], url: str, overwrite: bool) -> Path:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_name = str(info.get("id") or info.get("title") or hashlib.sha1(url.encode()).hexdigest()[:12])
    target = output_root / safe_path_name(raw_name)
    if target.exists():
        if not overwrite:
            raise ExtractionError(f"Output directory already exists: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target


def safe_path_name(value: str) -> str:
    cleaned = _SAFE_CHARS_RE.sub("-", value.strip()).strip("-._")
    return cleaned[:120] or "video"


def relative_or_name(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
