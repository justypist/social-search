from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from social_extract.core import extract_subtitles
from social_extract.errors import ExtractionError
from social_extract.models import ExtractConfig


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        _emit({"type": "error", "message": "worker requires one job file path"})
        return 2

    job_path = Path(args[0])
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        config = _build_config(job)
    except Exception as exc:
        _emit({"type": "error", "message": f"could not read job config: {exc}"})
        return 2

    def progress(stage: str, message: str, progress_value: float | None) -> None:
        _emit(
            {
                "type": "progress",
                "stage": stage,
                "message": message,
                "progress": progress_value,
            }
        )

    try:
        result = extract_subtitles(job["url"], config, progress_callback=progress)
    except ExtractionError as exc:
        _emit({"type": "error", "message": str(exc)})
        return 1
    except Exception as exc:
        _emit({"type": "error", "message": f"unexpected worker error: {exc}"})
        return 1

    _emit(
        {
            "type": "result",
            "source": result.source,
            "output_dir": str(result.output_dir),
            "subtitle_path": str(result.subtitle_path),
            "paragraph_subtitle_path": str(result.paragraph_subtitle_path),
            "transcript_text_path": str(result.transcript_text_path),
            "transcript_json_path": str(result.transcript_json_path),
            "meta_path": str(result.meta_path),
            "audio_path": str(result.audio_path) if result.audio_path else None,
            "video_path": str(result.video_path) if result.video_path else None,
            "pages_json_path": str(result.pages_json_path) if result.pages_json_path else None,
            "frames_dir": str(result.frames_dir) if result.frames_dir else None,
        }
    )
    return 0


def _build_config(job: dict[str, Any]) -> ExtractConfig:
    return ExtractConfig(
        output_root=Path(job["output_root"]),
        language=job["language"],
        model=job["model"],
        device=job["device"],
        compute_type=job["compute_type"],
        vad_filter=job["vad_filter"],
        keep_media=job["keep_media"],
        overwrite=job["overwrite"],
        http_headers=job["http_headers"],
        cookie_files=_job_cookie_files(job),
        cookies_from_browser=job.get("cookies_from_browser"),
        extract_visual=bool(job.get("extract_visual", False)),
    )


def _job_cookie_files(job: dict[str, Any]) -> tuple[Path, ...]:
    values = job.get("cookie_files")
    if isinstance(values, list):
        return tuple(Path(str(value)).expanduser() for value in values if str(value).strip())
    cookie_file = job.get("cookie_file")
    if cookie_file:
        return (Path(str(cookie_file)).expanduser(),)
    return ()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
