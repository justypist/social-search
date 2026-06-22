from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from social_extract.errors import ExtractionError
from social_extract.extractor import Extractor
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
        result = Extractor(progress_callback=progress).extract(job["url"], config)
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
            "transcript_text_path": str(result.transcript_text_path),
            "transcript_json_path": str(result.transcript_json_path),
            "meta_path": str(result.meta_path),
            "audio_path": str(result.audio_path) if result.audio_path else None,
            "video_path": str(result.video_path) if result.video_path else None,
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
    )


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
