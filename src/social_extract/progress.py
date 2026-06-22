from __future__ import annotations

from collections.abc import Callable


ProgressCallback = Callable[[str, str, float | None], None]
StageProgressCallback = Callable[[float | None, str], None]


def stage_progress_callback(
    progress_callback: ProgressCallback | None,
    stage: str,
    start: float,
    end: float,
) -> StageProgressCallback:
    def callback(percent: float | None, message: str) -> None:
        if progress_callback is None:
            return
        progress = None if percent is None else start + (end - start) * percent / 100
        progress_callback(stage, message, progress)

    return callback


def is_live_progress_message(stage: str, message: str) -> bool:
    if stage not in {"download_audio", "download_video", "transcribe"}:
        return False
    return message.startswith(("下载中", "转写中")) or message in {"下载完成"}
