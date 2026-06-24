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
    if stage in {"download_audio", "download_video"}:
        return message.startswith("下载中") or message == "下载完成"
    if stage == "transcribe":
        return message.startswith("转写中")
    if stage == "visual_detect":
        return message.startswith("检测文字画面 ")
    if stage == "visual_ocr":
        return message.startswith("识别画面文字 ")
    return False
