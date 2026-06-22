from __future__ import annotations

import pytest

from social_extract.progress import is_live_progress_message, stage_progress_callback


def test_stage_progress_callback_maps_local_percent_to_total_progress() -> None:
    events: list[tuple[str, str, float | None]] = []
    callback = stage_progress_callback(
        lambda stage, message, progress: events.append((stage, message, progress)),
        "download_audio",
        0.38,
        0.64,
    )

    callback(50.0, "下载中 50.0%")

    assert events == [("download_audio", "下载中 50.0%", pytest.approx(0.51))]


def test_live_progress_messages_are_identified() -> None:
    assert is_live_progress_message("download_audio", "下载中 50.0%")
    assert is_live_progress_message("transcribe", "转写中 25.0%")
    assert not is_live_progress_message("write", "正在写入字幕和转写文件")
