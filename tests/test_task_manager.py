from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from api.settings import WebSettings
from api.routers.tasks import CreateTaskRequest
from api.task_manager import TaskManager


class FakeStdout:
    async def readline(self) -> bytes:
        return b""


class FakeProcess:
    def __init__(self) -> None:
        self.stdout = FakeStdout()
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_stop_task_while_worker_process_is_starting(monkeypatch: Any, tmp_path: Path) -> None:
    asyncio.run(_stop_task_while_worker_process_is_starting(monkeypatch, tmp_path))


async def _stop_task_while_worker_process_is_starting(monkeypatch: Any, tmp_path: Path) -> None:
    process = FakeProcess()
    spawn_called = asyncio.Event()
    allow_spawn = asyncio.Event()

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        spawn_called.set()
        await allow_spawn.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    manager = _manager(tmp_path)
    await manager.start()
    try:
        created = await manager.create_task("https://example.test/video")
        task_id = created["id"]
        await asyncio.wait_for(spawn_called.wait(), timeout=1)

        running = await manager.get_task(task_id)
        assert running["status"] == "running"
        assert running["can_stop"] is True

        stopping = await manager.stop_task(task_id)
        assert stopping["status"] == "stopping"

        repeated = await manager.stop_task(task_id)
        assert repeated["status"] == "stopping"

        allow_spawn.set()
        await asyncio.wait_for(manager._queue.join(), timeout=1)

        finished = await manager.get_task(task_id)
        assert finished["status"] == "stopped"
        assert process.terminated is True
    finally:
        await manager.shutdown()


def test_worker_startup_failure_marks_task_failed(monkeypatch: Any, tmp_path: Path) -> None:
    asyncio.run(_worker_startup_failure_marks_task_failed(monkeypatch, tmp_path))


async def _worker_startup_failure_marks_task_failed(monkeypatch: Any, tmp_path: Path) -> None:
    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        raise OSError("spawn failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    manager = _manager(tmp_path)
    await manager.start()
    try:
        created = await manager.create_task("https://example.test/video")
        task_id = created["id"]
        await asyncio.wait_for(manager._queue.join(), timeout=1)

        failed = await manager.get_task(task_id)
        assert failed["status"] == "failed"
        assert failed["stage"] == "failed"
        assert failed["can_delete"] is True
        assert "worker 启动失败: spawn failed" == failed["error"]
    finally:
        await manager.shutdown()


def test_create_task_uses_requested_language(tmp_path: Path) -> None:
    asyncio.run(_create_task_uses_requested_language(tmp_path))


async def _create_task_uses_requested_language(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    created = await manager.create_task("https://example.test/video", language="zh")
    task_id = created["id"]
    record = manager._tasks[task_id]

    assert created["language"] == "zh"
    assert manager._job_payload(record)["language"] == "zh"


def test_create_task_defaults_to_configured_language(tmp_path: Path) -> None:
    asyncio.run(_create_task_defaults_to_configured_language(tmp_path))


async def _create_task_defaults_to_configured_language(tmp_path: Path) -> None:
    manager = _manager(tmp_path, language="en")

    created = await manager.create_task("https://example.test/video")
    task_id = created["id"]
    record = manager._tasks[task_id]

    assert created["language"] == "en"
    assert created["extract_visual"] is False
    assert manager._job_payload(record)["language"] == "en"
    assert manager._job_payload(record)["extract_visual"] is False


def test_create_task_persists_extract_visual_choice(tmp_path: Path) -> None:
    asyncio.run(_create_task_persists_extract_visual_choice(tmp_path))


async def _create_task_persists_extract_visual_choice(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    manager = TaskManager(_settings(tmp_path), state_dir=state_dir)

    created = await manager.create_task("https://example.test/video", extract_visual=True)
    task_id = created["id"]
    record = manager._tasks[task_id]
    payload = manager._job_payload(record)
    stored = json.loads((state_dir / "tasks.json").read_text(encoding="utf-8"))

    assert created["extract_visual"] is True
    assert record.extract_visual is True
    assert payload["extract_visual"] is True
    assert stored["tasks"][0]["extract_visual"] is True


def test_create_task_request_defaults_extract_visual_false() -> None:
    request = CreateTaskRequest(url="https://example.test/video")

    assert request.extract_visual is False


def test_visual_progress_logs_replace_previous_progress_message(tmp_path: Path) -> None:
    asyncio.run(_visual_progress_logs_replace_previous_progress_message(tmp_path))


async def _visual_progress_logs_replace_previous_progress_message(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    created = await manager.create_task("https://example.test/video", extract_visual=True)
    task_id = created["id"]

    await manager._handle_worker_line(
        task_id,
        json.dumps(
            {
                "type": "progress",
                "stage": "visual_detect",
                "message": "检测文字画面 25.0%",
                "progress": 0.81,
            }
        ),
    )
    await manager._handle_worker_line(
        task_id,
        json.dumps(
            {
                "type": "progress",
                "stage": "visual_detect",
                "message": "检测文字画面 50.0%",
                "progress": 0.82,
            }
        ),
    )

    task = await manager.get_task(task_id)
    progress_logs = [entry for entry in task["logs"] if entry["level"] == "progress"]
    assert len(progress_logs) == 1
    assert progress_logs[0]["message"] == "检测文字画面 50.0%"


def test_job_payload_includes_cookie_settings(tmp_path: Path) -> None:
    youtube_cookie_file = tmp_path / "youtube-cookies.txt"
    bilibili_cookie_file = tmp_path / "bilibili-cookies.txt"
    manager = _manager(
        tmp_path,
        cookie_files=(youtube_cookie_file, bilibili_cookie_file),
        cookies_from_browser="chrome:Default",
    )
    record = manager._tasks[
        asyncio.run(manager.create_task("https://example.test/video"))["id"]
    ]

    payload = manager._job_payload(record)

    assert payload["cookie_files"] == [str(youtube_cookie_file), str(bilibili_cookie_file)]
    assert payload["cookies_from_browser"] == "chrome:Default"


def test_completed_tasks_are_loaded_after_restart(tmp_path: Path) -> None:
    asyncio.run(_completed_tasks_are_loaded_after_restart(tmp_path))


async def _completed_tasks_are_loaded_after_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    manager = TaskManager(_settings(tmp_path), state_dir=state_dir)
    created = await manager.create_task("https://example.test/video")
    task_id = created["id"]
    output_dir = tmp_path / "output" / "video"
    output_dir.mkdir(parents=True)
    (output_dir / "transcript.txt").write_text("hello\n", encoding="utf-8")

    record = manager._tasks[task_id]
    record.status = "succeeded"
    record.stage = "done"
    record.progress = 100
    record.output_dir = str(output_dir)
    record.files = manager._scan_files(task_id, output_dir)
    manager._persist_tasks_locked()

    restarted = TaskManager(_settings(tmp_path), state_dir=state_dir)
    await restarted.start()
    try:
        tasks = await restarted.list_tasks()
    finally:
        await restarted.shutdown()

    assert [task["id"] for task in tasks] == [task_id]
    assert tasks[0]["status"] == "succeeded"
    assert tasks[0]["files"][0]["name"] == "transcript.txt"


def test_delete_task_removes_output_directory_and_persisted_record(tmp_path: Path) -> None:
    asyncio.run(_delete_task_removes_output_directory_and_persisted_record(tmp_path))


async def _delete_task_removes_output_directory_and_persisted_record(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    manager = TaskManager(_settings(tmp_path), state_dir=state_dir)
    created = await manager.create_task("https://example.test/video")
    task_id = created["id"]
    output_dir = tmp_path / "output" / "video"
    output_dir.mkdir(parents=True)
    (output_dir / "meta.json").write_text("{}", encoding="utf-8")

    record = manager._tasks[task_id]
    record.status = "succeeded"
    record.output_dir = str(output_dir)
    manager._persist_tasks_locked()

    await manager.delete_task(task_id)

    assert not output_dir.exists()
    stored = json.loads((state_dir / "tasks.json").read_text(encoding="utf-8"))
    assert stored["tasks"] == []


def test_delete_failed_duplicate_download_removes_existing_output_directory(tmp_path: Path) -> None:
    asyncio.run(_delete_failed_duplicate_download_removes_existing_output_directory(tmp_path))


def test_scan_files_hides_visual_frame_outputs(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    output_dir = tmp_path / "output" / "video"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True)
    (output_dir / "subtitle.srt").write_text("subtitle\n", encoding="utf-8")
    (output_dir / "pages.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "frames.json").write_text("{}\n", encoding="utf-8")
    (frames_dir / "000001.jpg").write_bytes(b"frame")
    (frames_dir / "page_0000.jpg").write_bytes(b"page")

    files = manager._scan_files("task-id", output_dir)

    assert [file["name"] for file in files] == ["pages.json", "subtitle.srt"]


async def _delete_failed_duplicate_download_removes_existing_output_directory(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    created = await manager.create_task("https://example.test/video")
    task_id = created["id"]
    output_dir = tmp_path / "output" / "video"
    output_dir.mkdir(parents=True)
    (output_dir / "transcript.txt").write_text("old\n", encoding="utf-8")

    await manager._handle_worker_line(
        task_id,
        json.dumps({"type": "error", "message": f"Output directory already exists: {output_dir}"}),
    )
    await manager._finish_task(task_id, 1)

    failed = await manager.get_task(task_id)
    assert failed["status"] == "failed"
    assert failed["output_dir"] == str(output_dir.resolve())

    await manager.delete_task(task_id)

    assert not output_dir.exists()


def _manager(
    tmp_path: Path,
    *,
    language: str = "auto",
    cookie_files: tuple[Path, ...] = (),
    cookies_from_browser: str | None = None,
) -> TaskManager:
    return TaskManager(
        _settings(
            tmp_path,
            language=language,
            cookie_files=cookie_files,
            cookies_from_browser=cookies_from_browser,
        ),
        state_dir=tmp_path / "state",
    )


def _settings(
    tmp_path: Path,
    *,
    language: str = "auto",
    cookie_files: tuple[Path, ...] = (),
    cookies_from_browser: str | None = None,
) -> WebSettings:
    return WebSettings(
        host="127.0.0.1",
        port=8000,
        concurrency=1,
        output_dir=tmp_path / "output",
        language=language,  # type: ignore[arg-type]
        model="small",
        device="cpu",
        compute_type="int8",
        vad_filter=False,
        keep_media=True,
        overwrite=True,
        task_log_limit=50,
        allowed_origins=[],
        http_headers={},
        cookie_files=cookie_files,
        cookies_from_browser=cookies_from_browser,
        env_file=tmp_path / ".env",
    )
