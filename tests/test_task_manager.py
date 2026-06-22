from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from api.settings import WebSettings
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
    manager = TaskManager(_settings(tmp_path))
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
    manager = TaskManager(_settings(tmp_path))
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


def _settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        host="127.0.0.1",
        port=8000,
        concurrency=1,
        output_dir=tmp_path / "output",
        language="auto",
        model="small",
        device="cpu",
        compute_type="int8",
        vad_filter=False,
        keep_media=True,
        overwrite=True,
        task_log_limit=50,
        allowed_origins=[],
        http_headers={},
        env_file=tmp_path / ".env",
    )
