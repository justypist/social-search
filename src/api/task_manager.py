from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlparse

from fastapi import HTTPException

from .settings import WebSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TaskStatus = Literal["queued", "running", "stopping", "succeeded", "failed", "cancelled", "stopped"]


@dataclass
class LogEntry:
    at: str
    level: str
    message: str


@dataclass
class TaskRecord:
    id: str
    url: str
    status: TaskStatus
    progress: int
    stage: str
    created_at: str
    updated_at: str
    sequence: int
    started_at: str | None = None
    finished_at: str | None = None
    output_dir: str | None = None
    source: str | None = None
    error: str | None = None
    message: str = ""
    logs: list[LogEntry] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    job_file: str | None = None
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)


class TaskManager:
    def __init__(self, settings: WebSettings) -> None:
        self._settings = settings
        self._tasks: dict[str, TaskRecord] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._jobs_dir = PROJECT_ROOT / ".social-search" / "jobs"

    async def start(self) -> None:
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        for index in range(self._settings.concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop(index), name=f"extract-worker-{index}"))

    async def shutdown(self) -> None:
        async with self._lock:
            running = [task for task in self._tasks.values() if task.process is not None]
            for task in running:
                task.status = "stopping"
                self._append_log(task, "warning", "服务正在关闭，停止任务")
                self._terminate_process(task.process)

        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def create_task(self, url: str) -> dict[str, Any]:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=422, detail="请输入有效的视频链接")

        now = _now()
        task_id = uuid.uuid4().hex
        async with self._lock:
            self._sequence += 1
            record = TaskRecord(
                id=task_id,
                url=url.strip(),
                status="queued",
                progress=0,
                stage="queued",
                created_at=now,
                updated_at=now,
                sequence=self._sequence,
            )
            self._append_log(record, "info", "任务已加入队列")
            self._tasks[task_id] = record
        await self._queue.put(task_id)
        return await self.get_task(task_id)

    async def list_tasks(self) -> list[dict[str, Any]]:
        async with self._lock:
            records = sorted(self._tasks.values(), key=lambda item: item.sequence, reverse=True)
            return [self._task_payload(record) for record in records]

    async def get_task(self, task_id: str) -> dict[str, Any]:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(status_code=404, detail="任务不存在")
            if record.status == "succeeded" and record.output_dir:
                record.files = self._scan_files(record.id, Path(record.output_dir))
            return self._task_payload(record)

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(status_code=404, detail="任务不存在")
            if record.status != "queued":
                raise HTTPException(status_code=409, detail="只有队列中的任务可以取消")
            record.status = "cancelled"
            record.stage = "cancelled"
            record.finished_at = _now()
            record.updated_at = record.finished_at
            self._append_log(record, "warning", "任务已取消")
            return self._task_payload(record)

    async def stop_task(self, task_id: str) -> dict[str, Any]:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(status_code=404, detail="任务不存在")
            if record.status == "queued":
                record.status = "cancelled"
                record.stage = "cancelled"
                record.finished_at = _now()
                record.updated_at = record.finished_at
                self._append_log(record, "warning", "任务尚未开始，已取消")
                return self._task_payload(record)
            if record.status not in {"running", "stopping"} or record.process is None:
                raise HTTPException(status_code=409, detail="只有运行中的任务可以停止")
            if record.status != "stopping":
                record.status = "stopping"
                record.stage = "stopping"
                record.updated_at = _now()
                self._append_log(record, "warning", "正在停止任务")
                self._terminate_process(record.process)
                asyncio.create_task(self._kill_later(record.id, record.process))
            return self._task_payload(record)

    async def delete_task(self, task_id: str) -> None:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(status_code=404, detail="任务不存在")
            if record.status in {"running", "stopping"}:
                raise HTTPException(status_code=409, detail="请先停止运行中的任务")
            self._tasks.pop(task_id)
            if record.job_file:
                Path(record.job_file).unlink(missing_ok=True)

    def get_file(self, task_id: str, file_path: str) -> Path:
        record = self._tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        if not record.output_dir:
            raise HTTPException(status_code=404, detail="任务还没有生成文件")

        base = Path(record.output_dir).resolve()
        target = (base / file_path).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="文件不存在") from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        return target

    def settings_payload(self) -> dict[str, Any]:
        return {
            "concurrency": self._settings.concurrency,
            "output_dir": str(self._settings.output_dir),
            "language": self._settings.language,
            "model": self._settings.model,
            "device": self._settings.device,
            "compute_type": self._settings.compute_type,
            "vad_filter": self._settings.vad_filter,
            "keep_media": self._settings.keep_media,
            "overwrite": self._settings.overwrite,
            "env_file": str(self._settings.env_file),
        }

    async def _worker_loop(self, index: int) -> None:
        while True:
            task_id = await self._queue.get()
            try:
                async with self._lock:
                    record = self._tasks.get(task_id)
                    if record is None or record.status != "queued":
                        continue
                    record.status = "running"
                    record.stage = "starting"
                    record.progress = 1
                    record.started_at = _now()
                    record.updated_at = record.started_at
                    self._append_log(record, "info", f"工作线程 {index + 1} 开始执行")
                await self._run_task(task_id)
            finally:
                self._queue.task_done()

    async def _run_task(self, task_id: str) -> None:
        job_path = self._jobs_dir / f"{task_id}.json"
        async with self._lock:
            record = self._tasks[task_id]
            job_payload = self._job_payload(record)
            record.job_file = str(job_path)
        job_path.write_text(json.dumps(job_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "api.worker",
            str(job_path),
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                self._terminate_process(process)
                return
            record.process = process
            record.updated_at = _now()

        assert process.stdout is not None
        while True:
            raw_line = await process.stdout.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                await self._handle_worker_line(task_id, line)

        return_code = await process.wait()
        await self._finish_task(task_id, return_code)

    async def _handle_worker_line(self, task_id: str, line: str) -> None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            async with self._lock:
                record = self._tasks.get(task_id)
                if record is not None:
                    if not self._handle_progress_line(record, line):
                        self._append_log(record, "info", line)
            return

        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            event_type = payload.get("type")
            if event_type == "progress":
                message = str(payload.get("message") or "")
                stage = str(payload.get("stage") or record.stage)
                progress = _progress_to_int(payload.get("progress"), record.progress)
                record.stage = stage
                record.progress = max(record.progress, progress)
                record.updated_at = _now()
                if message:
                    self._append_log(record, "info", message)
            elif event_type == "result":
                record.source = str(payload.get("source") or "")
                output_dir = payload.get("output_dir")
                if output_dir:
                    record.output_dir = str(output_dir)
                    record.files = self._scan_files(record.id, Path(record.output_dir))
                record.updated_at = _now()
                self._append_log(record, "success", "文件列表已生成")
            elif event_type == "error":
                message = str(payload.get("message") or "任务执行失败")
                record.error = message
                record.updated_at = _now()
                self._append_log(record, "error", message)

    async def _finish_task(self, task_id: str, return_code: int) -> None:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            record.process = None
            record.finished_at = _now()
            record.updated_at = record.finished_at

            if record.status == "stopping":
                record.status = "stopped"
                record.stage = "stopped"
                self._append_log(record, "warning", "任务已停止")
                return
            if return_code == 0 and record.output_dir:
                record.status = "succeeded"
                record.stage = "done"
                record.progress = 100
                record.files = self._scan_files(record.id, Path(record.output_dir))
                self._append_log(record, "success", "任务完成")
                return

            record.status = "failed"
            record.stage = "failed"
            record.error = record.error or f"worker exited with code {return_code}"
            self._append_log(record, "error", record.error)

    async def _kill_later(self, task_id: str, process: asyncio.subprocess.Process) -> None:
        await asyncio.sleep(5)
        if process.returncode is not None:
            return
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                self._append_log(record, "warning", "停止超时，强制结束进程")
        try:
            process.kill()
        except ProcessLookupError:
            pass

    def _job_payload(self, record: TaskRecord) -> dict[str, Any]:
        return {
            "url": record.url,
            "output_root": str(self._settings.output_dir),
            "language": self._settings.language,
            "model": self._settings.model,
            "device": self._settings.device,
            "compute_type": self._settings.compute_type,
            "vad_filter": self._settings.vad_filter,
            "keep_media": self._settings.keep_media,
            "overwrite": self._settings.overwrite,
            "http_headers": self._settings.http_headers,
        }

    def _append_log(self, record: TaskRecord, level: str, message: str) -> None:
        record.message = message
        record.logs.append(LogEntry(at=_now(), level=level, message=message))
        if len(record.logs) > self._settings.task_log_limit:
            del record.logs[: len(record.logs) - self._settings.task_log_limit]

    def _scan_files(self, task_id: str, output_dir: Path) -> list[dict[str, Any]]:
        if not output_dir.exists():
            return []
        files: list[dict[str, Any]] = []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(output_dir).as_posix()
            stat = path.stat()
            files.append(
                {
                    "name": rel,
                    "size": stat.st_size,
                    "modified_at": dt.datetime.fromtimestamp(stat.st_mtime, dt.UTC).isoformat(),
                    "download_url": f"/api/tasks/{task_id}/files/{quote(rel, safe='/')}",
                }
            )
        return files

    def _task_payload(self, record: TaskRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "url": record.url,
            "status": record.status,
            "progress": record.progress,
            "stage": record.stage,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "output_dir": record.output_dir,
            "source": record.source,
            "error": record.error,
            "message": record.message,
            "files": record.files,
            "logs": [entry.__dict__ for entry in record.logs],
            "can_cancel": record.status == "queued",
            "can_stop": record.status in {"running", "stopping"},
            "can_delete": record.status not in {"running", "stopping"},
        }

    @staticmethod
    def _terminate_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            pass

    def _handle_progress_line(self, record: TaskRecord, line: str) -> bool:
        progress = _parse_ytdlp_download_progress(line)
        if progress is None:
            return False

        percent = progress["percent"]
        if record.stage == "download_video":
            stage_start, stage_end = 48, 58
        else:
            stage_start, stage_end = 38, 64
            record.stage = "download_audio"

        record.progress = max(record.progress, round(stage_start + (stage_end - stage_start) * percent / 100))
        record.updated_at = _now()
        pieces = [f"下载中 {percent:.1f}%"]
        if progress["size"]:
            pieces.append(str(progress["size"]))
        if progress["speed"]:
            pieces.append(str(progress["speed"]))
        if progress["eta"]:
            pieces.append(f"ETA {progress['eta']}")
        self._append_progress_log(record, " | ".join(pieces))
        return True

    def _append_progress_log(self, record: TaskRecord, message: str) -> None:
        record.message = message
        if record.logs and record.logs[-1].level == "progress":
            record.logs[-1] = LogEntry(at=_now(), level="progress", message=message)
            return
        self._append_log(record, "progress", message)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _progress_to_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if 0 <= number <= 1:
        number *= 100
    return max(0, min(100, round(number)))


_DOWNLOAD_RE = re.compile(
    r"\[download\]\s+"
    r"(?P<percent>\d+(?:\.\d+)?)%\s+of\s+"
    r"(?P<size>\S+)"
    r"(?:\s+at\s+(?P<speed>\S+))?"
    r"(?:\s+ETA\s+(?P<eta>\S+))?"
)


def _parse_ytdlp_download_progress(line: str) -> dict[str, float | str | None] | None:
    matches = list(_DOWNLOAD_RE.finditer(line.replace("\r", " ")))
    if not matches:
        return None
    match = matches[-1]
    return {
        "percent": float(match.group("percent")),
        "size": match.group("size"),
        "speed": match.group("speed"),
        "eta": match.group("eta"),
    }
