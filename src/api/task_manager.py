from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import signal
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlparse

from fastapi import HTTPException
from social_extract.models import Language
from social_extract.progress import is_live_progress_message

from .settings import WebSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TaskStatus = Literal["queued", "running", "stopping", "succeeded", "failed", "cancelled", "stopped"]
TASK_STATUSES: set[str] = {"queued", "running", "stopping", "succeeded", "failed", "cancelled", "stopped"}
RESTART_STOPPED_STATUSES = {"queued", "running", "stopping"}
EXISTING_OUTPUT_PREFIX = "Output directory already exists: "


@dataclass
class LogEntry:
    at: str
    level: str
    message: str


@dataclass
class TaskRecord:
    id: str
    url: str
    language: Language
    extract_visual: bool
    describe_visual: bool
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
    stop_requested: bool = False
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)


class TaskManager:
    def __init__(self, settings: WebSettings, *, state_dir: Path | None = None) -> None:
        self._settings = settings
        self._tasks: dict[str, TaskRecord] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._state_dir = state_dir or PROJECT_ROOT / ".social-search"
        self._jobs_dir = self._state_dir / "jobs"
        self._tasks_file = self._state_dir / "tasks.json"

    async def start(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            changed = self._load_tasks_locked()
            if changed:
                self._persist_tasks_locked()
        for index in range(self._settings.concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop(index), name=f"extract-worker-{index}"))

    async def shutdown(self) -> None:
        async with self._lock:
            running = [task for task in self._tasks.values() if task.process is not None]
            for task in running:
                task.status = "stopping"
                self._append_log(task, "warning", "服务正在关闭，停止任务")
                self._terminate_process(task.process)
            if running:
                self._persist_tasks_locked()

        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def create_task(
        self,
        url: str,
        *,
        language: Language | None = None,
        extract_visual: bool = False,
        describe_visual: bool = False,
    ) -> dict[str, Any]:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=422, detail="请输入有效的视频链接")

        task_language = language or self._settings.language
        task_extract_visual = extract_visual or describe_visual
        now = _now()
        task_id = uuid.uuid4().hex
        async with self._lock:
            self._sequence += 1
            record = TaskRecord(
                id=task_id,
                url=url.strip(),
                language=task_language,
                extract_visual=task_extract_visual,
                describe_visual=describe_visual,
                status="queued",
                progress=0,
                stage="queued",
                created_at=now,
                updated_at=now,
                sequence=self._sequence,
            )
            visual_labels = []
            if record.extract_visual:
                visual_labels.append("画面文字：开启")
            if record.describe_visual:
                visual_labels.append("关键帧总结：开启")
            visual_label = f" | {' | '.join(visual_labels)}" if visual_labels else ""
            self._append_log(record, "info", f"任务已加入队列 | 语言：{_language_label(record.language)}{visual_label}")
            self._tasks[task_id] = record
            self._persist_tasks_locked()
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
            self._persist_tasks_locked()
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
                self._persist_tasks_locked()
                return self._task_payload(record)
            if record.status == "stopping":
                return self._task_payload(record)
            if record.status != "running":
                raise HTTPException(status_code=409, detail="只有运行中的任务可以停止")
            record.status = "stopping"
            record.stage = "stopping"
            record.stop_requested = True
            record.updated_at = _now()
            self._append_log(record, "warning", "正在停止任务")
            if record.process is not None:
                self._terminate_process(record.process)
                asyncio.create_task(self._kill_later(record.id, record.process))
            self._persist_tasks_locked()
            return self._task_payload(record)

    async def delete_task(self, task_id: str) -> None:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(status_code=404, detail="任务不存在")
            if record.status in {"running", "stopping"}:
                raise HTTPException(status_code=409, detail="请先停止运行中的任务")
            if record.output_dir:
                self._remove_output_dir(Path(record.output_dir))
            if record.job_file:
                Path(record.job_file).unlink(missing_ok=True)
            self._tasks.pop(task_id)
            self._persist_tasks_locked()

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
            "cookie_files": bool(self._settings.cookie_files),
            "cookies_from_browser": bool(self._settings.cookies_from_browser),
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
                    self._persist_tasks_locked()
                await self._run_task(task_id)
            finally:
                self._queue.task_done()

    async def _run_task(self, task_id: str) -> None:
        job_path = self._jobs_dir / f"{task_id}.json"
        async with self._lock:
            record = self._tasks[task_id]
            job_payload = self._job_payload(record)
            record.job_file = str(job_path)
            self._persist_tasks_locked()
        try:
            job_path.write_text(json.dumps(job_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            process_options: dict[str, Any] = {
                "cwd": str(PROJECT_ROOT),
                "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.STDOUT,
            }
            if os.name != "nt":
                process_options["start_new_session"] = True
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "api.worker",
                str(job_path),
                **process_options,
            )
        except Exception as exc:
            await self._finish_startup_failure(task_id, exc)
            return

        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                self._terminate_process(process)
                asyncio.create_task(self._kill_later(task_id, process))
                return
            record.process = process
            record.updated_at = _now()
            should_stop = record.status == "stopping" or record.stop_requested

        if should_stop:
            self._terminate_process(process)
            asyncio.create_task(self._kill_later(task_id, process))

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
                    self._persist_tasks_locked()
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
                    if is_live_progress_message(stage, message):
                        self._append_progress_log(record, message)
                    else:
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
                self._attach_existing_output_dir(record, message)
                record.updated_at = _now()
                self._append_log(record, "error", message)
            self._persist_tasks_locked()

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
                self._persist_tasks_locked()
                return
            if return_code == 0 and record.output_dir:
                record.status = "succeeded"
                record.stage = "done"
                record.progress = 100
                record.files = self._scan_files(record.id, Path(record.output_dir))
                self._append_log(record, "success", "任务完成")
                self._persist_tasks_locked()
                return

            record.status = "failed"
            record.stage = "failed"
            record.error = record.error or f"worker exited with code {return_code}"
            self._append_log(record, "error", record.error)
            self._persist_tasks_locked()

    async def _finish_startup_failure(self, task_id: str, exc: Exception) -> None:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            record.process = None
            record.finished_at = _now()
            record.updated_at = record.finished_at

            if record.status == "stopping" or record.stop_requested:
                record.status = "stopped"
                record.stage = "stopped"
                self._append_log(record, "warning", "任务已停止")
                self._persist_tasks_locked()
                return

            record.status = "failed"
            record.stage = "failed"
            record.error = f"worker 启动失败: {exc}"
            self._append_log(record, "error", record.error)
            self._persist_tasks_locked()

    async def _kill_later(self, task_id: str, process: asyncio.subprocess.Process) -> None:
        await asyncio.sleep(5)
        if process.returncode is not None:
            return
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                self._append_log(record, "warning", "停止超时，强制结束进程")
                self._persist_tasks_locked()
        self._kill_process(process)

    def _job_payload(self, record: TaskRecord) -> dict[str, Any]:
        return {
            "url": record.url,
            "output_root": str(self._settings.output_dir),
            "language": record.language,
            "extract_visual": record.extract_visual,
            "describe_visual": record.describe_visual,
            "model": self._settings.model,
            "device": self._settings.device,
            "compute_type": self._settings.compute_type,
            "vad_filter": self._settings.vad_filter,
            "keep_media": self._settings.keep_media,
            "overwrite": self._settings.overwrite,
            "http_headers": self._settings.http_headers,
            "cookie_files": [str(path) for path in self._settings.cookie_files],
            "cookies_from_browser": self._settings.cookies_from_browser,
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
            if _hide_from_file_list(rel):
                continue
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
            "language": record.language,
            "extract_visual": record.extract_visual,
            "describe_visual": record.describe_visual,
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

    def _load_tasks_locked(self) -> bool:
        changed = False
        if self._tasks_file.exists():
            try:
                payload = json.loads(self._tasks_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            raw_tasks = payload.get("tasks") if isinstance(payload, dict) else None
            if isinstance(raw_tasks, list):
                for raw_record in raw_tasks:
                    if not isinstance(raw_record, dict):
                        changed = True
                        continue
                    record = self._record_from_payload(raw_record)
                    if record is None:
                        changed = True
                        continue
                    if record.status in RESTART_STOPPED_STATUSES:
                        record.status = "stopped"
                        record.stage = "stopped"
                        record.finished_at = record.finished_at or _now()
                        record.updated_at = _now()
                        record.stop_requested = False
                        record.process = None
                        self._append_log(record, "warning", "服务重启后任务已标记为停止")
                        changed = True
                    if record.output_dir:
                        record.files = self._scan_files(record.id, Path(record.output_dir))
                    self._tasks[record.id] = record
                    self._sequence = max(self._sequence, record.sequence)

        changed = self._import_output_dirs_locked() or changed
        changed = self._import_job_files_locked() or changed
        return changed

    def _record_from_payload(self, payload: dict[str, Any]) -> TaskRecord | None:
        task_id = _clean_string(payload.get("id"))
        url = _clean_string(payload.get("url"))
        if not task_id or not url:
            return None

        status = _clean_string(payload.get("status")) or "failed"
        if status not in TASK_STATUSES:
            status = "failed"

        logs: list[LogEntry] = []
        for item in payload.get("logs") or []:
            if not isinstance(item, dict):
                continue
            at = _clean_string(item.get("at")) or _now()
            level = _clean_string(item.get("level")) or "info"
            message = _clean_string(item.get("message")) or ""
            logs.append(LogEntry(at=at, level=level, message=message))

        sequence = _coerce_int(payload.get("sequence"), self._sequence + 1)
        describe_visual = _coerce_bool(payload.get("describe_visual"), False)
        extract_visual = _coerce_bool(payload.get("extract_visual"), False) or describe_visual
        return TaskRecord(
            id=task_id,
            url=url,
            language=_coerce_language(payload.get("language"), self._settings.language),
            extract_visual=extract_visual,
            describe_visual=describe_visual,
            status=status,  # type: ignore[arg-type]
            progress=max(0, min(100, _coerce_int(payload.get("progress"), 0))),
            stage=_clean_string(payload.get("stage")) or status,
            created_at=_clean_string(payload.get("created_at")) or _now(),
            updated_at=_clean_string(payload.get("updated_at")) or _now(),
            sequence=sequence,
            started_at=_clean_string(payload.get("started_at")),
            finished_at=_clean_string(payload.get("finished_at")),
            output_dir=_clean_string(payload.get("output_dir")),
            source=_clean_string(payload.get("source")),
            error=_clean_string(payload.get("error")),
            message=_clean_string(payload.get("message")) or "",
            logs=logs[-self._settings.task_log_limit :],
            files=[],
            job_file=_clean_string(payload.get("job_file")),
            stop_requested=False,
            process=None,
        )

    def _import_output_dirs_locked(self) -> bool:
        output_root = self._settings.output_dir.expanduser()
        if not output_root.exists():
            return False

        changed = False
        for meta_path in sorted(output_root.rglob("meta.json")):
            output_dir = meta_path.parent
            task_id = uuid.uuid5(uuid.NAMESPACE_URL, str(output_dir.resolve())).hex
            if task_id in self._tasks:
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(meta, dict):
                continue
            url = _clean_string(meta.get("url"))
            if not url or any(record.output_dir == str(output_dir.resolve()) for record in self._tasks.values()):
                continue

            now = _now()
            created_at = _clean_string(meta.get("extracted_at")) or now
            self._sequence += 1
            describe_visual = _coerce_bool(meta.get("describe_visual"), False)
            extract_visual = _coerce_bool(meta.get("extract_visual"), False) or describe_visual
            record = TaskRecord(
                id=task_id,
                url=url,
                language=_coerce_language(meta.get("requested_language"), self._settings.language),
                extract_visual=extract_visual,
                describe_visual=describe_visual,
                status="succeeded",
                progress=100,
                stage="done",
                created_at=created_at,
                updated_at=created_at,
                sequence=self._sequence,
                started_at=created_at,
                finished_at=created_at,
                output_dir=str(output_dir.resolve()),
                source=_clean_string(meta.get("source")),
            )
            record.files = self._scan_files(record.id, Path(record.output_dir))
            self._append_log(record, "info", "已从本地输出目录恢复任务")
            self._tasks[record.id] = record
            changed = True
        return changed

    def _import_job_files_locked(self) -> bool:
        changed = False
        output_root = self._settings.output_dir.expanduser().resolve()
        for job_path in sorted(self._jobs_dir.glob("*.json")):
            task_id = job_path.stem
            if task_id in self._tasks:
                continue
            try:
                job = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(job, dict):
                continue
            url = _clean_string(job.get("url"))
            if not url or any(record.url == url and record.status == "succeeded" for record in self._tasks.values()):
                continue
            try:
                job_output_root = Path(str(job.get("output_root") or "")).expanduser().resolve()
            except OSError:
                continue
            if job_output_root != output_root:
                continue

            now = _now()
            self._sequence += 1
            describe_visual = _coerce_bool(job.get("describe_visual"), False)
            extract_visual = _coerce_bool(job.get("extract_visual"), False) or describe_visual
            record = TaskRecord(
                id=task_id,
                url=url,
                language=_coerce_language(job.get("language"), self._settings.language),
                extract_visual=extract_visual,
                describe_visual=describe_visual,
                status="stopped",
                progress=0,
                stage="stopped",
                created_at=now,
                updated_at=now,
                sequence=self._sequence,
                finished_at=now,
                job_file=str(job_path),
            )
            self._append_log(record, "warning", "已从本地任务记录恢复，服务重启后标记为停止")
            self._tasks[record.id] = record
            changed = True
        return changed

    def _persist_tasks_locked(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "tasks": [
                self._task_storage_payload(record)
                for record in sorted(self._tasks.values(), key=lambda item: item.sequence)
            ],
        }
        temp_path = self._tasks_file.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self._tasks_file)

    def _task_storage_payload(self, record: TaskRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "url": record.url,
            "language": record.language,
            "extract_visual": record.extract_visual,
            "describe_visual": record.describe_visual,
            "status": record.status,
            "progress": record.progress,
            "stage": record.stage,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "sequence": record.sequence,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "output_dir": record.output_dir,
            "source": record.source,
            "error": record.error,
            "message": record.message,
            "logs": [entry.__dict__ for entry in record.logs],
            "job_file": record.job_file,
        }

    def _attach_existing_output_dir(self, record: TaskRecord, message: str) -> None:
        if not message.startswith(EXISTING_OUTPUT_PREFIX):
            return
        raw_path = message.removeprefix(EXISTING_OUTPUT_PREFIX).strip()
        if not raw_path:
            return
        output_dir = Path(raw_path).expanduser()
        if not output_dir.exists():
            return
        record.output_dir = str(output_dir.resolve())
        record.files = self._scan_files(record.id, Path(record.output_dir))

    def _remove_output_dir(self, output_dir: Path) -> None:
        target = output_dir.expanduser()
        if not target.exists():
            return
        resolved_target = target.resolve()
        output_root = self._settings.output_dir.expanduser().resolve()
        if resolved_target == output_root:
            raise HTTPException(status_code=409, detail="拒绝删除输出根目录")
        try:
            resolved_target.relative_to(output_root)
        except ValueError as exc:
            if not self._looks_like_extraction_dir(resolved_target):
                raise HTTPException(status_code=409, detail="输出目录不在当前配置目录下，未删除文件") from exc

        if resolved_target.is_dir():
            shutil.rmtree(resolved_target)
        else:
            resolved_target.unlink(missing_ok=True)

    @staticmethod
    def _looks_like_extraction_dir(path: Path) -> bool:
        return path.is_dir() and any(
            (path / filename).exists()
            for filename in ("meta.json", "subtitle.srt", "transcript.txt", "transcript.json")
        )

    @staticmethod
    def _terminate_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        pid = getattr(process, "pid", None)
        if os.name != "nt" and pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        try:
            process.terminate()
        except ProcessLookupError:
            pass

    @staticmethod
    def _kill_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        pid = getattr(process, "pid", None)
        if os.name != "nt" and pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        try:
            process.kill()
        except ProcessLookupError:
            pass

    def _handle_progress_line(self, record: TaskRecord, line: str) -> bool:
        progress = _parse_ytdlp_download_progress(line)
        if progress is None:
            return False

        percent = progress["percent"]
        if record.stage == "download_video":
            if record.extract_visual:
                stage_start, stage_end = 38, 54
            else:
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


def _language_label(language: str) -> str:
    return {
        "auto": "自动检测",
        "zh": "中文",
        "en": "英文",
    }.get(language, language)


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


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return fallback


def _hide_from_file_list(relative_path: str) -> bool:
    return (
        relative_path == "frames.json"
        or relative_path == "visual_description_cache.json"
        or relative_path.startswith("frames/")
    )


def _coerce_language(value: Any, fallback: Language) -> Language:
    text = _clean_string(value)
    if text in {"auto", "zh", "en"}:
        return text  # type: ignore[return-value]
    return fallback


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
