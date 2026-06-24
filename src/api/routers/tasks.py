from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..dependencies import get_task_manager
from ..task_manager import TaskManager


router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class CreateTaskRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)
    language: Literal["auto", "zh", "en"] | None = None
    extract_visual: bool = False
    describe_visual: bool = False


@router.get("")
async def list_tasks(manager: TaskManager = Depends(get_task_manager)) -> dict[str, Any]:
    return {"tasks": await manager.list_tasks()}


@router.post("", status_code=201)
async def create_task(
    request: CreateTaskRequest,
    manager: TaskManager = Depends(get_task_manager),
) -> dict[str, Any]:
    return {
        "task": await manager.create_task(
            request.url,
            language=request.language,
            extract_visual=request.extract_visual,
            describe_visual=request.describe_visual,
        )
    }


@router.get("/{task_id}")
async def get_task(task_id: str, manager: TaskManager = Depends(get_task_manager)) -> dict[str, Any]:
    return {"task": await manager.get_task(task_id)}


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str, manager: TaskManager = Depends(get_task_manager)) -> dict[str, Any]:
    return {"task": await manager.cancel_task(task_id)}


@router.post("/{task_id}/stop")
async def stop_task(task_id: str, manager: TaskManager = Depends(get_task_manager)) -> dict[str, Any]:
    return {"task": await manager.stop_task(task_id)}


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: str, manager: TaskManager = Depends(get_task_manager)) -> None:
    await manager.delete_task(task_id)


@router.get("/{task_id}/files/{file_path:path}")
async def download_file(
    task_id: str,
    file_path: str,
    manager: TaskManager = Depends(get_task_manager),
) -> FileResponse:
    path = manager.get_file(task_id, file_path)
    return FileResponse(path, filename=path.name)
