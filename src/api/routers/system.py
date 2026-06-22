from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ..dependencies import get_task_manager
from ..task_manager import TaskManager


router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/config")
async def config(manager: TaskManager = Depends(get_task_manager)) -> dict[str, Any]:
    return manager.settings_payload()
