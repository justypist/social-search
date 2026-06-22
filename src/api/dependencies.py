from __future__ import annotations

from fastapi import Request

from .task_manager import TaskManager


def get_task_manager(request: Request) -> TaskManager:
    return request.app.state.task_manager
