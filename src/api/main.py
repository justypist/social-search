from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import system, tasks
from .settings import WebSettings, load_settings
from .task_manager import PROJECT_ROOT, TaskManager


def create_app(settings: WebSettings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()
    manager = TaskManager(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await manager.start()
        yield
        await manager.shutdown()

    app = FastAPI(title="Social Search", lifespan=lifespan)
    app.state.task_manager = manager
    app.state.settings = resolved_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.allowed_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(tasks.router)
    _mount_static_frontend(app, PROJECT_ROOT / "web")
    return app


def _mount_static_frontend(app: FastAPI, web_dir: Path) -> None:
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


app = create_app()


def main() -> None:
    import uvicorn

    settings = app.state.settings
    uvicorn.run(app, host=settings.host, port=settings.port)
