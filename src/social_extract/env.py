from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_env_path(env_file: Path | None = None) -> Path:
    return (env_file or Path(os.environ.get("SOCIAL_SEARCH_ENV_FILE", ".env"))).expanduser()


def ensure_env_file(env_path: Path) -> None:
    if env_path.exists():
        return
    example_path = env_path.with_name(".env.example")
    if example_path.exists():
        shutil.copyfile(example_path, env_path)


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = _strip_quotes(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_cookie_files(value: str) -> tuple[Path, ...]:
    return tuple(Path(item.strip()).expanduser() for item in value.split(";") if item.strip())
