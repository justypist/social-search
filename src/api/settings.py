from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from social_extract.models import Device, Language


ENV_DEFAULTS: dict[str, str] = {
    "SOCIAL_SEARCH_HOST": "127.0.0.1",
    "SOCIAL_SEARCH_PORT": "8000",
    "SOCIAL_SEARCH_CONCURRENCY": "2",
    "SOCIAL_SEARCH_OUTPUT_DIR": "out",
    "SOCIAL_SEARCH_LANGUAGE": "auto",
    "SOCIAL_SEARCH_MODEL": "medium",
    "SOCIAL_SEARCH_DEVICE": "auto",
    "SOCIAL_SEARCH_COMPUTE_TYPE": "auto",
    "SOCIAL_SEARCH_VAD_FILTER": "false",
    "SOCIAL_SEARCH_KEEP_MEDIA": "true",
    "SOCIAL_SEARCH_OVERWRITE": "false",
    "SOCIAL_SEARCH_TASK_LOG_LIMIT": "500",
    "SOCIAL_SEARCH_ALLOWED_ORIGINS": "http://localhost:8000,http://127.0.0.1:8000",
    "SOCIAL_SEARCH_HTTP_HEADERS": "",
}


@dataclass(frozen=True)
class WebSettings:
    host: str
    port: int
    concurrency: int
    output_dir: Path
    language: Language
    model: str
    device: Device
    compute_type: str
    vad_filter: bool
    keep_media: bool
    overwrite: bool
    task_log_limit: int
    allowed_origins: list[str]
    http_headers: dict[str, str]
    env_file: Path


def load_settings(env_file: Path | None = None) -> WebSettings:
    env_path = env_file or Path(os.environ.get("SOCIAL_SEARCH_ENV_FILE", ".env"))
    env_path = env_path.expanduser()
    _ensure_env_file(env_path)
    _load_dotenv(env_path)

    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    language = _choice("SOCIAL_SEARCH_LANGUAGE", {"auto", "zh", "en"})
    device = _choice("SOCIAL_SEARCH_DEVICE", {"auto", "cuda", "cpu"})

    return WebSettings(
        host=os.environ["SOCIAL_SEARCH_HOST"],
        port=_int("SOCIAL_SEARCH_PORT", minimum=1, maximum=65535),
        concurrency=_int("SOCIAL_SEARCH_CONCURRENCY", minimum=1),
        output_dir=Path(os.environ["SOCIAL_SEARCH_OUTPUT_DIR"]).expanduser(),
        language=language,  # type: ignore[arg-type]
        model=os.environ["SOCIAL_SEARCH_MODEL"],
        device=device,  # type: ignore[arg-type]
        compute_type=os.environ["SOCIAL_SEARCH_COMPUTE_TYPE"],
        vad_filter=_bool("SOCIAL_SEARCH_VAD_FILTER"),
        keep_media=_bool("SOCIAL_SEARCH_KEEP_MEDIA"),
        overwrite=_bool("SOCIAL_SEARCH_OVERWRITE"),
        task_log_limit=_int("SOCIAL_SEARCH_TASK_LOG_LIMIT", minimum=1),
        allowed_origins=_csv(os.environ["SOCIAL_SEARCH_ALLOWED_ORIGINS"]),
        http_headers=_headers(os.environ["SOCIAL_SEARCH_HTTP_HEADERS"]),
        env_file=env_path.resolve(),
    )


def _ensure_env_file(env_path: Path) -> None:
    if env_path.exists():
        return
    example_path = env_path.with_name(".env.example")
    if example_path.exists():
        shutil.copyfile(example_path, env_path)


def _load_dotenv(env_path: Path) -> None:
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


def _bool(key: str) -> bool:
    value = os.environ[key].strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean")


def _int(key: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(os.environ[key])
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    return value


def _choice(key: str, choices: set[str]) -> str:
    value = os.environ[key].strip()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{key} must be one of: {allowed}")
    return value


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _headers(value: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not value.strip():
        return headers
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        name, separator, header_value = item.partition(":")
        name = name.strip()
        header_value = header_value.strip()
        if not separator or not name:
            raise ValueError("SOCIAL_SEARCH_HTTP_HEADERS must use Name:Value pairs separated by semicolons")
        headers[name] = header_value
    return headers
