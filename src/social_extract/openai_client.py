from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .env import load_dotenv, resolve_env_path


def create_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    env_file: Path | None = None,
    **kwargs: Any,
) -> OpenAI:
    if env_file is not None:
        env_path = resolve_env_path(env_file)
        load_dotenv(env_path)

    resolved_api_key = (
        api_key
        or os.environ.get("SOCIAL_SEARCH_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    resolved_base_url = (
        base_url
        or os.environ.get("SOCIAL_SEARCH_OPENAI_BASE_URL")
        or None
    )

    if not resolved_api_key:
        raise ValueError(
            "OpenAI API key is required. Set SOCIAL_SEARCH_OPENAI_API_KEY or OPENAI_API_KEY."
        )

    client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client_kwargs.update(kwargs)

    return OpenAI(**client_kwargs)


__all__ = [
    "create_openai_client",
]
