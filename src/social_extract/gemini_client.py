from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from http.cookiejar import LoadError, MozillaCookieJar
from pathlib import Path
from typing import Any

from .env import load_dotenv, parse_cookie_files, resolve_env_path

GEMINI_COOKIE_NAMES = frozenset({"__Secure-1PSID", "__Secure-1PSIDTS"})


class GeminiConfigurationError(ValueError):
    """Raised when Gemini SDK credentials cannot be loaded from local settings."""


@dataclass(frozen=True)
class GeminiCredentials:
    secure_1psid: str
    secure_1psidts: str | None = None


def load_gemini_credentials(cookie_files: Iterable[Path | str]) -> GeminiCredentials:
    paths = tuple(Path(cookie_file).expanduser() for cookie_file in cookie_files)
    if not paths:
        raise GeminiConfigurationError(
            "SOCIAL_SEARCH_COOKIES must include a cookies.txt file containing __Secure-1PSID"
        )

    values: dict[str, str] = {}
    for path in paths:
        values.update(_read_gemini_cookie_values(path))

    secure_1psid = values.get("__Secure-1PSID")
    if not secure_1psid:
        raise GeminiConfigurationError(
            "Could not find __Secure-1PSID in SOCIAL_SEARCH_COOKIES files"
        )

    return GeminiCredentials(
        secure_1psid=secure_1psid,
        secure_1psidts=values.get("__Secure-1PSIDTS") or None,
    )


def load_gemini_credentials_from_env(env_file: Path | None = None) -> GeminiCredentials:
    env_path = resolve_env_path(env_file)
    load_dotenv(env_path)
    return load_gemini_credentials(
        parse_cookie_files(os.environ.get("SOCIAL_SEARCH_COOKIES", ""))
    )


def create_gemini_client(
    *,
    cookie_files: Iterable[Path | str] | None = None,
    env_file: Path | None = None,
    proxy: str | None = None,
    client_cls: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    credentials = (
        load_gemini_credentials_from_env(env_file)
        if cookie_files is None
        else load_gemini_credentials(cookie_files)
    )

    if client_cls is None:
        from gemini_webapi import GeminiClient

        client_cls = GeminiClient

    return client_cls(
        credentials.secure_1psid,
        credentials.secure_1psidts,
        proxy=proxy,
        **kwargs,
    )


def _read_gemini_cookie_values(path: Path) -> dict[str, str]:
    jar = MozillaCookieJar(str(path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except FileNotFoundError as exc:
        raise GeminiConfigurationError(f"Cookie file does not exist: {path}") from exc
    except (LoadError, OSError) as exc:
        raise GeminiConfigurationError(f"Could not read cookie file: {path}") from exc

    values: dict[str, str] = {}
    for cookie in jar:
        if cookie.name in GEMINI_COOKIE_NAMES and _is_google_cookie_domain(cookie.domain):
            values[cookie.name] = cookie.value
    return values


def _is_google_cookie_domain(domain: str) -> bool:
    normalized = domain.lstrip(".").lower()
    return normalized == "google.com" or normalized.endswith(".google.com")


__all__ = [
    "GeminiConfigurationError",
    "GeminiCredentials",
    "create_gemini_client",
    "load_gemini_credentials",
    "load_gemini_credentials_from_env",
]
