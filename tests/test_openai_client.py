from __future__ import annotations

from pathlib import Path

import pytest

from social_extract.openai_client import create_openai_client


def test_create_openai_client_uses_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SOCIAL_SEARCH_OPENAI_API_KEY=sk-test-key\n"
        "SOCIAL_SEARCH_OPENAI_BASE_URL=https://proxy.example\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SOCIAL_SEARCH_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SOCIAL_SEARCH_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = create_openai_client(env_file=env_file)

    assert client.api_key == "sk-test-key"
    assert str(client.base_url) == "https://proxy.example"


def test_create_openai_client_uses_fallback_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-fallback\n", encoding="utf-8")
    monkeypatch.delenv("SOCIAL_SEARCH_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = create_openai_client(env_file=env_file)

    assert client.api_key == "sk-fallback"


def test_create_openai_client_missing_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.delenv("SOCIAL_SEARCH_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key is required"):
        create_openai_client(env_file=env_file)


def test_create_openai_client_explicit_args_override_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SOCIAL_SEARCH_OPENAI_API_KEY=sk-env-key\n"
        "SOCIAL_SEARCH_OPENAI_BASE_URL=https://env-proxy.example\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SOCIAL_SEARCH_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SOCIAL_SEARCH_OPENAI_BASE_URL", raising=False)

    client = create_openai_client(
        api_key="sk-explicit",
        base_url="https://explicit-proxy.example",
        env_file=env_file,
    )

    assert client.api_key == "sk-explicit"
    assert str(client.base_url) == "https://explicit-proxy.example"
