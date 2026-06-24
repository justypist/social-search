from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from social_extract.gemini_client import (
    GeminiConfigurationError,
    create_gemini_client,
    load_gemini_credentials,
)


class StubGeminiClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def test_loads_gemini_credentials_from_netscape_cookie_file(tmp_path: Path) -> None:
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        "\n".join(
            [
                "# Netscape HTTP Cookie File",
                ".youtube.com\tTRUE\t/\tTRUE\t0\tSID\tyoutube",
                "#HttpOnly_.google.com\tTRUE\t/\tTRUE\t0\t__Secure-1PSID\tpsid",
                ".google.com\tTRUE\t/\tTRUE\t0\t__Secure-1PSIDTS\tpsidts",
            ]
        ),
        encoding="utf-8",
    )

    credentials = load_gemini_credentials((cookie_file,))

    assert credentials.secure_1psid == "psid"
    assert credentials.secure_1psidts == "psidts"


def test_loads_gemini_credentials_from_multiple_cookie_files(tmp_path: Path) -> None:
    first_cookie_file = tmp_path / "first-cookies.txt"
    second_cookie_file = tmp_path / "second-cookies.txt"
    first_cookie_file.write_text(
        "# Netscape HTTP Cookie File\n.google.com\tTRUE\t/\tTRUE\t0\t__Secure-1PSID\tpsid\n",
        encoding="utf-8",
    )
    second_cookie_file.write_text(
        "# Netscape HTTP Cookie File\n.google.com\tTRUE\t/\tTRUE\t0\t__Secure-1PSIDTS\tpsidts\n",
        encoding="utf-8",
    )

    credentials = load_gemini_credentials((first_cookie_file, second_cookie_file))

    assert credentials.secure_1psid == "psid"
    assert credentials.secure_1psidts == "psidts"


def test_missing_required_gemini_cookie_raises_configuration_error(tmp_path: Path) -> None:
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        "# Netscape HTTP Cookie File\n.google.com\tTRUE\t/\tTRUE\t0\t__Secure-1PSIDTS\tpsidts\n",
        encoding="utf-8",
    )

    with pytest.raises(GeminiConfigurationError, match="__Secure-1PSID"):
        load_gemini_credentials((cookie_file,))


def test_create_gemini_client_uses_social_search_cookies_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cookie_file = tmp_path / "cookies.txt"
    env_file = tmp_path / ".env"
    cookie_file.write_text(
        "\n".join(
            [
                "# Netscape HTTP Cookie File",
                ".google.com\tTRUE\t/\tTRUE\t0\t__Secure-1PSID\tpsid",
                ".google.com\tTRUE\t/\tTRUE\t0\t__Secure-1PSIDTS\tpsidts",
            ]
        ),
        encoding="utf-8",
    )
    env_file.write_text(f"SOCIAL_SEARCH_COOKIES={cookie_file}\n", encoding="utf-8")
    monkeypatch.delenv("SOCIAL_SEARCH_COOKIES", raising=False)

    client = create_gemini_client(
        env_file=env_file,
        proxy="http://proxy.example",
        client_cls=StubGeminiClient,
        verify=False,
    )

    assert client.args == ("psid", "psidts")
    assert client.kwargs == {"proxy": "http://proxy.example", "verify": False}
