from __future__ import annotations

from pathlib import Path

from social_extract.cli import _load_env_values


def test_cli_loads_ytdlp_settings_from_env_file(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    youtube_cookie_file = tmp_path / "youtube-cookies.txt"
    bilibili_cookie_file = tmp_path / "bilibili-cookies.txt"
    env_file.write_text(
        "\n".join(
            [
                "SOCIAL_SEARCH_HTTP_HEADERS=User-Agent:Example;X-Test:1",
                f"SOCIAL_SEARCH_COOKIES={youtube_cookie_file};{bilibili_cookie_file}",
                "SOCIAL_SEARCH_COOKIES_FROM_BROWSER=chrome:Default",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SOCIAL_SEARCH_ENV_FILE", str(env_file))
    monkeypatch.delenv("SOCIAL_SEARCH_HTTP_HEADERS", raising=False)
    monkeypatch.delenv("SOCIAL_SEARCH_COOKIES", raising=False)
    monkeypatch.delenv("SOCIAL_SEARCH_COOKIES_FROM_BROWSER", raising=False)

    values = _load_env_values()

    assert values.http_headers == {"User-Agent": "Example", "X-Test": "1"}
    assert values.cookie_files == (youtube_cookie_file, bilibili_cookie_file)
    assert values.cookies_from_browser == "chrome:Default"
