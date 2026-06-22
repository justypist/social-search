from pathlib import Path

import pytest

from social_extract.errors import ExtractionError
from social_extract.yt_dlp_client import YtDlpClient


def test_bilibili_origin_header_is_added_by_default() -> None:
    options = YtDlpClient()._base_options(
        skip_download=True,
        url="https://www.bilibili.com/video/BV15E421A7tj/",
    )

    assert options["http_headers"]["Referer"] == "https://www.bilibili.com/"
    assert options["http_headers"]["Origin"] == "https://www.bilibili.com"


def test_custom_headers_override_site_defaults() -> None:
    options = YtDlpClient({"Origin": "https://example.test", "X-Test": "1"})._base_options(
        skip_download=True,
        url="https://www.bilibili.com/video/BV15E421A7tj/",
    )

    assert options["http_headers"]["Origin"] == "https://example.test"
    assert options["http_headers"]["X-Test"] == "1"


def test_cookie_file_is_passed_to_ytdlp_options(tmp_path: Path) -> None:
    cookie_file = tmp_path / "cookies.txt"

    options = YtDlpClient(cookie_file=cookie_file)._base_options(skip_download=True)

    assert options["cookiefile"] == str(cookie_file)


def test_probe_options_ignore_missing_download_formats() -> None:
    options = YtDlpClient()._base_options(skip_download=True)

    assert options["ignore_no_formats_error"] is True


def test_multiple_cookie_files_are_merged_for_ytdlp_options(tmp_path: Path) -> None:
    youtube_cookie_file = tmp_path / "youtube-cookies.txt"
    bilibili_cookie_file = tmp_path / "bilibili-cookies.txt"
    youtube_cookie_file.write_text(
        "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tyoutube\n",
        encoding="utf-8",
    )
    bilibili_cookie_file.write_text(
        "# Netscape HTTP Cookie File\n.bilibili.com\tTRUE\t/\tTRUE\t0\tSESSDATA\tbilibili\n",
        encoding="utf-8",
    )

    options = YtDlpClient(cookie_files=(youtube_cookie_file, bilibili_cookie_file))._base_options(
        skip_download=True
    )

    merged_cookie_file = options["cookiefile"]
    assert hasattr(merged_cookie_file, "read")
    assert "SID\tyoutube" in merged_cookie_file.read()
    merged_cookie_file.seek(0)
    assert "SESSDATA\tbilibili" in merged_cookie_file.read()


def test_cookies_from_browser_is_parsed_like_ytdlp_cli() -> None:
    options = YtDlpClient(cookies_from_browser="chrome+BASICTEXT:Default")._base_options(skip_download=True)

    assert options["cookiesfrombrowser"] == ("chrome", "Default", "BASICTEXT", None)


def test_invalid_cookies_from_browser_raises_extraction_error() -> None:
    with pytest.raises(ExtractionError, match="Unsupported cookies browser"):
        YtDlpClient(cookies_from_browser="unknown")._base_options(skip_download=True)
