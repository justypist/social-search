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
