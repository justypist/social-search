from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .errors import ExtractionError
from .models import SubtitleRef


class YtDlpClient:
    def __init__(self, http_headers: dict[str, str] | None = None) -> None:
        self._http_headers = http_headers or {}

    def probe(self, url: str) -> dict[str, Any]:
        try:
            with YoutubeDL(self._base_options(skip_download=True, url=url)) as ydl:
                info = ydl.extract_info(url, download=False)
        except DownloadError as exc:
            raise ExtractionError(f"yt-dlp failed to inspect URL: {exc}") from exc

        if not isinstance(info, dict):
            raise ExtractionError("yt-dlp returned an unsupported info payload")
        if info.get("_type") == "playlist" or info.get("entries"):
            raise ExtractionError("Playlists are not supported in the first CLI version")
        return info

    def download_subtitle_text(self, subtitle: SubtitleRef) -> str:
        if subtitle.data:
            return subtitle.data
        if not subtitle.url:
            raise ExtractionError("Selected subtitle has no data or URL")

        try:
            with YoutubeDL(self._base_options(skip_download=True, url=subtitle.url)) as ydl:
                response = ydl.urlopen(subtitle.url)
                payload = response.read()
        except Exception as exc:  # yt-dlp can surface transport errors from several libraries.
            raise ExtractionError(f"Failed to download subtitle: {exc}") from exc

        return _decode_text(payload)

    def download_audio(self, url: str, output_dir: Path) -> Path:
        return self._download_media(url, output_dir, prefix="audio", format_spec="bestaudio/best")

    def download_video(self, url: str, output_dir: Path) -> Path:
        return self._download_media(
            url,
            output_dir,
            prefix="video",
            format_spec="bestvideo+bestaudio/best",
        )

    def _download_media(self, url: str, output_dir: Path, prefix: str, format_spec: str) -> Path:
        options = self._base_options(skip_download=False, url=url)
        options.update(
            {
                "format": format_spec,
                "outtmpl": str(output_dir / f"{prefix}.%(ext)s"),
                "overwrites": True,
            }
        )
        try:
            with YoutubeDL(options) as ydl:
                ydl.download([url])
        except DownloadError as exc:
            raise ExtractionError(f"yt-dlp failed to download {prefix}: {exc}") from exc

        candidates = [
            path
            for path in output_dir.glob(f"{prefix}.*")
            if path.is_file() and path.suffix not in {".part", ".ytdl", ".json"}
        ]
        if not candidates:
            raise ExtractionError(f"yt-dlp finished without creating {prefix} media")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _base_options(self, skip_download: bool, url: str | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": skip_download,
        }
        headers = _default_headers_for_url(url)
        headers.update(self._http_headers)
        if headers:
            options["http_headers"] = headers
        return options


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _default_headers_for_url(url: str | None) -> dict[str, str]:
    if not url:
        return {}
    host = urlparse(url).hostname or ""
    if host == "bilibili.com" or host.endswith(".bilibili.com"):
        return {
            "Referer": "https://www.bilibili.com/",
            "Origin": "https://www.bilibili.com",
        }
    return {}
