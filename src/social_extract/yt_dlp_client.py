from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .errors import ExtractionError
from .models import SubtitleRef


DownloadProgressCallback = Callable[[float | None, str], None]


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

    def download_audio(
        self,
        url: str,
        output_dir: Path,
        *,
        progress_callback: DownloadProgressCallback | None = None,
    ) -> Path:
        return self._download_media(
            url,
            output_dir,
            prefix="audio",
            format_spec="bestaudio/best",
            progress_callback=progress_callback,
        )

    def download_video(
        self,
        url: str,
        output_dir: Path,
        *,
        progress_callback: DownloadProgressCallback | None = None,
    ) -> Path:
        return self._download_media(
            url,
            output_dir,
            prefix="video",
            format_spec="bestvideo+bestaudio/best",
            progress_callback=progress_callback,
        )

    def _download_media(
        self,
        url: str,
        output_dir: Path,
        prefix: str,
        format_spec: str,
        *,
        progress_callback: DownloadProgressCallback | None = None,
    ) -> Path:
        options = self._base_options(skip_download=False, url=url)
        options.update(
            {
                "format": format_spec,
                "outtmpl": str(output_dir / f"{prefix}.%(ext)s"),
                "overwrites": True,
            }
        )
        if progress_callback is not None:
            options["progress_hooks"] = [_download_progress_hook(progress_callback)]
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


def _download_progress_hook(callback: DownloadProgressCallback) -> Callable[[dict[str, Any]], None]:
    last_percent: float | None = None
    last_emit = 0.0

    def hook(payload: dict[str, Any]) -> None:
        nonlocal last_percent, last_emit

        status = payload.get("status")
        if status == "finished":
            callback(100.0, "下载完成")
            last_percent = 100.0
            last_emit = time.monotonic()
            return
        if status != "downloading":
            return

        percent = _download_percent(payload)
        now = time.monotonic()
        if not _should_emit_download_progress(percent, last_percent, now, last_emit):
            return

        callback(percent, _format_download_progress(payload, percent))
        last_percent = percent
        last_emit = now

    return hook


def _should_emit_download_progress(
    percent: float | None,
    last_percent: float | None,
    now: float,
    last_emit: float,
) -> bool:
    if last_emit == 0.0:
        return True
    if percent is None:
        return now - last_emit >= 1.0
    if percent >= 100:
        return True
    if last_percent is None:
        return True
    if percent - last_percent >= 1.0:
        return True
    return now - last_emit >= 1.5


def _download_percent(payload: dict[str, Any]) -> float | None:
    total = payload.get("total_bytes") or payload.get("total_bytes_estimate")
    downloaded = payload.get("downloaded_bytes")
    if not isinstance(total, int | float) or total <= 0:
        return None
    if not isinstance(downloaded, int | float):
        return None
    return max(0.0, min(100.0, downloaded / total * 100))


def _format_download_progress(payload: dict[str, Any], percent: float | None) -> str:
    pieces = [f"下载中 {percent:.1f}%" if percent is not None else "下载中"]
    downloaded = _format_bytes(payload.get("downloaded_bytes"))
    total = _format_bytes(payload.get("total_bytes") or payload.get("total_bytes_estimate"))
    if downloaded and total:
        pieces.append(f"{downloaded}/{total}")
    elif downloaded:
        pieces.append(downloaded)

    speed = _format_bytes(payload.get("speed"))
    if speed:
        pieces.append(f"{speed}/s")

    eta = _format_duration(payload.get("eta"))
    if eta:
        pieces.append(f"ETA {eta}")

    return " | ".join(pieces)


def _format_bytes(value: Any) -> str | None:
    if not isinstance(value, int | float) or value <= 0:
        return None
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{size:.0f}{units[unit_index]}"
    return f"{size:.1f}{units[unit_index]}"


def _format_duration(value: Any) -> str | None:
    if not isinstance(value, int | float) or value < 0:
        return None
    seconds = int(value)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    if hour:
        return f"{hour:d}:{minute:02d}:{second:02d}"
    return f"{minute:02d}:{second:02d}"


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
