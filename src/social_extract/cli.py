from __future__ import annotations

import os
from contextlib import nullcontext
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .core import extract_subtitles
from .env import ensure_env_file, load_dotenv, parse_cookie_files, resolve_env_path
from .errors import ExtractionError
from .models import ExtractConfig
from .progress import ProgressCallback


class LanguageOption(StrEnum):
    auto = "auto"
    zh = "zh"
    en = "en"


class DeviceOption(StrEnum):
    auto = "auto"
    cuda = "cuda"
    cpu = "cpu"


app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def main(
    url: Annotated[str, typer.Argument(help="Video URL to extract subtitles from.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output root directory.")] = Path("out"),
    lang: Annotated[LanguageOption, typer.Option("--lang", help="Subtitle/transcription language.")] = LanguageOption.auto,
    model: Annotated[str, typer.Option("--model", help="faster-whisper model name or path.")] = "medium",
    device: Annotated[DeviceOption, typer.Option("--device", help="Whisper device.")] = DeviceOption.auto,
    compute_type: Annotated[str, typer.Option("--compute-type", help="Whisper compute type.")] = "auto",
    vad_filter: Annotated[
        bool,
        typer.Option("--vad-filter/--no-vad-filter", help="Enable faster-whisper VAD pre-filtering."),
    ] = False,
    keep_media: Annotated[
        bool,
        typer.Option("--keep-media/--no-keep-media", help="Keep downloaded audio/video files."),
    ] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an existing output directory.")] = False,
    add_header: Annotated[
        list[str] | None,
        typer.Option("--add-header", help="Extra yt-dlp HTTP header as Name:Value. May be repeated."),
    ] = None,
    show_progress: Annotated[
        bool,
        typer.Option("--progress/--no-progress", help="Show live extraction progress."),
    ] = True,
) -> None:
    env_values = _load_env_values()
    headers = env_values.http_headers
    headers.update(_parse_add_headers(add_header or []))
    config = ExtractConfig(
        output_root=output,
        language=lang.value,
        model=model,
        device=device.value,
        compute_type=compute_type,
        vad_filter=vad_filter,
        keep_media=keep_media,
        overwrite=overwrite,
        http_headers=headers,
        cookie_files=env_values.cookie_files,
        cookies_from_browser=env_values.cookies_from_browser,
    )

    status_context = console.status("准备提取") if show_progress else nullcontext(None)
    with status_context as status:
        progress_callback = _console_progress(status) if show_progress else None
        try:
            result = extract_subtitles(url, config, progress_callback=progress_callback)
        except ExtractionError as exc:
            console.print(f"[red]Extraction failed:[/red] {exc}")
            raise typer.Exit(1) from exc

    console.print(f"[green]Extracted subtitles[/green] via [bold]{result.source}[/bold]")
    console.print(f"Output: {result.output_dir}")
    console.print(f"SRT: {result.subtitle_path}")
    console.print(f"Paragraph SRT: {result.paragraph_subtitle_path}")
    console.print(f"Text: {result.transcript_text_path}")
    console.print(f"Metadata: {result.meta_path}")


def _console_progress(status: object) -> ProgressCallback:
    def progress(stage: str, message: str, progress_value: float | None) -> None:
        del stage
        suffix = "" if progress_value is None else f" {progress_value * 100:.0f}%"
        status.update(f"{message}{suffix}")  # type: ignore[attr-defined]

    return progress


def _parse_add_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        name, separator, header_value = value.partition(":")
        name = name.strip()
        header_value = header_value.strip()
        if not separator or not name:
            raise typer.BadParameter("--add-header must use the form Name:Value")
        headers[name] = header_value
    return headers


class _EnvValues:
    def __init__(
        self,
        *,
        http_headers: dict[str, str],
        cookie_files: tuple[Path, ...],
        cookies_from_browser: str | None,
    ) -> None:
        self.http_headers = http_headers
        self.cookie_files = cookie_files
        self.cookies_from_browser = cookies_from_browser


def _load_env_values() -> _EnvValues:
    env_path = resolve_env_path()
    ensure_env_file(env_path)
    load_dotenv(env_path)
    return _EnvValues(
        http_headers=_parse_env_headers(os.environ.get("SOCIAL_SEARCH_HTTP_HEADERS", "")),
        cookie_files=parse_cookie_files(os.environ.get("SOCIAL_SEARCH_COOKIES", "")),
        cookies_from_browser=_string_or_none(os.environ.get("SOCIAL_SEARCH_COOKIES_FROM_BROWSER", "")),
    )


def _parse_env_headers(value: str) -> dict[str, str]:
    try:
        return _parse_add_headers([item.strip() for item in value.split(";") if item.strip()])
    except typer.BadParameter as exc:
        raise typer.BadParameter("SOCIAL_SEARCH_HTTP_HEADERS must use Name:Value pairs separated by semicolons") from exc


def _string_or_none(value: str) -> str | None:
    text = value.strip()
    return text or None
