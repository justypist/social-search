from __future__ import annotations

from api.worker import _build_config


def test_worker_config_defaults_extract_visual_false() -> None:
    config = _build_config(_job())

    assert config.extract_visual is False
    assert config.describe_visual is False


def test_worker_config_accepts_extract_visual_true() -> None:
    job = _job()
    job["extract_visual"] = True

    config = _build_config(job)

    assert config.extract_visual is True


def test_worker_config_accepts_describe_visual_true() -> None:
    job = _job()
    job["describe_visual"] = True

    config = _build_config(job)

    assert config.describe_visual is True


def _job() -> dict:
    return {
        "url": "https://example.test/video",
        "output_root": "out",
        "language": "auto",
        "model": "small",
        "device": "cpu",
        "compute_type": "int8",
        "vad_filter": False,
        "keep_media": True,
        "overwrite": True,
        "http_headers": {},
        "cookie_files": [],
        "cookies_from_browser": None,
    }
