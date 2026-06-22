from social_extract.subtitles import select_subtitle


def test_selects_manual_srt_before_automatic_vtt() -> None:
    info = {
        "subtitles": {
            "en": [{"ext": "srt", "data": "manual"}],
        },
        "automatic_captions": {
            "en": [{"ext": "vtt", "data": "auto"}],
        },
    }

    subtitle = select_subtitle(info, "en")

    assert subtitle is not None
    assert subtitle.source == "manual"
    assert subtitle.ext == "srt"


def test_matches_chinese_language_variants() -> None:
    info = {
        "subtitles": {
            "zh-Hans": [{"ext": "vtt", "data": "caption"}],
        },
    }

    subtitle = select_subtitle(info, "zh")

    assert subtitle is not None
    assert subtitle.language == "zh-Hans"
