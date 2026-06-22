from social_extract.formats import format_srt_timestamp, subtitle_text_to_transcript


def test_parse_vtt_and_strip_markup() -> None:
    transcript = subtitle_text_to_transcript(
        """WEBVTT

00:00:01.000 --> 00:00:03.500 align:start
<v Speaker>你好 &amp; hello</v>

00:01:00.000 --> 00:01:02.000
second line
""",
        "vtt",
        "zh",
    )

    assert transcript.language == "zh"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].start == 1.0
    assert transcript.segments[0].end == 3.5
    assert transcript.segments[0].text == "你好 & hello"


def test_parse_srt_timestamp_and_text() -> None:
    transcript = subtitle_text_to_transcript(
        """1
00:00:01,200 --> 00:00:02,500
hello

""",
        "srt",
        "en",
    )

    assert transcript.segments[0].start == 1.2
    assert transcript.segments[0].end == 2.5
    assert transcript.segments[0].text == "hello"


def test_format_srt_timestamp() -> None:
    assert format_srt_timestamp(3661.2345) == "01:01:01,234"
