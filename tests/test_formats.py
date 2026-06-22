from social_extract.formats import aggregate_transcript, format_srt_timestamp, subtitle_text_to_transcript
from social_extract.models import Segment, Transcript


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


def test_aggregate_transcript_groups_segments_into_minute_paragraphs() -> None:
    transcript = Transcript(
        language="zh",
        segments=[
            Segment(0.0, 12.0, "第一句"),
            Segment(18.0, 35.0, "第二句"),
            Segment(58.0, 62.0, "跨过一分钟"),
            Segment(70.0, 74.0, "下一段"),
            Segment(140.0, 145.0, "间隔较远"),
        ],
    )

    paragraphs = aggregate_transcript(transcript)

    assert paragraphs.language == "zh"
    assert paragraphs.segments == [
        Segment(0.0, 62.0, "第一句 第二句 跨过一分钟"),
        Segment(70.0, 74.0, "下一段"),
        Segment(140.0, 145.0, "间隔较远"),
    ]
