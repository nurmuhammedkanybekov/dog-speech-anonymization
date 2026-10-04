"""Tests for vad_anonymization.detection.

SpeechDetector.detect() itself is exercised as a manual, documented
smoke test in the author's own working notes rather than here: it
requires downloading the real Silero VAD model and produces results
that depend on real audio content, which makes it an integration check
rather than a fast, deterministic unit test. What's unit-tested here is
the pure logic that doesn't need the model at all.
"""

from __future__ import annotations

from vad_anonymization.detection import SpeechSegment, total_speech_duration


def test_speech_segment_duration():
    segment = SpeechSegment(start_s=2.0, end_s=5.5)
    assert segment.duration_s == 3.5


def test_total_speech_duration_sums_all_segments():
    segments = [
        SpeechSegment(0.0, 2.0),
        SpeechSegment(3.0, 4.5),
        SpeechSegment(10.0, 10.25),
    ]
    assert total_speech_duration(segments) == 2.0 + 1.5 + 0.25


def test_total_speech_duration_empty_list_is_zero():
    assert total_speech_duration([]) == 0.0
