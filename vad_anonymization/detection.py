"""A thin, reusable wrapper around Silero VAD.

Silero VAD is a small pretrained Voice Activity Detector: given an
audio clip, it reports which time windows contain a human talking. It
has no awareness of *what* is said, and no awareness of dogs, noise,
or anything else - it only distinguishes "speech" from "not speech".
That single, narrow capability is exactly what Task 1 (anonymization)
needs as its first stage: find the speech, then suppress it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from silero_vad import get_speech_timestamps, load_silero_vad


@dataclass(frozen=True)
class SpeechSegment:
    """A single detected speech interval, in seconds."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Length of the segment, in seconds."""
        return self.end_s - self.start_s


class SpeechDetector:
    """Loads the Silero VAD model once and reuses it across clips.

    Loading the model has a fixed cost (a small one-time download on
    first use, plus initialization); wrapping it in a class lets a
    script detect speech in several clips without repeating that cost
    for each one.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """``threshold`` is Silero's speech-probability cut-off (its default 0.5).

        Lower = catches more speech but also more non-speech (e.g. dogs);
        higher = fewer false alarms but more missed speech.
        """
        self._model = load_silero_vad()
        self.threshold = threshold

    def detect(self, waveform: torch.Tensor) -> list[SpeechSegment]:
        """Detect speech segments in a mono waveform.

        Args:
            waveform: 1-D float tensor of audio samples, sampled at the
                rate Silero VAD expects (16 kHz).

        Returns:
            Detected speech segments, in chronological order.
        """
        raw_segments = get_speech_timestamps(waveform, self._model, threshold=self.threshold, return_seconds=True)
        return [SpeechSegment(start_s=seg["start"], end_s=seg["end"]) for seg in raw_segments]


def total_speech_duration(segments: list[SpeechSegment]) -> float:
    """Sum the duration of a list of speech segments, in seconds."""
    return sum(seg.duration_s for seg in segments)
