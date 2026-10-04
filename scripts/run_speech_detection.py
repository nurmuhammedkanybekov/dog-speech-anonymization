"""Step 1: detect and mute human speech in an audio file.

This is the productionized version of the original ``vad_demo.py``
proof of concept (see the author's own working notes, Session 1, for
the original exploratory version and the two issues worked around along
the way - that script was superseded by this one and removed once its
logic was carried over verbatim). The logic is unchanged; what changed
is packaging it as reusable, tested, documented modules instead of one
linear script.

Usage:
    python -m scripts.run_speech_detection INPUT.wav OUTPUT.wav

Run from inside this directory so that the ``vad_anonymization``
package is importable.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from vad_anonymization.audio_io import load_audio, save_audio
from vad_anonymization.detection import SpeechDetector, SpeechSegment, total_speech_duration

SAMPLE_RATE = 16_000  # Silero VAD operates at 16 kHz.

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def mute_segments(waveform: torch.Tensor, segments: list[SpeechSegment], sample_rate: int) -> torch.Tensor:
    """Return a copy of ``waveform`` with each segment's samples zeroed."""
    muted = waveform.clone()
    for segment in segments:
        start_sample = int(segment.start_s * sample_rate)
        end_sample = int(segment.end_s * sample_rate)
        muted[start_sample:end_sample] = 0.0
    return muted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to the input .wav file")
    parser.add_argument("output", type=Path, help="Path to write the speech-muted .wav file to")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading %s", args.input)
    waveform = load_audio(args.input, SAMPLE_RATE)

    logger.info("Loading Silero VAD model...")
    detector = SpeechDetector()

    segments = detector.detect(waveform)
    logger.info("Detected %d speech segment(s):", len(segments))
    for segment in segments:
        logger.info("  %.2fs -> %.2fs", segment.start_s, segment.end_s)

    clip_duration_s = waveform.shape[-1] / SAMPLE_RATE
    speech_duration_s = total_speech_duration(segments)
    logger.info(
        "Total detected speech: %.2fs of %.2fs (%.0f%%)",
        speech_duration_s,
        clip_duration_s,
        100 * speech_duration_s / clip_duration_s if clip_duration_s else 0.0,
    )

    muted = mute_segments(waveform, segments, SAMPLE_RATE)
    save_audio(args.output, muted, SAMPLE_RATE)
    logger.info("Saved muted audio to %s", args.output)


if __name__ == "__main__":
    main()
