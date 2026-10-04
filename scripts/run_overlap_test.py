"""Step 2: does speech detection survive a dog barking at the same time?

This is the productionized version of the original ``overlap_test.py``
proof of concept (see the author's own working notes, Session 2, for
the full rationale - the
original script was superseded by this one and removed once its logic
was carried over verbatim). The test design is unchanged - only the
implementation has been split into reusable, tested modules and given
a machine-readable results output.

Test design
-----------
A) Speech + dog overlap. A stretch of real, continuous speech (ground
   truth: speech for its entire duration) with a real dog bark overlaid
   at a known offset. If detected speech coverage falls short of the
   excerpt's full duration - especially within the noise window - that
   is a real detection failure: a masked window of speech that would
   slip through anonymization un-suppressed.

B) Dog bark alone. Ground truth: zero speech. Any detected segment here
   is a false positive - the detector mistaking a bark for a voice on
   its own, independent of any real speech being present.

Usage:
    python -m scripts.run_overlap_test [options]

Run from inside this directory so that the ``vad_anonymization``
package is importable. Defaults point at the Session 1/2 sample files
in ../data/; override with flags to test different clips or timing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vad_anonymization.audio_io import load_audio, save_audio
from vad_anonymization.detection import SpeechDetector, total_speech_duration
from vad_anonymization.mixing import build_overlap_mixture

SAMPLE_RATE = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--speech",
        type=Path,
        default=DATA_DIR / "sample_speech.wav",
        help="Path to the source speech recording (default: project sample from Session 1)",
    )
    parser.add_argument(
        "--noise",
        type=Path,
        default=DATA_DIR / "dog_bark_esc50_1-100032-A-0.wav",
        help="Path to the noise clip to overlay, e.g. a dog bark (default: ESC-50 sample from Session 2)",
    )
    parser.add_argument(
        "--excerpt-start",
        type=float,
        default=20.30,
        help="Start of the speech excerpt within --speech, in seconds "
        "(default: 20.30s, the longest continuous-speech segment Session 1 detected)",
    )
    parser.add_argument(
        "--excerpt-end",
        type=float,
        default=32.60,
        help="End of the speech excerpt within --speech, in seconds (default: 32.60s)",
    )
    parser.add_argument(
        "--noise-offset",
        type=float,
        default=3.0,
        help="Where the noise clip starts within the excerpt, in seconds (default: 3.0s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs",
        help="Directory to write generated audio and results.json into",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    speech = load_audio(args.speech, SAMPLE_RATE)
    dog = load_audio(args.noise, SAMPLE_RATE)

    excerpt_start_sample = int(args.excerpt_start * SAMPLE_RATE)
    excerpt_end_sample = int(args.excerpt_end * SAMPLE_RATE)
    excerpt = speech[excerpt_start_sample:excerpt_end_sample]

    mixture = build_overlap_mixture(
        speech_excerpt=excerpt,
        noise_clip=dog,
        sample_rate=SAMPLE_RATE,
        noise_offset_s=args.noise_offset,
    )
    logger.info(
        "Built %.2fs mixture; noise overlaps %.2fs -> %.2fs",
        mixture.excerpt_duration_s,
        mixture.noise_start_s,
        mixture.noise_end_s,
    )

    mixture_path = args.output_dir / "overlap_speech_dog.wav"
    save_audio(mixture_path, mixture.waveform, SAMPLE_RATE)
    dog_only_path = args.output_dir / "dog_only_16k.wav"
    save_audio(dog_only_path, dog, SAMPLE_RATE)

    detector = SpeechDetector()

    logger.info("\n=== A) Speech + dog overlap ===")
    mix_segments = detector.detect(mixture.waveform)
    for segment in mix_segments:
        logger.info("  %.2fs -> %.2fs", segment.start_s, segment.end_s)
    covered_s = total_speech_duration(mix_segments)
    coverage_pct = 100 * covered_s / mixture.excerpt_duration_s if mixture.excerpt_duration_s else 0.0
    logger.info(
        "  Detected speech covers %.2fs of %.2fs excerpt (%.0f%%); ground truth is 100%%",
        covered_s,
        mixture.excerpt_duration_s,
        coverage_pct,
    )

    logger.info("\n=== B) Dog bark alone (ground truth: 0s of speech) ===")
    dog_segments = detector.detect(dog)
    if dog_segments:
        for segment in dog_segments:
            logger.info("  %.2fs -> %.2fs  <-- false positive", segment.start_s, segment.end_s)
    else:
        logger.info("  (none detected - correct)")

    results = {
        "excerpt_duration_s": mixture.excerpt_duration_s,
        "noise_start_s": mixture.noise_start_s,
        "noise_end_s": mixture.noise_end_s,
        "check_a_speech_plus_dog": {
            "detected_segments": [{"start_s": s.start_s, "end_s": s.end_s} for s in mix_segments],
            "covered_s": covered_s,
            "coverage_pct": coverage_pct,
            "ground_truth_pct": 100.0,
        },
        "check_b_dog_alone": {
            "detected_segments": [{"start_s": s.start_s, "end_s": s.end_s} for s in dog_segments],
            "false_positive": len(dog_segments) > 0,
        },
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("\nSaved outputs and results.json to %s", args.output_dir)


if __name__ == "__main__":
    main()
