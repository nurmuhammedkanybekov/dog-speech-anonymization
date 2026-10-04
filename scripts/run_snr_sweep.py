"""Step 3: how much dog noise does it take to break speech detection?

Follow-up to the Step 2 overlap test, following Dr. Kovacs's suggestion
to control the noise level explicitly with a signal-to-noise ratio (SNR)
instead of mixing clips at whatever loudness they happen to have.

For every noise clip and every SNR level, a speech excerpt is mixed with
the clip (centred, trimmed to fit - never padded) at that SNR, and two
things are measured:

* recall_in_window - the fraction of the time span where the dog is
  present that the detector still flags as speech. Ground truth there is
  "speech the whole time", so anything below the clean-speech baseline is
  speech the dog masked, i.e. speech that would leak through
  anonymization un-muted.
* false_alarm_s - seconds flagged as speech when the same scaled noise is
  played alone (ground truth: no speech at all).

Usage (from inside speech_anonymization/):
    python -m scripts.run_snr_sweep
    python -m scripts.run_snr_sweep --snr 20 10 0 -10 --output-dir outputs/snr_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

import torch

from vad_anonymization.audio_io import load_audio
from vad_anonymization.detection import SpeechDetector, SpeechSegment
from vad_anonymization.mixing import build_snr_mixture

SAMPLE_RATE = 16_000
GRID_S = 0.01  # 10 ms resolution for overlap bookkeeping
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speech", type=Path, default=DATA_DIR / "sample_speech.wav")
    parser.add_argument("--excerpt-start", type=float, default=20.30)
    parser.add_argument("--excerpt-end", type=float, default=32.60)
    parser.add_argument(
        "--noise-dirs",
        type=Path,
        nargs="+",
        default=[DATA_DIR / "esc50_dog", DATA_DIR / "audioset_candidates"],
        help="Folders whose .wav/.flac files are used as noise clips",
    )
    parser.add_argument("--snr", type=float, nargs="+", default=[20, 10, 0, -10, -20, -25, -30, -35, -40],
        help="SNR levels in dB (barks are intermittent, so the detector only starts failing far below 0 dB)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "snr_sweep"
    )
    return parser.parse_args()


def to_mask(segments: list[SpeechSegment], duration_s: float) -> torch.Tensor:
    """Boolean speech mask on a GRID_S grid."""
    n = int(round(duration_s / GRID_S))
    mask = torch.zeros(n, dtype=torch.bool)
    for seg in segments:
        mask[int(round(seg.start_s / GRID_S)) : int(round(seg.end_s / GRID_S))] = True
    return mask


def window_recall(mask: torch.Tensor, start_s: float, end_s: float) -> float:
    window = mask[int(round(start_s / GRID_S)) : int(round(end_s / GRID_S))]
    return float(window.float().mean().item()) if window.numel() else float("nan")


def source_of(path: Path) -> str:
    return "audioset" if "audioset" in path.name.lower() else "esc50"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    speech_full = load_audio(args.speech, SAMPLE_RATE)
    excerpt = speech_full[int(args.excerpt_start * SAMPLE_RATE) : int(args.excerpt_end * SAMPLE_RATE)]
    duration_s = excerpt.shape[-1] / SAMPLE_RATE

    noise_paths = sorted(p for d in args.noise_dirs for p in d.iterdir() if p.suffix.lower() in {".wav", ".flac"})
    logger.info("Speech excerpt: %.2fs | %d noise clips | SNR levels: %s dB", duration_s, len(noise_paths), args.snr)

    detector = SpeechDetector()
    clean_mask = to_mask(detector.detect(excerpt), duration_s)
    logger.info("Clean-speech baseline coverage: %.1f%%", 100 * clean_mask.float().mean().item())

    rows = []
    for i, path in enumerate(noise_paths, 1):
        noise = load_audio(path, SAMPLE_RATE)
        for snr in args.snr:
            mix = build_snr_mixture(excerpt, noise, SAMPLE_RATE, snr_db=snr)
            mix_mask = to_mask(detector.detect(mix.waveform), duration_s)
            fa_segments = detector.detect(mix.noise_only)
            fa_mask = to_mask(fa_segments, duration_s)
            rows.append(
                {
                    "clip": path.name,
                    "source": source_of(path),
                    "snr_db": snr,
                    "noise_gain_x": round(mix.noise_gain, 5),
                    "noise_start_s": round(mix.noise_start_s, 3),
                    "noise_end_s": round(mix.noise_end_s, 3),
                    "recall_in_window": round(window_recall(mix_mask, mix.noise_start_s, mix.noise_end_s), 4),
                    "clean_recall_in_window": round(window_recall(clean_mask, mix.noise_start_s, mix.noise_end_s), 4),
                    "coverage_excerpt": round(float(mix_mask.float().mean().item()), 4),
                    "false_alarm_s": round(sum(s.duration_s for s in fa_segments), 3),
                    # if the bark alone already triggers detection here, high recall in the
                    # window would partly be the dog, not the speech - so this is tracked too
                    "noise_alone_flagged_in_window": round(window_recall(fa_mask, mix.noise_start_s, mix.noise_end_s), 4),
                }
            )
        logger.info("[%d/%d] %s done", i, len(noise_paths), path.name)

    with open(args.output_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for source in ["all", "esc50", "audioset"]:
        for snr in args.snr:
            sel = [r for r in rows if r["snr_db"] == snr and (source == "all" or r["source"] == source)]
            if not sel:
                continue
            recalls = [r["recall_in_window"] for r in sel]
            summary.append(
                {
                    "source": source,
                    "snr_db": snr,
                    "n_clips": len(sel),
                    "mean_recall": round(statistics.mean(recalls), 4),
                    "median_recall": round(statistics.median(recalls), 4),
                    "min_recall": round(min(recalls), 4),
                    "share_recall_below_95": round(sum(r < 0.95 for r in recalls) / len(sel), 4),
                    "share_with_false_alarm": round(sum(r["false_alarm_s"] > 0 for r in sel) / len(sel), 4),
                    "mean_false_alarm_s": round(statistics.mean(r["false_alarm_s"] for r in sel), 3),
                }
            )
    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "speech": str(args.speech.name),
                "excerpt_s": [args.excerpt_start, args.excerpt_end],
                "snr_db": args.snr,
                "n_noise_clips": len(noise_paths),
                "noise_power": "active frames (within 30 dB of loudest 20 ms frame)",
                "speech_power": "whole excerpt",
                "clean_baseline_coverage": round(float(clean_mask.float().mean().item()), 4),
            },
            indent=2,
        )
    )

    logger.info("\n%-8s %6s %12s %10s %14s %12s", "source", "SNR", "mean recall", "min", "clips <95%", "false alarm")
    for s in summary:
        logger.info(
            "%-8s %6.0f %11.1f%% %9.1f%% %13.0f%% %11.0f%%",
            s["source"],
            s["snr_db"],
            100 * s["mean_recall"],
            100 * s["min_recall"],
            100 * s["share_recall_below_95"],
            100 * s["share_with_false_alarm"],
        )
    logger.info("\nSaved results.csv, summary.csv, config.json to %s", args.output_dir)


if __name__ == "__main__":
    main()
