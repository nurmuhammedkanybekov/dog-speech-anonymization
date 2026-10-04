"""Step 4: remove the speech, keep the dog - which method does it best?

Compares anonymization methods on synthetic mixtures where both the clean
speech and the clean dog sound are known, so every output can be scored
for privacy (how much speech is left) and utility (how much dog is left).
See ``vad_anonymization/metrics.py`` for what each number means.

Methods:
    passthrough           - do nothing (reference)
    vad_mute              - Silero VAD + mute (the original Task 1 approach)
    subtract_<model>      - x - s_hat, s_hat from a pretrained enhancer
    gated_subtract_<model>- the same, but only inside VAD-detected speech

Usage (from inside speech_anonymization/):
    python -m scripts.download_models          # once
    python -m scripts.run_method_comparison
    python -m scripts.run_method_comparison --speech-dir ../data/librispeech/LibriSpeech/test-clean
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import time
import multiprocessing
from pathlib import Path

import torch

from vad_anonymization.anonymize import gated_subtract, subtract, vad_mute
from vad_anonymization.audio_io import load_audio, save_audio
from vad_anonymization.detection import SpeechDetector
from vad_anonymization.enhancers import ENHANCERS
from vad_anonymization.metrics import dog_retention_db, sdr_db, si_sdr_db, speech_leakage_db, stoi
from vad_anonymization.mixing import build_scene

SAMPLE_RATE = 16_000
CLIP_S = 5.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_EXT = {".wav", ".flac"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speech", type=Path, default=DATA_DIR / "sample_speech.wav", help="single speech file, cut into 5 s chunks")
    p.add_argument("--speech-dir", type=Path, default=None, help="folder of speech files (e.g. LibriSpeech); overrides --speech")
    p.add_argument("--per-speaker", type=int, default=2, help="with --speech-dir: clips per speaker (LibriSpeech layout)")
    p.add_argument("--dog-dirs", type=Path, nargs="+", default=[DATA_DIR / "esc50_dog", DATA_DIR / "audioset_candidates"])
    p.add_argument("--snr", type=float, nargs="+", default=[10, 0, -10])
    p.add_argument("--enhancers", nargs="+", default=list(ENHANCERS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--summarize-only", action="store_true", help="recompute summary.csv from an existing results.csv")
    p.add_argument("--max-dogs", type=int, default=None, help="limit number of dog clips (quick runs)")
    p.add_argument("--n-examples", type=int, default=4, help="mixtures at 0 dB to save as .wav for listening")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "method_comparison")
    return p.parse_args()


def collect_speech_clips(args: argparse.Namespace, detector: SpeechDetector) -> list[tuple[str, torch.Tensor]]:
    """5 s speech clips that are mostly speech (>= 70% VAD coverage)."""
    n = int(CLIP_S * SAMPLE_RATE)
    candidates: list[tuple[str, torch.Tensor]] = []
    if args.speech_dir:
        by_speaker: dict[str, list[Path]] = {}
        for f in sorted(q for q in args.speech_dir.rglob("*") if q.suffix.lower() in AUDIO_EXT):
            speaker = f.relative_to(args.speech_dir).parts[0]
            by_speaker.setdefault(speaker, []).append(f)
        for speaker, files in by_speaker.items():
            taken = 0
            for f in files:
                if taken >= args.per_speaker:
                    break
                wave = load_audio(f, SAMPLE_RATE)
                if wave.shape[-1] >= n:
                    candidates.append((f"{speaker}/{f.stem}", wave[:n]))
                    taken += 1
    else:
        wave = load_audio(args.speech, SAMPLE_RATE)
        for i in range(wave.shape[-1] // n):
            candidates.append((f"{args.speech.stem}@{i * CLIP_S:.0f}s", wave[i * n : (i + 1) * n]))
    keep = []
    for name, clip in candidates:
        covered = sum(seg.duration_s for seg in detector.detect(clip)) / CLIP_S
        if covered >= 0.7:
            keep.append((name, clip))
    return keep


def collect_dog_clips(dirs: list[Path]) -> list[tuple[str, str, torch.Tensor]]:
    n = int(CLIP_S * SAMPLE_RATE)
    clips = []
    for d in dirs:
        for f in sorted(q for q in d.iterdir() if q.suffix.lower() in AUDIO_EXT):
            wave = load_audio(f, SAMPLE_RATE)[:n]
            if wave.shape[-1] < n:
                wave = torch.nn.functional.pad(wave, (0, n - wave.shape[-1]))
            source = {"esc50_dog": "esc50", "audioset_candidates": "audioset", "dog_windows": "elte_lab"}.get(d.name, d.name)
            clips.append((f.name, source, wave))
    return clips


_W: dict = {}


def _init_worker(enhancer_names: list[str]) -> None:
    torch.set_num_threads(1)
    _W["detector"] = SpeechDetector()
    _W["enhancers"] = {name: ENHANCERS[name]() for name in enhancer_names}


def _vad_seconds(wave: torch.Tensor) -> float:
    return round(sum(seg.duration_s for seg in _W["detector"].detect(wave)), 3)


def evaluate_one(job: dict) -> tuple[list[dict], dict | None]:
    detector, enhancers = _W["detector"], _W["enhancers"]
    sc = build_scene(job["speech"], job["dog"], SAMPLE_RATE, snr_db=job["snr"])
    x, s, d = sc.waveform, sc.speech, sc.dog
    segments = detector.detect(x)

    outputs = {"passthrough": x, "vad_mute": vad_mute(x, segments)}
    for name, enh in enhancers.items():
        s_hat = enh.enhance(x)
        outputs[f"subtract_{name}"] = subtract(x, s_hat)
        outputs[f"gated_subtract_{name}"] = gated_subtract(x, s_hat, segments)

    # regions of the scene timeline: dog overlapping speech, and dog alone
    overlap = slice(int(sc.dog_span_s[0] * SAMPLE_RATE), int(sc.speech_span_s[1] * SAMPLE_RATE))
    dog_only = slice(int(sc.speech_span_s[1] * SAMPLE_RATE), int(sc.dog_span_s[1] * SAMPLE_RATE))
    ref = {"vad_speech_s_clean_dog": _vad_seconds(d), "vad_speech_s_clean_speech": _vad_seconds(s)}
    rows = []
    for method, y in outputs.items():
        rows.append(
            {
                "speech_clip": job["speech_name"],
                "dog_clip": job["dog_name"],
                "dog_source": job["dog_source"],
                "snr_db": job["snr"],
                "method": method,
                "speech_leakage_db": round(speech_leakage_db(y, s), 2),
                "stoi": round(stoi(s, y, SAMPLE_RATE), 4),
                "vad_speech_s": _vad_seconds(y),
                "dog_retention_db": round(dog_retention_db(y, d), 2),
                "dog_si_sdr_db": round(si_sdr_db(y, d), 2),
                "dog_sdr_db": round(sdr_db(y, d), 2),
                "overlap_dog_sdr_db": round(sdr_db(y[overlap], d[overlap]), 2),
                "dog_only_sdr_db": round(sdr_db(y[dog_only], d[dog_only]), 2),
                **ref,
            }
        )
    example = None
    if job.get("save_example"):
        example = {"name": job["example_name"], "speech": s, "dog": d, "outputs": outputs}
    return rows, example


METRICS = ["speech_leakage_db", "stoi", "vad_speech_s", "dog_retention_db", "dog_si_sdr_db", "dog_sdr_db",
           "overlap_dog_sdr_db", "dog_only_sdr_db"]


def summarize(rows: list[dict], snrs: list[float]) -> list[dict]:
    """Mean of every metric per (SNR, method), ignoring undefined values.

    ``dog_only_sdr_db`` is undefined (NaN) for dog clips that happen to be
    silent in the dog-only part of the scene, so it is averaged over the
    clips where it exists and that count is reported as ``n_dog_only``.
    """
    methods = list(dict.fromkeys(r["method"] for r in rows))
    summary = []
    for snr in snrs:
        for m in methods:
            sel = [r for r in rows if float(r["snr_db"]) == snr and r["method"] == m]
            entry = {"snr_db": snr, "method": m, "n": len(sel)}
            for key in METRICS:
                vals = [float(r[key]) for r in sel if math.isfinite(float(r[key]))]
                entry[f"mean_{key}"] = round(statistics.mean(vals), 3) if vals else float("nan")
            entry["n_dog_only"] = sum(math.isfinite(float(r["dog_only_sdr_db"])) for r in sel)
            entry["share_vad_speech_free"] = round(
                sum(float(r["vad_speech_s"]) <= float(r["vad_speech_s_clean_dog"]) for r in sel) / len(sel), 3
            )
            summary.append(entry)
    return summary


def print_summary(summary: list[dict]) -> None:
    logger.info("\n%-30s %5s %8s %6s %6s %7s %8s %8s %8s", "method", "SNR", "leak dB", "STOI", "VAD s", "dog dB",
                "dog SDR", "ovl SDR", "only SDR")
    for e in summary:
        logger.info("%-30s %5.0f %8.1f %6.2f %6.2f %7.1f %8.1f %8.1f %8.1f", e["method"], e["snr_db"],
                    e["mean_speech_leakage_db"], e["mean_stoi"], e["mean_vad_speech_s"], e["mean_dog_retention_db"],
                    e["mean_dog_sdr_db"], e["mean_overlap_dog_sdr_db"], e["mean_dog_only_sdr_db"])


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        rows = list(csv.DictReader(open(out_dir / "results.csv")))
        summary = summarize(rows, args.snr)
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print_summary(summary)
        return

    detector = SpeechDetector()
    speech_clips = collect_speech_clips(args, detector)
    dog_clips = collect_dog_clips(args.dog_dirs)[: args.max_dogs]
    logger.info("%d speech clips, %d dog clips, SNR %s dB, enhancers %s", len(speech_clips), len(dog_clips), args.snr, args.enhancers)

    jobs = []
    for i, (dog_name, source, dog) in enumerate(dog_clips):
        speech_name, speech = speech_clips[i % len(speech_clips)]
        for snr in args.snr:
            save = snr == 0 and i < args.n_examples
            jobs.append(
                {
                    "speech": speech, "speech_name": speech_name, "dog": dog, "dog_name": dog_name,
                    "dog_source": source, "snr": snr, "save_example": save,
                    "example_name": f"ex{i + 1}_{Path(dog_name).stem}",
                }
            )

    rows: list[dict] = []
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")  # fork + torch threads can deadlock
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.enhancers,)) as pool:
        for k, (job_rows, example) in enumerate(pool.imap(evaluate_one, jobs), 1):
            rows.extend(job_rows)
            if example:
                ex_dir = out_dir / "examples" / example["name"]
                save_audio(ex_dir / "0_clean_speech.wav", example["speech"], SAMPLE_RATE)
                save_audio(ex_dir / "0_clean_dog.wav", example["dog"], SAMPLE_RATE)
                for method, y in example["outputs"].items():
                    peak = float(y.abs().max())
                    save_audio(ex_dir / f"{method}.wav", y / peak if peak > 1 else y, SAMPLE_RATE)
            if k % 20 == 0 or k == len(jobs):
                logger.info("  %d/%d mixtures (%.0fs)", k, len(jobs), time.time() - t0)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows, args.snr)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    (out_dir / "config.json").write_text(json.dumps({
        "speech_source": str(args.speech_dir or args.speech.name), "n_speech_clips": len(speech_clips),
        "speech_clips": [n for n, _ in speech_clips], "n_dog_clips": len(dog_clips), "clip_s": CLIP_S, "scene": "10 s: speech 0-5 s, dog 2.5-7.5 s, silence 7.5-10 s",
        "snr_db": args.snr, "enhancers": args.enhancers, "methods": methods,
    }, indent=2))

    print_summary(summary)
    logger.info("\nSaved results.csv, summary.csv, config.json and examples/ to %s", out_dir)


if __name__ == "__main__":
    main()
