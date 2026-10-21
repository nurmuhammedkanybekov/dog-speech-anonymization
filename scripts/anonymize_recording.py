"""Step 18b: anonymize a new recording - mute the human sounds, keep everything else.

This is the recommended pipeline of Steps 16-17 as one command, for recordings
the models have never seen. It needs no annotations:

    EfficientAT mn40 (frozen) -> trained logistic regression -> 5-frame smoothing
    -> threshold for the chosen strictness -> mute the flagged time (+-0.12 s)

The muting is applied to the ORIGINAL file (its own sample rate and all its
channels), so the dog sounds that are kept are untouched. Next to the output
it writes ``<output>.segments.csv`` (what was muted, in seconds).

Strictness levels (thresholds measured on unseen participants, README Step 17):
``light``, ``standard`` (default) and ``strict``; stricter = fewer readable words, more dog time lost.

Models (``--model``, README Step 21): ``models/human_detector_eat40.npz`` (default, for
ELTE-like recordings), ``models/human_detector_eat40_aug2.npz`` (unknown recordings,
privacy first) and ``models/human_detector_eat40_aug.npz`` (unknown recordings, keeps
more dog sound but can leak speech under barks).

Setup (once): ``git clone https://github.com/fschmid56/EfficientAT models/EfficientAT``
and ``python -m scripts.train_final_detector`` (or copy ``models/human_detector_eat40.npz``).

Usage (from inside speech_anonymization/):
    python -m scripts.anonymize_recording input.wav anonymized.wav
    python -m scripts.anonymize_recording input.wav anonymized.wav --strictness strict
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import uniform_filter1d

from scripts.extract_efficientat import load_model, embed
from scripts.train_final_detector import MODEL_PATH

SR = 16_000
WIN_S, HOP_S, PAD_S = 0.96, 0.24, 0.12
EAT_DIR = Path(__file__).resolve().parent.parent / "models" / "EfficientAT"


def human_probability(mono16: torch.Tensor, detector, eat) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame probability of a human sound and the frame centres (s), on the Step 9 grid."""
    total = len(mono16) / SR
    n = max(1, int((total - WIN_S) // HOP_S) + 1)
    _, logits = embed(mono16, *eat[:2], n)
    z = (logits - detector["mean"]) / detector["scale"] @ detector["coef"] + detector["intercept"]
    prob = uniform_filter1d(1 / (1 + np.exp(-z)), int(detector["smooth"]), mode="nearest")
    return prob, WIN_S / 2 + HOP_S * np.arange(n)


def flagged_segments(prob: np.ndarray, centres: np.ndarray, threshold: float, total_s: float) -> list[tuple[float, float]]:
    segs: list[list[float]] = []
    for c in centres[prob >= threshold]:
        a, b = max(0.0, c - HOP_S / 2 - PAD_S), min(total_s, c + HOP_S / 2 + PAD_S)
        if segs and a <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    return [(a, b) for a, b in segs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--strictness", default="standard", choices=["light", "standard", "strict"])
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    args = p.parse_args()
    torch.set_num_threads(2)
    det = dict(np.load(args.model))
    levels = [str(x) for x in det["levels"]]
    threshold = float(det["thresholds"][levels.index(args.strictness)])

    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)       # (samples, channels), original rate
    import torchaudio

    mono = torch.from_numpy(audio.mean(1))
    mono16 = torchaudio.functional.resample(mono, sr, SR) if sr != SR else mono
    eat = load_model(EAT_DIR, str(det["efficientat_model"]))
    prob, centres = human_probability(mono16, det, eat)
    total = audio.shape[0] / sr
    segs = flagged_segments(prob, centres, threshold, total)
    out = audio.copy()
    for a, b in segs:
        out[int(a * sr) : int(np.ceil(b * sr))] = 0.0
    info = sf.info(args.input)
    sf.write(args.output, out, sr, subtype=info.subtype if info.subtype in ("PCM_16", "PCM_24", "PCM_32", "FLOAT") else "PCM_16")
    with open(args.output.with_suffix(args.output.suffix + ".segments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s"])
        w.writerows([[round(a, 3), round(b, 3)] for a, b in segs])
    muted = sum(b - a for a, b in segs)
    print(f"{args.input.name}: {len(segs)} segments muted, {muted:.1f} s of {total:.1f} s ({100 * muted / total:.1f}%), "
          f"strictness '{args.strictness}' -> {args.output}")


if __name__ == "__main__":
    main()
