"""Plot the trained detector against Silero VAD on the real recordings.

x: share of dog-call time that gets flagged (what removal would touch but should not)
y: share of annotated human time that gets flagged (what removal needs to catch)

Usage (after train_human_detector and eval_real_detection):
    python -m scripts.plot_human_detector
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detector-dir", type=Path, default=OUT / "human_detector")
    p.add_argument("--silero-dir", type=Path, default=OUT / "real_detection")
    args = p.parse_args()

    z = np.load(args.detector_dir / "oof_predictions.npz")
    prob, human, other, dog = z["prob"], z["human"], z["other"], z["dog"]
    pos = human >= 0.5
    dog_only = (human == 0) & (other == 0) & (dog >= 0.5)
    ths = np.quantile(prob, np.linspace(0, 1, 400))
    x = np.array([(prob[dog_only] >= t).mean() for t in ths]) * 100
    y = np.array([(prob[pos] >= t).mean() for t in ths]) * 100
    info = json.loads((args.detector_dir / "summary.json").read_text())

    silero = list(csv.DictReader(open(args.silero_dir / "summary.csv")))
    sx = [100 * float(r["dog_only_false_alarm"]) for r in silero]
    sy = [100 * float(r["recall"]) for r in silero]

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(x, y, color=BLUE, lw=2.2, zorder=3)
    ax.plot([0, 100], [0, 100], color=GRID, lw=1, ls="--", zorder=1)
    ax.scatter(sx, sy, color=ORANGE, s=36, zorder=4)
    ax.annotate("Silero VAD (thresholds 0.3 / 0.5 / 0.7)", (max(sx), max(sy)), xytext=(10, 6),
                textcoords="offset points", color=ORANGE, fontsize=9)
    ax.annotate(f"trained detector (VGGish + logistic regression)\nleave-one-participant-out, AUC {info['roc_auc']:.2f}",
                (x[np.argmin(abs(y - 70))], 70), xytext=(14, -34), textcoords="offset points", color=BLUE, fontsize=9)
    for target in (50, 80, 90):
        i = np.argmin(abs(y - target))
        ax.scatter([x[i]], [y[i]], color=BLUE, s=22, zorder=4)
        ax.annotate(f"{y[i]:.0f}% caught / {x[i]:.0f}% of dog time touched", (x[i], y[i]), xytext=(8, -12),
                    textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Dog-call time flagged as human (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("Annotated human time caught (%)", fontsize=9, color=MUTED)
    ax.set_title("Finding the human sounds in the real ELTE recordings", loc="left", fontsize=12, color=INK)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    fig.tight_layout()
    out = args.detector_dir / "detector_vs_silero.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
