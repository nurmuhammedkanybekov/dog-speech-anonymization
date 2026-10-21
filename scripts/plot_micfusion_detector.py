"""Plot the detector-level gain of Step 15: human-time recall vs dog-only frames flagged.

Usage (after train_micfusion_detector):
    python -m scripts.plot_micfusion_detector
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "outputs" / "human_detector_micfusion"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [("baseline", "one microphone (Step 9)", "#52514e", "--"),
          ("baseline_smooth", "one microphone + smoothing", "#1baf7a", "-"),
          ("pooled_max_smooth", "three microphones + smoothing (Step 15)", "#eda100", "-")]


def main() -> None:
    z = np.load(OUT / "all_variants.npz")
    human, dog = z["human"], z["dog"]
    pos, cost = human >= 0.5, (human == 0) & (dog >= 0.5)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = np.linspace(0, 0.5, 101)
    for key, label, color, ls in SERIES:
        p = z[key]
        th = np.quantile(p[cost], 1 - xs)
        rec = [(p[pos] >= t).mean() for t in th]
        ax.plot(xs * 100, np.array(rec) * 100, ls, color=color, lw=1.8, label=label)
    for x in (6, 24.9):
        ax.axvline(x, color=GRID, lw=1.2, zorder=0)
    ax.text(6.4, 95, "strict", fontsize=7.5, color=MUTED)
    ax.text(25.3, 95, "balanced", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Dog-only frames flagged as human (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("Annotated human time found (%)", fontsize=9, color=MUTED)
    fig.suptitle("Human detector: one vs three microphones", x=0.01, ha="left", fontsize=12, color=INK)
    ax.set_title("31,066 frames of 0.24 s, 17 participants, each tested on a model that never saw them", loc="left",
                 fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    fig.tight_layout()
    fig.savefig(OUT / "recall_vs_dog_cost.png", dpi=200, facecolor=SURFACE)
    print(f"saved {OUT / 'recall_vs_dog_cost.png'}")


if __name__ == "__main__":
    main()
