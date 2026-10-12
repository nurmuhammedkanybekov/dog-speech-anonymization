"""Plot the privacy / dog-sound trade-off of the full pipeline on real recordings.

x: share of dog-vocalization time (without a human) that is muted, i.e. lost
y: share of annotated human events where Whisper still reads 2+ words

Usage (after run_pipeline_real):
    python -m scripts.plot_pipeline_real
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# name -> (label, colour, label offset in points, horizontal alignment)
LABELS = {
    "original (no processing)": ("no processing", MUTED, (8, 2), "left"),
    "silero_mute": ("Silero VAD + mute (original approach)", MUTED, (8, -10), "left"),
    "mute_strict": ("trained detector + mute (strict)", GREEN, (0, 9), "center"),
    "mute_balanced": ("trained detector + mute (balanced)", GREEN, (0, 9), "center"),
    "hybrid_strict": ("automatic hybrid (strict)", BLUE, (0, 9), "center"),
    "hybrid_balanced": ("automatic hybrid (balanced)", BLUE, (8, 4), "left"),
    "hybrid_balanced_dog07": ("automatic hybrid (balanced, privacy-first dog threshold)", BLUE, (8, -12), "left"),
    "hybrid_high_recall": ("automatic hybrid (high recall)", BLUE, (8, 4), "left"),
    "oracle hybrid (expert labels)": ("oracle hybrid (expert labels)", ORANGE, (8, -12), "left"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "pipeline_real")
    args = p.parse_args()
    rows = {r["config"]: r for r in csv.DictReader(open(args.input_dir / "summary.csv"))}
    floor = float(rows["background floor (no human)"]["share_2plus_words"]) * 100

    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(0, floor, color=GRID, alpha=0.6, zorder=0)
    ax.annotate("Whisper's own hallucination level (background audio, no human)", (1, floor), xytext=(0, -4),
                textcoords="offset points", va="top", fontsize=7.5, color=MUTED)
    for name, (label, color, offset, ha) in LABELS.items():
        if name not in rows:
            continue
        r = rows[name]
        x = float(r["dog_muted"] or 0) * 100
        y = float(r["share_2plus_words"]) * 100
        ax.scatter([x], [y], s=46, color=color, zorder=3)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8, ha=ha,
                    color=color if color != MUTED else INK)
    ax.set_xlim(-2, 50)
    ax.set_ylim(0, max(30, ax.get_ylim()[1]))
    ax.set_xlabel("Dog-vocalization time muted, i.e. lost (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("Human events where Whisper reads 2+ words (%)", fontsize=9, color=MUTED)
    fig.suptitle("Full pipeline on real recordings, no expert labels used", x=0.01, ha="left", fontsize=12, color=INK)
    ax.set_title("581 annotated human events, 30 sessions; detectors tested on unseen participants", loc="left", fontsize=9, color=MUTED)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    fig.tight_layout()
    out = args.input_dir / "privacy_vs_dog_loss.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
