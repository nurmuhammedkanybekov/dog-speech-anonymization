"""Plot the privacy / dog-sound trade-off of the full pipeline on real recordings.

x: share of dog-vocalization time (without a human) that is muted, i.e. lost
y: share of annotated human events where Whisper still reads 2+ words

Usage (after run_pipeline_real):
    python -m scripts.plot_pipeline_real
    python -m scripts.plot_pipeline_real --improved-dir outputs/pipeline_real_gtcrn_ft --oracle-override 0.036
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
# config -> (short label, colour, label offset in data units (dx, dy), horizontal alignment)
LABELS = {
    "original (no processing)": ("no processing", MUTED, (1.0, 0.3), "left"),
    "silero_mute": ("Silero VAD + mute (original approach)", MUTED, (1.0, -1.1), "left"),
    "mute_strict": ("detector + mute, strict", GREEN, (0.5, -1.2), "left"),
    "mute_balanced": ("detector + mute, balanced", GREEN, (0.5, 1.0), "left"),
    "hybrid_strict": ("hybrid, strict", BLUE, (-2.0, 0.9), "left"),
    "hybrid_balanced": ("hybrid, balanced", BLUE, (0.7, 0.4), "left"),
    "hybrid_balanced_dog07": ("hybrid, privacy-first", BLUE, (-0.7, 0.0), "right"),
    "hybrid_high_recall": ("hybrid, high recall", BLUE, (0.7, 0.4), "left"),
    "oracle hybrid (expert labels)": ("oracle hybrid (DTLN)", ORANGE, (1.0, 0.9), "left"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "pipeline_real")
    p.add_argument("--improved-dir", type=Path, default=None,
                   help="second run with a better remover (e.g. outputs/pipeline_real_gtcrn_ft): its hybrids are drawn too")
    p.add_argument("--oracle-override", type=float, default=None, help="oracle hybrid share with the better remover (e.g. 0.036)")
    args = p.parse_args()
    rows = {r["config"]: r for r in csv.DictReader(open(args.input_dir / "summary.csv"))}
    floor = float(rows["background floor (no human)"]["share_2plus_words"]) * 100

    fig, ax = plt.subplots(figsize=(8.8, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(0, floor, color=GRID, alpha=0.6, zorder=0)
    ax.text(16, 1.0, "shaded: Whisper's own hallucination level (background audio, no human)", va="center", ha="left",
            fontsize=7.5, color=MUTED)

    def point(name, r, color, label=True):
        x = float(r["dog_muted"] or 0) * 100
        y = float(r["share_2plus_words"]) * 100
        ax.scatter([x], [y], s=46, color=color, zorder=3)
        if label and name in LABELS:
            text, c, (dx, dy), ha = LABELS[name]
            ax.text(x + dx, y + dy, text, fontsize=8, ha=ha, va="center", color=c if c != MUTED else INK)
        return x, y

    for name in LABELS:
        if name in rows:
            point(name, rows[name], LABELS[name][1])
    if args.improved_dir:
        better = {r["config"]: r for r in csv.DictReader(open(args.improved_dir / "summary.csv"))}
        for name in ("hybrid_strict", "hybrid_balanced", "hybrid_balanced_dog07", "hybrid_high_recall"):
            if name in better and name in rows:
                x0, y0 = float(rows[name]["dog_muted"]) * 100, float(rows[name]["share_2plus_words"]) * 100
                x1, y1 = point(name, better[name], GOLD, label=False)
                ax.annotate("", (x1, y1 + 0.25), (x0, y0 - 0.25), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2), zorder=2)
    if args.oracle_override is not None:
        y = args.oracle_override * 100
        ax.scatter([0], [y], s=46, color=GOLD, zorder=3)
        ax.text(1.0, y - 0.9, "oracle hybrid (GTCRN)", fontsize=8, va="center", color=GOLD)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=MUTED, label="baselines"),
        plt.Line2D([], [], marker="o", ls="", color=GREEN, label="trained detector + mute"),
        plt.Line2D([], [], marker="o", ls="", color=BLUE, label="automatic hybrid, DTLN remover"),
    ]
    if args.improved_dir:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=GOLD, label="same with fine-tuned GTCRN remover (arrow)"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="expert labels instead of detectors"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    ax.set_xlim(-2, 50)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Dog-vocalization time muted, i.e. lost (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("Human events where Whisper reads 2+ words (%)", fontsize=9, color=MUTED)
    fig.suptitle("Full pipeline on real recordings, no expert labels used", x=0.01, ha="left", fontsize=12, color=INK)
    ax.set_title("581 annotated human events, 30 sessions; every model tested on unseen participants", loc="left",
                 fontsize=9, color=MUTED)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    fig.tight_layout()
    out = (args.improved_dir or args.input_dir) / "privacy_vs_dog_loss.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
