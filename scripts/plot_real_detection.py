"""Plot which real human vocalizations Silero VAD detects (by loudness and length).

Usage (after analyze_real_human_events):
    python -m scripts.plot_real_detection
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURFACE, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"


def binned(rows, key, edges, labels):
    out = []
    for (lo, hi), label in zip(edges, labels):
        sel = [r for r in rows if lo <= float(r[key]) < hi]
        hit = sum(float(r["max_speech_prob"]) >= 0.5 for r in sel)
        out.append((label, len(sel), 100 * hit / len(sel) if sel else 0.0))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "real_detection")
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.input_dir / "human_events.csv")))

    panels = [
        ("How loud the human sound is", binned(rows, "level_dbfs", [(-200, -60), (-60, -50), (-50, -40), (-40, 0)],
                                             ["< -60 dB", "-60 to -50", "-50 to -40", "> -40 dB"])),
        ("How long the human sound is", binned(rows, "duration_s", [(0, 0.5), (0.5, 1), (1, 2), (2, 1e9)],
                                              ["< 0.5 s", "0.5-1 s", "1-2 s", "> 2 s"])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE, sharey=True)
    for ax, (title, data) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        labels = [d[0] for d in data]
        vals = [d[2] for d in data]
        bars = ax.bar(labels, vals, color=BLUE, width=0.6, zorder=3)
        for bar, (_, n, v) in zip(bars, data):
            ax.annotate(f"{v:.0f}%", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
            ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, -26),
                        textcoords="offset points", ha="center", fontsize=8, color=MUTED, annotation_clip=False)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylim(0, 50)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Detected by Silero VAD (%)", fontsize=9, color=MUTED)
    fig.suptitle(f"Real ELTE play sessions: {len(rows)} annotated human vocalizations", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    out = args.input_dir / "human_detection_by_level_and_length.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
