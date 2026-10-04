"""Plot the SNR sweep results (needs matplotlib, see requirements-dev.txt).

Usage (from inside speech_anonymization/, after run_snr_sweep):
    python -m scripts.plot_snr_sweep
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {  # fixed categorical order: slot 1 blue, slot 2 orange
    "esc50": ("ESC-50 dog clips", "#2a78d6", "o"),
    "audioset": ("AudioSet dog clips", "#eb6834", "s"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "snr_sweep")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.input_dir / "summary.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    panels = [
        ("mean_recall", "Speech still detected while the dog is barking", "Share of the dog window flagged as speech"),
        ("share_with_false_alarm", "Bark alone mistaken for speech", "Share of clips with a false alarm"),
    ]
    for ax, (key, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for source, (label, color, marker) in SERIES.items():
            sel = sorted((r for r in rows if r["source"] == source), key=lambda r: -float(r["snr_db"]))
            if not sel:
                continue
            x = [float(r["snr_db"]) for r in sel]
            y = [100 * float(r[key]) for r in sel]
            n = sel[0]["n_clips"]
            ax.plot(x, y, color=color, lw=2, marker=marker, ms=7, mec=SURFACE, mew=2, label=f"{label} (n={n})")
            ax.annotate(f"{y[-1]:.0f}%", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.invert_xaxis()  # left = quiet dog, right = loud dog
        ax.set_ylim(-3, 105)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("SNR (dB)  →  louder dog", fontsize=9, color=MUTED)
        ax.set_ylabel(ylabel + " (%)", fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    out = args.input_dir / "snr_sweep.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
