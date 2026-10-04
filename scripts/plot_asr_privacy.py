"""Plot how many words Whisper can still read back after each method.

Usage (after run_asr_privacy):
    python -m scripts.plot_asr_privacy
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SNR_COLORS = {10.0: "#2a78d6", 0.0: "#eb6834", -10.0: "#1baf7a"}
LABELS = {
    "passthrough": "no processing",
    "vad_mute": "VAD + mute",
    "subtract_dtln": "DTLN subtraction",
    "gated_subtract_dtln": "gated DTLN subtraction",
    "gated_subtract_gtcrn": "gated GTCRN subtraction",
    "gated_subtract_deepfilternet3": "gated DeepFilterNet3 subtraction",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "asr_privacy")
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.input_dir / "summary.csv")))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    y = np.arange(len(methods))[::-1]
    h = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k, (snr, color) in enumerate(SNR_COLORS.items()):
        vals = [100 * float(next(r for r in rows if r["method"] == m and float(r["snr_db"]) == snr)["mean_words_recovered"])
                for m in methods]
        bars = ax.barh(y + (1 - k) * h, vals, height=h, color=color, zorder=3, label=f"speech {snr:+.0f} dB vs dog".replace("+0 ", "0 "))
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}%", (v, b.get_y() + h / 2), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y, [LABELS.get(m, m) for m in methods], fontsize=9, color=INK)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Words of the original sentence Whisper can still read back (%)", fontsize=9, color=MUTED)
    n = rows[0]["n"]
    fig.suptitle("Independent privacy check: can Whisper still read the words?", x=0.01, ha="left", fontsize=12, color=INK)
    ax.set_title(f"Whisper small.en as the attacker, {n} LibriSpeech + dog scenes per level", loc="left", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    out = args.input_dir / "words_recovered.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
