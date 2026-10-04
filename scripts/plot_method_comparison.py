"""Privacy vs. dog-preservation trade-off plot for the method comparison.

Usage (from inside speech_anonymization/, after run_method_comparison):
    python -m scripts.plot_method_comparison
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURFACE, REF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"
# fixed categorical order (validated palette); identity also carried by marker + direct label
METHODS = {  # label, color, marker, label offset (pt), label alignment
    "vad_mute": ("VAD + mute", "#2a78d6", "o", (0, 10), "center"),
    "gated_subtract_dtln": ("DTLN", "#eb6834", "s", (-9, -3), "right"),
    "gated_subtract_gtcrn": ("GTCRN", "#1baf7a", "D", (0, -17), "center"),
    "gated_subtract_deepfilternet3": ("DeepFilterNet3", "#eda100", "^", (9, 3), "left"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "method_comparison"
    )
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.input_dir / "summary.csv")))
    snrs = sorted({float(r["snr_db"]) for r in rows}, reverse=True)

    fig, axes = plt.subplots(1, len(snrs), figsize=(4.6 * len(snrs), 4.6), facecolor=SURFACE, sharey=True)
    for ax, snr in zip(axes, snrs):
        ax.set_facecolor(SURFACE)
        sel = {r["method"]: r for r in rows if float(r["snr_db"]) == snr}
        ref = sel["passthrough"]
        ax.scatter(float(ref["mean_speech_leakage_db"]), float(ref["mean_overlap_dog_sdr_db"]), s=70,
                   facecolors="none", edgecolors=REF, linewidths=2, zorder=3)
        ax.annotate("no processing", (float(ref["mean_speech_leakage_db"]), float(ref["mean_overlap_dog_sdr_db"])),
                    xytext=(-8, -4), textcoords="offset points", ha="right", va="top", fontsize=8.5, color=MUTED)
        for key, (label, color, marker, offset, align) in METHODS.items():
            r = sel[key]
            x, y = float(r["mean_speech_leakage_db"]), float(r["mean_overlap_dog_sdr_db"])
            ax.scatter(x, y, s=80, color=color, marker=marker, edgecolors=SURFACE, linewidths=2, zorder=4, label=label)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha=align, va="center", fontsize=8.5, color=INK)
        ax.set_xlim(-62, 6)
        ax.set_title(f"SNR {snr:+.0f} dB", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Speech left in output (dB)  ←  more private", fontsize=9, color=MUTED)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.axhline(0, color=REF, lw=0.8, ls=":")
    axes[0].set_ylabel("Dog kept where it overlaps speech\n(SDR vs. clean dog, dB)  ↑ better", fontsize=9, color=MUTED)
    axes[-1].legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK, title="speech removal (gated)", title_fontsize=8.5)
    fig.suptitle("Remove the speech, keep the dog: best is top-left", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    out = args.input_dir / "privacy_vs_dog.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
