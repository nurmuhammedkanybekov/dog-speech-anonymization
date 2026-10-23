"""Step 25 figure: privacy vs dog sound lost, as judged by three different speech recognizers.

One panel per attacker (Whisper small, Whisper large-v3-turbo, NVIDIA Parakeet
TDT 0.6B v3). Each line is one detector at several thresholds. The y axis is
the share of the words each attacker read in the ORIGINAL events that it can
still read after muting. Unlike "2+ words", this mostly ignores invented text
(turbo writes text on 24.5% of background windows): on background audio the
share of an event's words that turn up is 0.0% (Whisper small, Parakeet) and
1.6% (turbo).

Reads ``outputs/detectors_pipeline/comparison.csv`` (Whisper small, from
``compare_detectors_pipeline``) and ``outputs/attack_<model>/summary.csv``
(``attack_stronger_asr``). Writes ``outputs/detectors_pipeline/step25_three_attackers.png``.

Usage (from inside speech_anonymization/):
    python -m scripts.plot_step25
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.compare_detectors_pipeline import baseline_rates  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [("vgg1", "VGGish (Step 9)", "#8a8985", ["c03", "c06", "c10", "c15"]),
          ("eat40_1", "EfficientAT mn40 (Steps 17-24 default)", "#2a78d6", ["c03", "c06", "c10", "c15"]),
          ("wenc_eat_dog", "Whisper encoder + mn40, human vs dog (Step 25)", "#d23c3c", ["c03", "c06", "c10", "c15"])]
ATTACKERS = [("small", "Whisper small (244M)"), ("turbo", "Whisper large-v3-turbo (809M)"),
             ("parakeet", "Parakeet TDT 0.6B v3 (not Whisper)")]


def main() -> None:
    comp = {(r["detector"], r["dog_frames_flagged"]): r for r in csv.DictReader(open(OUT / "detectors_pipeline" / "comparison.csv"))}
    dog = {f"{d}_mute_{c}": float(r["dog_muted"]) for (d, c), r in comp.items()}
    readable = {"small": {f"{d}_mute_{c}": float(r["words_recovered"]) for (d, c), r in comp.items()}}
    original_small, floor_small = baseline_rates()      # Whisper small, recounted (Step 19)
    floors, originals = {"small": floor_small}, {"small": original_small}
    for model, _ in ATTACKERS[1:]:
        f = OUT / f"attack_{model}" / "summary.csv"
        if not f.exists():
            continue
        readable[model] = {}
        for r in csv.DictReader(open(f)):
            v = r.get(f"{model}_words_recovered", r.get("turbo_words_recovered"))
            if r["version"].startswith("background"):
                floors[model] = float(v) if v else None
            elif r["version"] == "original":
                originals[model] = float(r.get(f"{model}_readable_2plus", r.get("turbo_readable_2plus")))
            elif not r["version"].startswith("McNemar"):
                readable[model][r["version"]] = float(v)

    panels = [a for a in ATTACKERS if a[0] in readable]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.4), facecolor=SURFACE, sharex=True)
    for ax, (model, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ymax = 37
        for det, label, color, costs in SERIES:
            pts = [(dog[n] * 100, readable[model][n] * 100) for c in costs
                   if (n := f"{det}_mute_{c}") in readable[model] and n in dog]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, "-o", color=color, lw=1.8, ms=5, label=label, zorder=3)
        ax.set_xlim(0, 22)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{title}\nreads 2+ words in {originals[model] * 100:.0f}% of the original events", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Dog-only vocalization time muted (%)", fontsize=8.5, color=MUTED)
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[0].set_ylabel("Original words still read back (%)", fontsize=8.5, color=MUTED)
    axes[-1].legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK)
    fig.suptitle("Step 25: for all three attackers, fewer words are left at the same dog loss",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, 0.885, "581 annotated human events, 17 participants, each tested on models that never saw them; "
             "points = thresholds flagging 3 / 6 / 10 / 15% of dog-only frames", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path = OUT / "detectors_pipeline" / "step25_three_attackers.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
