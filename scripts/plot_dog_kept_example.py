"""Steps 27-28 figure: one real dog call under speech - original, muted, dog kept, dog kept + room tone.

Reads a clip of the listening pack (``make_listening_pack``) and draws the four
versions with the annotated human sound and dog call marked. Writes
``outputs/dog_extraction/example_clip.png`` (an image, no audio).

Usage (from inside speech_anonymization/):
    python -m scripts.plot_dog_kept_example --clip 3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from vad_anonymization.elte_barks import load_sessions  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INK, MUTED = "#0b0b0b", "#52514e"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", type=int, default=3)
    args = p.parse_args()
    d = OUT / "listening_overlap" / f"clip_{args.clip:02d}"
    row = [r for r in csv.DictReader(open(OUT / "listening_overlap" / "answers.csv")) if r["clip"] == d.name][0]
    sess = {s.name: s for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks")}[row["session"]]
    t0, t1 = float(row["start_s"]), float(row["end_s"])
    titles = {"a_original": "original (person + dog)", "b_muted": "standard: muted", "c_dog_kept": "--keep-dog more: dog kept",
              "d_all_room_tone": "--keep-dog all: extractor output, room tone where no dog is heard"}
    titles = {k: v for k, v in titles.items() if (d / f"{k}.wav").exists()}
    fig, axes = plt.subplots(len(titles), 1, figsize=(8, 2.1 * len(titles) + 0.1), sharex=True, facecolor="#fcfcfb")
    ref = None
    for ax, (name, title) in zip(axes, titles.items()):
        x, sr = sf.read(d / f"{name}.wav")
        Pxx, f, t, im = ax.specgram(x + 1e-9, NFFT=512, Fs=sr, noverlap=384, cmap="magma", vmin=-140, vmax=-40,
                                    xextent=(t0, t0 + len(x) / sr))
        ax.set_ylim(0, 8000)
        ax.set_ylabel("Hz", fontsize=8, color=MUTED)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        for e in sess.human:
            if e.end_s > t0 and e.start_s < t1:
                ax.axvspan(max(t0, e.start_s), min(t1, e.end_s), ymin=0.97, ymax=1.0, color="#2a78d6")
        for e in sess.dog:
            if e.end_s > t0 and e.start_s < t1:
                ax.axvspan(max(t0, e.start_s), min(t1, e.end_s), ymin=0.93, ymax=0.96, color="#d23c3c")
        ax.tick_params(labelsize=8, colors=MUTED)
    axes[-1].set_xlabel("time in the session (s)   blue bar: annotated human sound, red bar: annotated dog call",
                        fontsize=8, color=MUTED)
    fig.suptitle(f"A dog call under speech ({row['session']}, {row['dog_call'].lower()}), "
                 "models that never heard this dog or person", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / "dog_extraction" / "example_clip.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="#fcfcfb")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
