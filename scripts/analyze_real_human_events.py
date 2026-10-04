"""Step 5b: why does the speech detector miss most real human vocalizations?

For every expert-annotated human vocalization in the ELTE play sessions,
this records its length, its loudness (dBFS) and the highest Silero speech
probability inside it. If loud events are detected but quiet ones are not,
the problem is recording level; if even loud events score low, the sounds
are mostly not speech-like (laughs, calls, kissing/clicking sounds...).

Usage (from inside speech_anonymization/):
    python -m scripts.analyze_real_human_events
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from silero_vad import load_silero_vad

from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

SAMPLE_RATE = 16_000
CHUNK = 512  # Silero's frame size at 16 kHz (32 ms)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def frame_probabilities(model, audio: torch.Tensor) -> torch.Tensor:
    model.reset_states()
    probs = [model(audio[i : i + CHUNK], SAMPLE_RATE).item() for i in range(0, audio.shape[-1] - CHUNK, CHUNK)]
    return torch.tensor(probs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "real_detection")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_silero_vad()
    rows = []
    for sess in load_sessions(args.data_root):
        audio = load_audio(sess.audio_path, SAMPLE_RATE)
        probs = frame_probabilities(model, audio)
        frame_t = torch.arange(len(probs)) * CHUNK / SAMPLE_RATE
        for e in sess.human:
            seg = audio[int(e.start_s * SAMPLE_RATE) : int(e.end_s * SAMPLE_RATE)]
            inside = probs[(frame_t >= e.start_s) & (frame_t < e.end_s)]
            if seg.numel() == 0 or inside.numel() == 0:
                continue
            level = 10 * math.log10(float(seg.double().pow(2).mean()) + 1e-20)
            rows.append({
                "session": sess.name, "start_s": round(e.start_s, 3), "duration_s": round(e.duration_s, 3),
                "level_dbfs": round(level, 1), "max_speech_prob": round(float(inside.max()), 3),
                "mean_speech_prob": round(float(inside.mean()), 3),
            })
        print(f"{sess.name}: {len(sess.human)} human events")
    with open(args.output_dir / "human_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def share(sel):
        return sum(r["max_speech_prob"] >= 0.5 for r in sel) / len(sel) if sel else float("nan")

    print(f"\n{len(rows)} human events")
    for lo, hi in [(-120, -60), (-60, -50), (-50, -40), (-40, 0)]:
        sel = [r for r in rows if lo <= r["level_dbfs"] < hi]
        print(f"  level {lo:>4} to {hi:>4} dBFS: {len(sel):4d} events, {100 * share(sel):5.1f}% reach speech prob >= 0.5")
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 100)]:
        sel = [r for r in rows if lo <= r["duration_s"] < hi]
        print(f"  length {lo:>3}-{hi:<3} s: {len(sel):4d} events, {100 * share(sel):5.1f}% reach speech prob >= 0.5")


if __name__ == "__main__":
    main()
