"""Step 6: if detection were perfect, does subtraction remove REAL human sounds?

"Oracle" experiment: the expert annotations stand in for a perfect detector.
Inside every annotated human event (+0.1 s), the speech estimate of a
pretrained enhancer is subtracted (gated subtraction); everything else in
the recording is left untouched. This separates the two problems:
detection (known to be weak on these recordings) and removal.

Real recordings have no clean ground truth, so only measurable things are
reported, per human event:

* ``change_db``         - energy after / before inside the event.
* ``above_floor_db``    - energy left in the event relative to the local
                          background noise floor (10th percentile of 50 ms
                          frame energies in +-5 s around the event). Near 0 dB
                          means the human sound was taken down to the room
                          noise; the larger, the more is left.
* ``dog_overlap``       - whether a dog sound overlaps the event (then the
                          remaining energy is partly dog, which is wanted).

Because the recordings are very quiet (median human event ~ -60 dBFS),
each method is also run with the event's surroundings normalized to a
normal speech level before enhancement ("_norm"), and scaled back after.

Usage (from inside speech_anonymization/):
    python -m scripts.run_oracle_real
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

from vad_anonymization.anonymize import gated_subtract
from vad_anonymization.audio_io import load_audio, save_audio
from vad_anonymization.detection import SpeechSegment
from vad_anonymization.elte_barks import load_sessions
from vad_anonymization.enhancers import ENHANCERS

SAMPLE_RATE = 16_000
PAD_S = 0.1
CONTEXT_S = 1.0
TARGET_RMS_DB = -25.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def energy_db(x: torch.Tensor) -> float:
    return 10 * math.log10(float(x.double().pow(2).mean()) + 1e-20)


def noise_floor_db(audio: torch.Tensor, t0: float, t1: float) -> float:
    a = max(0, int((t0 - 5) * SAMPLE_RATE))
    b = min(audio.shape[-1], int((t1 + 5) * SAMPLE_RATE))
    region = audio[a:b]
    hop = int(0.05 * SAMPLE_RATE)
    frames = region[: (region.shape[-1] // hop) * hop].reshape(-1, hop)
    energies = frames.double().pow(2).mean(dim=1)
    return 10 * math.log10(float(torch.quantile(energies, 0.10)) + 1e-20)


def speech_estimate(audio: torch.Tensor, segments: list[SpeechSegment], enhancer, normalize: bool) -> torch.Tensor:
    """Run the enhancer only around the segments (with context); zeros elsewhere."""
    est = torch.zeros_like(audio)
    for seg in segments:
        a = max(0, int((seg.start_s - CONTEXT_S) * SAMPLE_RATE))
        b = min(audio.shape[-1], int((seg.end_s + CONTEXT_S) * SAMPLE_RATE))
        chunk = audio[a:b]
        gain = 1.0
        if normalize:
            rms = float(chunk.double().pow(2).mean().sqrt())
            gain = min(1000.0, 10 ** (TARGET_RMS_DB / 20) / rms) if rms > 0 else 1.0
            peak = float(chunk.abs().max()) * gain
            if peak > 0.99:
                gain *= 0.99 / peak
        s_hat = enhancer.enhance(chunk * gain) / gain
        s0, s1 = int(seg.start_s * SAMPLE_RATE) - a, int(seg.end_s * SAMPLE_RATE) - a
        est[a + s0 : a + s1] = s_hat[s0:s1]
    return est


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--enhancers", nargs="+", default=["dtln", "gtcrn"])
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "oracle_real")
    args = p.parse_args()
    (args.output_dir / "examples").mkdir(parents=True, exist_ok=True)
    enhancers = {name: ENHANCERS[name]() for name in args.enhancers}

    rows, examples_saved = [], 0
    for sess in load_sessions(args.data_root):
        audio = load_audio(sess.audio_path, SAMPLE_RATE)
        total = audio.shape[-1] / SAMPLE_RATE
        segs = [SpeechSegment(max(0.0, e.start_s - PAD_S), min(total, e.end_s + PAD_S)) for e in sess.human]
        outputs = {}
        for name, enh in enhancers.items():
            for norm in (False, True):
                est = speech_estimate(audio, segs, enh, norm)
                outputs[f"{name}{'_norm' if norm else ''}"] = gated_subtract(audio, est, segs)
        for k, e in enumerate(sess.human):
            a, b = int(e.start_s * SAMPLE_RATE), int(e.end_s * SAMPLE_RATE)
            if b - a < int(0.05 * SAMPLE_RATE):
                continue
            before = energy_db(audio[a:b])
            floor = noise_floor_db(audio, e.start_s, e.end_s)
            overlap = any(min(d.end_s, e.end_s) - max(d.start_s, e.start_s) > 0 for d in sess.dog)
            row = {"id": f"{sess.name}_{k:03d}", "session": sess.name, "start_s": round(e.start_s, 3),
                   "duration_s": round(e.duration_s, 3), "dog_overlap": overlap,
                   "level_dbfs": round(before, 1), "above_floor_db_before": round(before - floor, 1)}
            for m, y in outputs.items():
                after = energy_db(y[a:b])
                row[f"{m}_change_db"] = round(after - before, 1)
                row[f"{m}_above_floor_db"] = round(after - floor, 1)
            rows.append(row)
            if examples_saved < args.n_examples and not overlap and e.duration_s > 1.0 and before - floor > 10:
                t0, t1 = max(0, a - SAMPLE_RATE), min(audio.shape[-1], b + SAMPLE_RATE)
                g = min(1000.0, 10 ** (-20 / 20) / (float(audio[a:b].double().pow(2).mean().sqrt()) + 1e-12))
                ex = args.output_dir / "examples" / row["id"]
                save_audio(ex / "0_original.wav", torch.tanh(audio[t0:t1] * g), SAMPLE_RATE)
                for m, y in outputs.items():
                    save_audio(ex / f"{m}.wav", torch.tanh(y[t0:t1] * g), SAMPLE_RATE)
                examples_saved += 1
        print(f"{sess.name}: {len(sess.human)} human events")

    with open(args.output_dir / "per_event.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    methods = [k[: -len("_change_db")] for k in rows[0] if k.endswith("_change_db")]
    summary = []
    for group, sel in [("human only", [r for r in rows if not r["dog_overlap"]]),
                       ("human + dog overlap", [r for r in rows if r["dog_overlap"]])]:
        audible = [r for r in sel if r["above_floor_db_before"] >= 6]
        for m in methods:
            summary.append({
                "group": group, "method": m, "n_events": len(sel), "n_audible": len(audible),
                "median_change_db": round(statistics.median(r[f"{m}_change_db"] for r in sel), 1),
                "median_above_floor_before_db": round(statistics.median(r["above_floor_db_before"] for r in audible), 1),
                "median_above_floor_after_db": round(statistics.median(r[f"{m}_above_floor_db"] for r in audible), 1),
                "share_down_to_floor_3db": round(sum(r[f"{m}_above_floor_db"] <= 3 for r in audible) / len(audible), 3),
            })
    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\n{'group':22s} {'method':12s} {'events':>6s} {'audible':>7s} {'change':>7s} {'floor+ before':>13s} {'after':>6s} {'to floor':>8s}")
    for s in summary:
        print(f"{s['group']:22s} {s['method']:12s} {s['n_events']:6d} {s['n_audible']:7d} {s['median_change_db']:6.1f}dB "
              f"{s['median_above_floor_before_db']:11.1f}dB {s['median_above_floor_after_db']:5.1f}dB {100 * s['share_down_to_floor_3db']:7.0f}%")


if __name__ == "__main__":
    main()
