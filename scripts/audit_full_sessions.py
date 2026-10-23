"""Step 22b: privacy audit of WHOLE recordings, not only the annotated human events.

Every privacy number so far looked only at the 581 windows around annotated
human events (Steps 8-21 cover those). Speech that the annotators did not mark (quiet talk across the
room, a word under panting) was never checked, before or after anonymization.
Here Whisper reads every recording from start to end in 8 s windows: the
original, and the output of each detector and setting (leave-one-participant-out
predictions, as in the pipeline).

Only windows with NO annotated human sound (none within 0.5 s) are read here:
the annotated events are already covered. Windows with 2+ readable words
are the interesting ones. Each is either unannotated
speech (a leak no earlier number could see) or Whisper making up words over
dog sounds.

Usage (from inside speech_anonymization/):
    python -m scripts.audit_full_sessions --worker 0 --n-workers 2      # and --worker 1
    python -m scripts.audit_full_sessions --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scripts.run_asr_real import loud
from scripts.run_pipeline_real import frames_to_segments
from vad_anonymization.anonymize import segments_to_sample_mask, vad_mute
from vad_anonymization.asr import WhisperASR, normalize_text
from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
WIN_S = 8.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"
CONFIGS = ["eat40_1_mute_c06", "eat40_1_mute_c10", "eat40_1_mute_c15"]
MARGIN_S = 0.5   # windows are audited only if no annotated human sound lies within this margin


def load_configs() -> dict:
    cfg = {}
    for f in ("configs_v16.json", "configs_v17.json", "configs_v21.json"):
        cfg.update(json.loads((OUT / f).read_text()))
    return {k: cfg[k] for k in CONFIGS}


def overlap(a: float, b: float, events) -> float:
    return sum(max(0.0, min(b, e.end_s) - max(a, e.start_s)) for e in events)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    out = OUT / "full_session_audit"
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    cfg = load_configs()
    if not args.summarize_only:
        import torch

        torch.set_num_threads(1)
        asr = WhisperASR("small", language="", num_threads=1)
        probs = {}
        for name, (_, th, _, path) in cfg.items():
            arr = np.load(PROJECT_ROOT / "speech_anonymization" / path)["prob"]
            i, probs[name] = 0, {}
            for s in sessions:
                n = len(np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"])
                probs[name][s.name] = arr[i : i + n] >= th
                i += n
        t0 = time.time()
        for k, s in enumerate(sessions):
            f = out / "sessions" / f"{s.name}.json"
            if k % args.n_workers != args.worker or f.exists():
                continue
            x = load_audio(s.audio_path, SR)
            total = len(x) / SR
            centres = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
            versions = {"original": x}
            for name in cfg:
                versions[name] = vad_mute(x, frames_to_segments(centres, probs[name][s.name], total))
            cache: dict = {}
            windows = []
            for a in np.arange(0, total, WIN_S):
                b = min(total, a + WIN_S)
                if b - a < 1.0:
                    continue
                ia, ib = int(a * SR), int(b * SR)
                if overlap(a - MARGIN_S, b + MARGIN_S, s.human) > 0:
                    continue
                row = {"start_s": round(float(a), 2), "end_s": round(float(b), 2),
                       "human_s": 0.0, "dog_s": round(overlap(a, b, s.dog), 2)}
                for name, y in versions.items():
                    if name != "original" and len(normalize_text(row["original"]).split()) < 2:
                        row[name] = None      # nothing readable in the original: nothing to remove
                        continue
                    seg = y[ia:ib]
                    if not torch.any(seg != 0):
                        row[name] = ""
                        continue
                    h = hashlib.md5(seg.numpy().tobytes()).hexdigest()
                    if h not in cache:
                        cache[h] = asr.transcribe(loud(seg))
                    row[name] = cache[h]
                windows.append(row)
            f.write_text(json.dumps({"session": s.name, "windows": windows}, ensure_ascii=False))
            print(f"{s.name}: {len(windows)} windows, {len(cache)} transcribed ({time.time() - t0:.0f}s)", flush=True)

    files = [out / "sessions" / f"{s.name}.json" for s in sessions]
    if not all(f.exists() for f in files):
        print(f"{sum(f.exists() for f in files)}/{len(files)} sessions done - summary skipped")
        return
    rows = []
    for f in files:
        j = json.loads(f.read_text())
        for w in j["windows"]:
            rows.append({"session": j["session"], **w})

    def nw(t: str) -> int:
        return len(normalize_text(t).split())

    summary = []
    for name in ["original"] + CONFIGS:
        for kind, sel in (("all audited windows", lambda r: True),
                          ("windows with dog sound", lambda r: r["dog_s"] > 0)):
            rs = [r for r in rows if sel(r)]
            summary.append({"version": name, "windows": kind, "n": len(rs),
                            "readable_2plus": sum(nw(r[name] or "") >= 2 for r in rs),
                            "readable_5plus": sum(nw(r[name] or "") >= 5 for r in rs),
                            "share_2plus": round(sum(nw(r[name] or "") >= 2 for r in rs) / len(rs), 3)})
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(out / "windows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
