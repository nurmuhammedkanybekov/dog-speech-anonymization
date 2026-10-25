"""Steps 24a / 25e: what to mute in every play session, for an anonymized release of the dataset.

Runs the deployable detector (since Step 26 ``models/human_detector_v3.npz``,
Whisper encoder + EfficientAT, trained on all 17 participants; ``--model
models/human_detector_eat40.npz --release release_v1`` for the Step 24 version)
on all 30 play sessions and writes, per
strictness level, the segments to mute, one CSV per session. It also writes
a per-session report with the share of the session muted, and, from the
expert annotations, how many human events are covered and how many dog calls
of each type stay intact.

The model has seen these participants during training, so the report numbers
are better than what a new recording would get. The leave-one-participant-out
numbers of Steps 17-22 are the honest estimate for new recordings.

The muting itself is applied to the original 48 kHz files by
``apply_mute_segments.py``. That script needs only numpy and ffmpeg, so it
runs on any laptop.

Usage (from inside speech_anonymization/):
    python -m scripts.make_release_segments          # -> outputs/release_v3/{light,standard,strict}/<session>.csv + report.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.anonymize_recording import EAT_DIR, flagged_segments, human_probability
from scripts.extract_efficientat import load_model
from scripts.train_final_detector import MODEL_V3
from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def covered(segs, a: float, b: float) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in segs) / max(1e-9, b - a)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=MODEL_V3)
    p.add_argument("--release", default="release_v3", help="folder name under outputs/")
    args = p.parse_args()
    OUT = OUTPUTS / args.release
    torch.set_num_threads(2)
    det = dict(np.load(args.model))
    levels = [str(x) for x in det["levels"]]
    with contextlib.redirect_stdout(io.StringIO()):
        eat = load_model(EAT_DIR, str(det["efficientat_model"]))
    report = []
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        x = load_audio(s.audio_path, SR)
        total = len(x) / SR
        prob, centres = human_probability(x, det, eat)
        for i, lv in enumerate(levels):
            segs = flagged_segments(prob, centres, float(det["thresholds"][i]), total)
            (OUT / lv).mkdir(parents=True, exist_ok=True)
            with open(OUT / lv / f"{s.name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["start_s", "end_s"])
                w.writerows([[round(a, 3), round(b, 3)] for a, b in segs])
            intact, n_type = Counter(), Counter()
            for e in s.dog:
                if any(min(h.end_s, e.end_s) > max(h.start_s, e.start_s) for h in s.human):
                    continue                       # calls under a human sound are lost by design
                n_type[e.label] += 1
                intact[e.label] += covered(segs, e.start_s, e.end_s) == 0
            report.append({"session": s.name, "level": lv, "duration_s": round(total, 1),
                           "muted_s": round(sum(b - a for a, b in segs), 1),
                           "muted_share": round(sum(b - a for a, b in segs) / total, 3),
                           "human_events": len(s.human),
                           "human_events_mostly_muted": sum(covered(segs, e.start_s, e.end_s) >= 0.5 for e in s.human),
                           "dog_calls_intact": f"{sum(intact.values())}/{sum(n_type.values())}",
                           "intact_by_type": json.dumps({k: f"{intact[k]}/{n_type[k]}" for k in sorted(n_type)})})
        print(s.name, [(r["level"], r["muted_share"]) for r in report[-len(levels):]], flush=True)
    with open(OUT / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    (OUT / "README.txt").write_text(
        f"Mute segments for an anonymized release of the ELTE BARKS play sessions ({args.release}).\n"
        f"Detector: {args.model.name}, {det['trained_on']} (features: {det.get('features', det['efficientat_model'])}).\n"
        "Per level: one CSV per session with the [start_s, end_s] intervals to set to zero in the original file.\n"
        "report.csv: share muted, annotated human events at least half muted, dog calls (without human overlap) left intact.\n"
        f"Apply with: python -m scripts.apply_mute_segments --level standard --release {args.release}\n")


if __name__ == "__main__":
    main()
