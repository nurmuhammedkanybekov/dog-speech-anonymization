"""Step 28: how much the dog extractor removes on the REAL recordings (no speech recognizer needed).

On real recordings the true dog signal under speech is unknown, but two kinds of
annotated events show the extractor's behaviour directly:

* human-only events (no dog call overlapping): every bit of output is leaked human
  sound plus background, so the output/input energy ratio is an upper bound on the
  speech left;
* dog-only events (no human sound overlapping): the output/input ratio is how much
  of the dog is kept.

Each session is processed by the extractor trained on the other participant group,
over the whole recording (mask power 3, as deployed). Reported per participant
group and overall: energy-weighted ratios in dB, and the share of human-only events
whose output is at least 20 dB below the input ("practically silent").

Writes ``outputs/dog_extraction/real_events<tag>.json``.

Usage (from inside speech_anonymization/):
    python -m scripts.eval_extractor_real --tag ""        # Step 27 extractors
    python -m scripts.eval_extractor_real --tag _tts      # Step 28 extractors
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.eval_dog_extraction import load_extractor, run_extractor
from scripts.finetune_gtcrn import participant_group
from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs" / "dog_extraction"


def overlaps(e, others) -> bool:
    return any(min(o.end_s, e.end_s) > max(o.start_s, e.start_s) for o in others)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="")
    p.add_argument("--mask-power", type=float, default=3.0)
    args = p.parse_args()
    torch.set_num_threads(2)
    models: dict = {}
    rows = []
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        g = participant_group(s.name)
        if g not in models:
            models[g] = load_extractor(g, args.tag, "w")
        x = load_audio(s.audio_path, SR).numpy().astype(np.float32)
        y = run_extractor(models[g], "w", x, s.name, args.mask_power).astype(np.float32)
        for kind, events, others in (("human", s.human, s.dog), ("dog", s.dog, s.human)):
            for e in events:
                if overlaps(e, others):
                    continue
                a, b = int(e.start_s * SR), int(e.end_s * SR)
                pin = float((x[a:b].astype(np.float64) ** 2).sum())
                if b - a < SR // 10 or pin <= 0:
                    continue
                rows.append({"session": s.name, "group": g, "kind": kind, "in": pin,
                             "out": float((y[a:b].astype(np.float64) ** 2).sum())})
        print(s.name, flush=True)
    db = lambda a, b: round(10 * math.log10(a / b + 1e-12), 2)
    res = {}
    for g in ("odd", "even", "all"):
        sel = [r for r in rows if g == "all" or r["group"] == g]
        hum = [r for r in sel if r["kind"] == "human"]
        dog = [r for r in sel if r["kind"] == "dog"]
        res[g] = {
            "human_only_events": len(hum),
            "human_left_db": db(sum(r["out"] for r in hum), sum(r["in"] for r in hum)),
            "human_events_20db_quieter": round(float(np.mean([r["out"] < 0.01 * r["in"] for r in hum])), 3),
            "human_left_db_median_event": round(float(np.median([10 * math.log10(r["out"] / r["in"] + 1e-12) for r in hum])), 2),
            "dog_only_events": len(dog),
            "dog_kept_db": db(sum(r["out"] for r in dog), sum(r["in"] for r in dog)),
            "dog_kept_db_median_event": round(float(np.median([10 * math.log10(r["out"] / r["in"] + 1e-12) for r in dog])), 2),
        }
        print(g, res[g], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"real_events{args.tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
