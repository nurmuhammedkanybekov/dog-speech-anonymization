"""Step 22a: what the lab actually loses - dog calls by type, counted as events.

So far the dog cost was "share of dog-only vocalization TIME muted". But 89%
of the annotated dog time is panting (3,521 of 3,966 s). So that number
mostly measures panting, and says little about the rarer calls (growls,
grunts, barks, whines, moans). The lab also analyses calls as EVENTS: a
growl with its middle muted is a damaged growl, however short the cut.

For every detector and setting this counts, per call type, how many of the
annotated dog events are:

* intact - not a single muted sample inside the event;
* damaged - partly muted (less than half);
* lost - half or more muted.

Muted time = the same segments the pipeline mutes (flagged 0.24 s frames,
+-0.12 s padding, merged), from the leave-one-participant-out predictions,
so each session is processed by models that never saw that participant.
Events that overlap a human event are reported separately ("with human"):
there, losing the call is the price of privacy and not a detector mistake.

Usage (from inside speech_anonymization/):
    python -m scripts.eval_dog_events
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.run_pipeline_real import frames_to_segments
from vad_anonymization.elte_barks import load_sessions

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"
GRID = 100  # 10 ms
KEEP = ("vgg1", "eat40_1", "aug2_40", "obj40", "wenc_eat_dog", "wenc_eat_all", "fine_all", "fine_dog")
TYPES = ["Pant", "Growl", "Grunt", "Bark/yelp", "Whine", "Moan", "Other", "Sneeze", "Cough/woof", "Shake"]


def configs() -> dict[str, tuple[float, str]]:
    """name -> (threshold, prediction file), for the detectors compared in Steps 16-23."""
    cfg = {}
    for f in ("configs_v16.json", "configs_v17.json", "configs_v21.json", "configs_v23.json", "configs_v25.json",
              "configs_v25_fine.json"):
        if not (OUT / f).exists():
            continue
        for name, v in json.loads((OUT / f).read_text()).items():
            if name.split("_mute_")[0] in KEEP and len(v) > 3:
                cfg[name] = (v[1], v[3])
    return cfg


def main() -> None:
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    centres = {s.name: np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"] for s in sessions}
    lens = [len(centres[s.name]) for s in sessions]
    rows = []
    for name, (th, path) in sorted(configs().items()):
        z = np.load(PROJECT_ROOT / "speech_anonymization" / path)
        prob, fine = z["prob"], "hop_s" in z
        if fine:  # Step 25c: 20 ms predictions, muted with fine_segments
            from scripts.train_fine_detector import fine_segments

            fparts = dict(zip([s.name for s in sessions], np.split(prob, np.cumsum(z["lengths"])[:-1])))
        counts = defaultdict(lambda: np.zeros(4))      # type -> intact, damaged, lost, n
        time = defaultdict(lambda: np.zeros(2))        # type -> muted s, total s
        i = 0
        for s, n in zip(sessions, lens):
            p = prob[i : i + n]
            i += n
            total = centres[s.name][-1] + 1.0
            T = int(total * GRID) + 1
            m = np.zeros(T, bool)
            if fine:
                t = np.load(OUT / "whisper_enc" / f"{s.name}.npz")["times"]
                segs = fine_segments(t, fparts[s.name] >= th, total, float(z["pad_s"]), float(z["gap_s"]))
            else:
                segs = [(g.start_s, g.end_s) for g in frames_to_segments(centres[s.name], p >= th, total)]
            for a_s, b_s in segs:
                m[int(a_s * GRID) : int(np.ceil(b_s * GRID))] = True
            for e in s.dog:
                a, b = int(e.start_s * GRID), max(int(e.start_s * GRID) + 1, int(np.ceil(e.end_s * GRID)))
                share = m[a:b].mean()
                with_human = any(min(h.end_s, e.end_s) > max(h.start_s, e.start_s) for h in s.human)
                key = (e.label if e.label in TYPES else "Other") + (" (with human)" if with_human else "")
                counts[key] += [share == 0, 0 < share < 0.5, share >= 0.5, 1]
                time[key] += [m[a:b].sum() / GRID, (b - a) / GRID]
        for key in sorted(counts, key=lambda k: -counts[k][3]):
            c, t = counts[key], time[key]
            rows.append({"config": name, "call_type": key, "events": int(c[3]),
                         "intact": round(c[0] / c[3], 3), "damaged": round(c[1] / c[3], 3), "lost": round(c[2] / c[3], 3),
                         "time_muted": round(t[0] / t[1], 3)})
    out = OUT / "dog_events"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "by_call_type.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if "with human" not in r["call_type"]:
            print(r)


if __name__ == "__main__":
    main()
