"""Step 26: every dog call type counts the same - the fix for barks.

The Step 25 detector muted more barks than the previous one (67% vs 75% of the
barks untouched). All the barks it muted came from one dog (P-03, 16 of its 21
barks): when that dog is left out, the model has seen barks of only two other
dogs (25 barks, 27 s), against 3,521 s of panting. Trained as before, a model
learns what pants look like and treats an unfamiliar bark as "not a pant",
i.e. possibly a voice.

The fix changes only the training weights. Each dog-only training frame gets
the weight (total dog-only frames / frames of its call type) ** alpha, and the
negative frames are then rescaled to their original total weight. With
alpha = 1 every call type weighs the same in total; alpha = 0 is Step 25.
Everything else (Whisper encoder + mn40 input, human vs dog-only, C = 0.05,
balanced classes, 5-frame smoothing, leave one participant out) is unchanged.

Writes ``outputs/human_detector_whisper/oof_wenc+eat_dog_w<alpha>.npz`` and
``outputs/configs_v26.json`` (``barkw05_*`` for alpha 0.5, ``barkw10_*`` for 1).

Usage (from inside speech_anonymization/):
    python -m scripts.train_calltype_detector --alpha 0.5 1.0      # ~10 min each
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.run_pipeline_real import frames_to_segments
from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share
from scripts.train_multimic_detector import OUT
from scripts.train_whisper_detector import load
from vad_anonymization.elte_barks import load_sessions

TYPES = ["Pant", "Growl", "Grunt", "Bark/yelp", "Moan", "Whine", "Other", "Sneeze", "Cough/woof", "Shake"]
COSTS = {0.03: "c03", 0.06: "c06", 0.10: "c10", 0.15: "c15"}


def frame_call_type(centres: np.ndarray, sess) -> np.ndarray:
    """Index into TYPES of the call type covering most of each frame (-1: none)."""
    sh = np.stack([overlap_share(centres, [e for e in sess.dog if e.label == t], HOP_S / 2) for t in TYPES], 1)
    return np.where(sh.max(1) > 0, sh.argmax(1), -1)


def calltype_weights(types: np.ndarray, negative: np.ndarray, alpha: float) -> np.ndarray:
    """Sample weights: negatives reweighted per call type, total negative weight unchanged."""
    w = np.ones(len(types))
    if alpha > 0:
        counts = np.bincount(types[negative & (types >= 0)], minlength=len(TYPES)).astype(float)
        for k in range(len(TYPES)):
            if counts[k] > 0:
                w[negative & (types == k)] = (counts.sum() / counts[k]) ** alpha
        w[negative] *= negative.sum() / w[negative].sum()
    return w


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=float, nargs="+", default=[0.5, 1.0])
    args = p.parse_args()
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    meta = load(sessions)
    for m, s in zip(meta, sessions):
        m["c"] = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"][: len(m["h"])]
        m["type"] = frame_call_type(m["c"], s)
    h = np.concatenate([m["h"] for m in meta])
    d = np.concatenate([m["d"] for m in meta])
    ty = np.concatenate([m["type"] for m in meta])
    pos, dog_only = h >= 0.5, (h == 0) & (d >= 0.5)
    use = pos | dog_only
    X = np.hstack([np.vstack([m["wenc"] for m in meta]), np.vstack([m["eat"] for m in meta])])
    gid = np.concatenate([[m["group"]] * len(m["h"]) for m in meta])
    sid = np.concatenate([[k] * len(m["h"]) for k, m in enumerate(meta)])
    cfg_file = OUT / "configs_v26.json"
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    summary = {}
    for alpha in args.alpha:
        w = calltype_weights(ty, dog_only, alpha)
        prob = np.zeros(len(h))
        for g in sorted(set(gid)):
            tr = (gid != g) & use
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), pos[tr],
                                                                                       sample_weight=w[tr])
            te = gid == g
            raw = clf.predict_proba(sc.transform(X[te]))[:, 1]
            for k in sorted(set(sid[te])):
                sel = sid[te] == k
                raw[sel] = uniform_filter1d(raw[sel], 5, mode="nearest")
            prob[te] = raw
        path = f"outputs/human_detector_whisper/oof_wenc+eat_dog_w{alpha}.npz"
        np.savez_compressed(PROJECT_ROOT / "speech_anonymization" / path, prob=prob, human=h, dog=d)
        tag = f"barkw{int(round(alpha * 10)):02d}"
        res = {}
        for q, c in COSTS.items():
            th = float(np.quantile(prob[dog_only], 1 - q))
            cfg[f"{tag}_mute_{c}"] = ["trained", round(th, 4), None, path]
            intact = defaultdict(lambda: [0, 0])
            i = 0
            for m, s in zip(meta, sessions):
                n = len(m["h"])
                segs = frames_to_segments(m["c"], prob[i : i + n] >= th, m["c"][-1] + 1)
                i += n
                for e in s.dog:
                    if any(min(x.end_s, e.end_s) > max(x.start_s, e.start_s) for x in s.human):
                        continue
                    t = e.label if e.label in TYPES else "Other"
                    intact[t][0] += not any(min(g.end_s, e.end_s) > max(g.start_s, e.start_s) for g in segs)
                    intact[t][1] += 1
            res[c] = {"recall": round(float((prob[pos] >= th).mean()), 3),
                      "intact": {t: f"{a}/{b}" for t, (a, b) in intact.items()}}
        summary[tag] = res
        print(tag, json.dumps(res), flush=True)
    cfg_file.write_text(json.dumps(cfg, indent=1))
    (OUT / "human_detector_whisper" / "summary_calltype.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
