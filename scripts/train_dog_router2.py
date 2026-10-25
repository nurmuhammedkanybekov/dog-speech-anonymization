"""Step 28b: a dog router that also listens to the extractor's output.

Same recipe as the Step 27 router (logistic regression, leave one participant
out, frames with a person weighted 5x, 3-frame smoothing; positive = a dog call
covers at least half of the frame), with more inputs:

* ``x``  - the original audio's features (Whisper encoder + mn40, as Step 27);
* ``y``  - the same two networks on the dog extractor's output
  (``extracted_features``; the extractor of the other participant group);
* ``sp`` - optionally the microphone cues of Step 13.

The extractor's output keeps the dog and removes most of the voice, so a dog
call should stand out more clearly there; any speech it lets through should
still sound like speech to the same networks, which the router learns to
reject.

Writes ``outputs/human_detector_whisper/oof_dog_router2_<inputs>.npz``; with
``--final`` the deployable ``models/dog_router_v2.npz`` (inputs x+y, one
microphone; thresholds from the leave-one-participant-out predictions at 2%
(safe) and 5% (more) of the person-only frames let through).

Usage (from inside speech_anonymization/):
    python -m scripts.train_dog_router2 --inputs x+y
    python -m scripts.train_dog_router2 --inputs x+y --final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.train_dog_router import HUMAN_WEIGHT, LEVELS, SMOOTH, spatial
from scripts.train_multimic_detector import OUT
from scripts.train_whisper_detector import load
from vad_anonymization.elte_barks import load_sessions

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROUTER2_PATH = Path(__file__).resolve().parent.parent / "models" / "dog_router_v2.npz"


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", default="x+y")
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    meta = load(sessions)
    keys = args.inputs.split("+")
    for m, s in zip(meta, sessions):
        n = len(m["h"])
        m["x"] = np.hstack([m["wenc"], m["eat"]])
        if "y" in keys:
            z = np.load(OUT / "extracted_feats" / f"{s.name}.npz")
            m["y"] = np.hstack([z["wenc"].astype(np.float32)[:n], z["eat"].astype(np.float32)[:n]])
        if "sp" in keys:
            m["sp"] = spatial(s, n)
    h = np.concatenate([m["h"] for m in meta])
    d = np.concatenate([m["d"] for m in meta])
    X = np.hstack([np.vstack([m[k] for m in meta]) for k in keys])
    gid = np.concatenate([[m["group"]] * len(m["h"]) for m in meta])
    sid = np.concatenate([[k] * len(m["h"]) for k, m in enumerate(meta)])
    pos, neg = d >= 0.5, d == 0
    use, hum = pos | neg, h >= 0.5
    w = np.where(hum, HUMAN_WEIGHT, 1.0)
    prob = np.zeros(len(h))
    for g in sorted(set(gid)):
        tr = (gid != g) & use
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), pos[tr],
                                                                                     sample_weight=w[tr])
        te = gid == g
        raw = clf.predict_proba(sc.transform(X[te]))[:, 1]
        for k in sorted(set(sid[te])):
            sel = sid[te] == k
            raw[sel] = uniform_filter1d(raw[sel], SMOOTH, mode="nearest")
        prob[te] = raw
    name = args.inputs.replace("+", "_") + (f"_C{args.C:g}" if args.C != 0.05 else "")
    out = OUT / "human_detector_whisper" / f"oof_dog_router2_{name}.npz"
    np.savez_compressed(out, prob=prob, human=h, dog=d)
    print(f"{args.inputs}: AUC all {roc_auc_score(pos[use], prob[use]):.3f}; "
          f"inside human frames {roc_auc_score(pos[hum & use], prob[hum & use]):.3f}")
    th = {}
    for lv, q in {**LEVELS, "10%": 0.10}.items():
        th[lv] = float(np.quantile(prob[hum & neg], 1 - q))
        print(f"  {lv}: {q:.0%} of person-only frames let through -> {(prob[hum & pos] >= th[lv]).mean():.1%} of dog + person frames",
              flush=True)
    if args.final:
        sc = StandardScaler().fit(X[use])
        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(sc.transform(X[use]), pos[use],
                                                                                     sample_weight=w[use])
        np.savez(ROUTER2_PATH, mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
                 coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
                 levels=np.array(list(LEVELS)), thresholds=np.array([th[k] for k in LEVELS], np.float32),
                 smooth=np.int32(SMOOTH), inputs=np.array(args.inputs),
                 features=np.array("whisper_small+mn40_as on the original, then on the extractor output"))
        print(f"saved {ROUTER2_PATH}")


if __name__ == "__main__":
    main()
