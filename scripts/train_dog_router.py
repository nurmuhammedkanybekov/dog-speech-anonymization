"""Step 27e: is a dog vocalizing right now? A router for the dog extractor.

Step 27b showed what decides privacy: run the dog extractor everywhere a
person is detected and the speech that the extractor misses becomes readable
again; run it only where a dog call is actually present (the annotations, an
oracle) and privacy stays almost as good as muting, because there the dog
covers what is left of the speech. So the extractor needs a gate: a detector
of dog vocalizations that works while a person is talking.

Inputs per 0.24 s frame (leave one participant out, logistic regression,
C = 0.05, 3-frame smoothing):

* ``wenc+eat`` (default, one microphone, deployable): the Whisper encoder and
  EfficientAT mn40 scores, the same features as the human detector;
* ``wenc+eat+sp``: plus the microphone cues of Step 13 (level differences
  between the clip-on M1, which is on the dog, and the room microphones, per
  band, and time differences, each relative to the session's median).

Positive: frames at least half covered by a dog call (with or without a
person); negative: frames with no dog call. Frames with a person count 5x,
because the router is only used inside the time where a person was detected.

Writes ``outputs/human_detector_whisper/oof_dog_router_<inputs>_w5.npz``; with
``--final`` also the deployable model ``models/dog_router_v1.npz`` (trained on
all participants; thresholds from the leave-one-participant-out predictions at
2% (safe) and 5% (more) of the person-only frames let through).

Usage (from inside speech_anonymization/):
    python -m scripts.train_dog_router                      # one microphone (~10 min)
    python -m scripts.train_dog_router --inputs wenc+eat+sp # with the microphone cues
    python -m scripts.train_dog_router --final              # + models/dog_router_v1.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.train_multimic_detector import OUT, spatial_by_role
from scripts.train_whisper_detector import load
from vad_anonymization.elte_barks import load_sessions

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HUMAN_WEIGHT = 5.0
SMOOTH = 3
ROUTER_PATH = Path(__file__).resolve().parent.parent / "models" / "dog_router_v1.npz"
LEVELS = {"safe": 0.02, "more": 0.05}


def spatial(sess, n: int) -> np.ndarray:
    F = spatial_by_role(np.load(OUT / "spatial_features" / f"{sess.name}.npz"), sess.name)
    return (F[:n] if len(F) >= n else np.vstack([F, np.repeat(F[-1:], n - len(F), 0)])).astype(np.float32)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", default="wenc+eat", choices=["wenc+eat", "wenc+eat+sp"])
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    meta = load(sessions)
    if "sp" in args.inputs:
        for m, s in zip(meta, sessions):
            m["sp"] = spatial(s, len(m["h"]))
    h = np.concatenate([m["h"] for m in meta])
    d = np.concatenate([m["d"] for m in meta])
    keys = args.inputs.split("+")
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
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), pos[tr],
                                                                                   sample_weight=w[tr])
        te = gid == g
        raw = clf.predict_proba(sc.transform(X[te]))[:, 1]
        for k in sorted(set(sid[te])):
            sel = sid[te] == k
            raw[sel] = uniform_filter1d(raw[sel], SMOOTH, mode="nearest")
        prob[te] = raw
    out = OUT / "human_detector_whisper" / f"oof_dog_router_{args.inputs.replace('+', '_')}_w5.npz"
    np.savez_compressed(out, prob=prob, human=h, dog=d)
    print(f"AUC all frames {roc_auc_score(pos[use], prob[use]):.3f}; "
          f"inside human frames (dog + person vs person only) {roc_auc_score(pos[hum & use], prob[hum & use]):.3f}")
    th = {}
    for name, q in LEVELS.items():
        th[name] = float(np.quantile(prob[hum & neg], 1 - q))
        print(f"  {name}: {q:.0%} of person-only frames let through -> {(prob[hum & pos] >= th[name]).mean():.1%} of dog + person frames")
    print(f"saved {out}")
    if args.final:
        assert args.inputs == "wenc+eat", "the deployable router uses one microphone"
        sc = StandardScaler().fit(X[use])
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[use]), pos[use],
                                                                                   sample_weight=w[use])
        np.savez(ROUTER_PATH, mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
                 coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
                 levels=np.array(list(LEVELS)), thresholds=np.array([th[k] for k in LEVELS], np.float32),
                 smooth=np.int32(SMOOTH), features=np.array("whisper_small+mn40_as"),
                 trained_on=np.array(f"ELTE BARKS play sessions, 17 participants, {int(use.sum())} frames"))
        print(f"saved {ROUTER_PATH}: thresholds {th}")


if __name__ == "__main__":
    main()
