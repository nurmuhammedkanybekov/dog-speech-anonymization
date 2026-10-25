"""Steps 18a, 25d and 26: train the deployable human detector on ALL participants and save it.

``--recipe v3`` (default since Step 26): as v2, plus call-type training weights
(alpha = 1: every dog call type weighs the same, see train_calltype_detector);
thresholds from ``outputs/configs_v26.json`` (``barkw10``). Output:
``models/human_detector_v3.npz``.

``--recipe v2`` (default since Step 25): Whisper small encoder (mean over the
0.96 s window) + EfficientAT mn40 class scores, trained on human vs dog-only
frames, thresholds at 3 / 6 / 10% of dog-only frames flagged (light /
standard / strict) from the leave-one-participant-out predictions of the same
recipe (``outputs/human_detector_whisper``, input ``wenc+eat``, objective
``dog``). Output: ``models/human_detector_v2.npz``.

``--recipe eat40`` is the Step 18 model described below.

The recommended setup from Steps 16-17 (EfficientAT mn40 class scores of one
microphone -> logistic regression -> 5-frame smoothing) is trained once on
all 17 participants. The thresholds come from the leave-one-participant-out
predictions of the same recipe (``human_detector_micfusion_eat40_logits``,
variant ``baseline+smooth``), so each named strictness level has a measured
dog cost and privacy result on unseen people (see README Step 17).

Output: ``models/human_detector_eat40.npz`` (about 20 kB: scaler, weights,
thresholds), used by ``scripts/anonymize_recording.py``.

Usage (from inside speech_anonymization/), after extract_efficientat --model mn40_as:
    python -m scripts.train_final_detector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share
from scripts.train_multimic_detector import OUT, roles
from vad_anonymization.elte_barks import load_sessions

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "human_detector_eat40.npz"
MODEL_V2 = Path(__file__).resolve().parent.parent / "models" / "human_detector_v2.npz"
MODEL_V3 = Path(__file__).resolve().parent.parent / "models" / "human_detector_v3.npz"
LEVELS_V2 = {"light": "c03", "standard": "c06", "strict": "c10"}
# strictness -> share of dog-only frames flagged (LOPO), see README Step 17
LEVELS = {"light": "0.06", "standard": "0.1", "strict": "0.15"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", choices=["v3", "v2", "eat40"], default="v3")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--features", default="efficientat_mn40")
    p.add_argument("--lopo-dir", type=Path, default=OUT / "human_detector_micfusion_eat40_logits")
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--exclude", nargs="*", default=[], help="participants to leave out (to test the saved model on them)")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.recipe in ("v2", "v3"):
        return train_v2(args, calltype_alpha=1.0 if args.recipe == "v3" else 0.0)
    args.output = args.output or MODEL_PATH

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xs, ys = [], []
    sessions = [s for s in load_sessions(args.data_root) if s.name[:4] not in args.exclude]
    for sess in sessions:
        c = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")["centres"]
        X = np.load(OUT / args.features / f"{sess.name}_{roles(sess.name)[0]}.npz")["logits"].astype(np.float32)[: len(c)]
        h = overlap_share(c[: len(X)], sess.human, HOP_S / 2)
        Xs.append(X)
        ys.append(h)
    X, y = np.vstack(Xs), np.concatenate(ys)
    use = (y >= 0.5) | (y == 0)
    scaler = StandardScaler().fit(X[use])
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(scaler.transform(X[use]), y[use] >= 0.5)
    th = json.loads((args.lopo_dir / "thresholds.json").read_text())["baseline+smooth"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             levels=np.array(list(LEVELS)), thresholds=np.array([th[LEVELS[k]] for k in LEVELS], np.float32),
             smooth=np.int32(5), hop_s=np.float32(HOP_S), efficientat_model=np.array("mn40_as"),
             trained_on=np.array(f"ELTE BARKS play sessions, {len({x.name[:4] for x in sessions})} participants, "
                                 f"{int(use.sum())} frames"))
    print(f"saved {args.output}: thresholds {dict(zip(LEVELS, [round(th[LEVELS[k]], 3) for k in LEVELS]))}")


def train_v2(args, calltype_alpha: float = 0.0) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from scripts.train_whisper_detector import pool_whisper

    from scripts.train_calltype_detector import calltype_weights, frame_call_type

    Xs, hs, ds, ts = [], [], [], []
    sessions = [s for s in load_sessions(args.data_root) if s.name[:4] not in args.exclude]
    for sess in sessions:
        c = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")["centres"]
        eat = np.load(OUT / "efficientat_mn40" / f"{sess.name}_{roles(sess.name)[0]}.npz")["logits"].astype(np.float32)[: len(c)]
        c = c[: len(eat)]
        Xs.append(np.hstack([pool_whisper(sess.name, c), eat]))
        hs.append(overlap_share(c, sess.human, HOP_S / 2))
        ds.append(overlap_share(c, sess.dog, HOP_S / 2))
        ts.append(frame_call_type(c, sess))
    X, h, d, ty = np.vstack(Xs), np.concatenate(hs), np.concatenate(ds), np.concatenate(ts)
    use = (h >= 0.5) | ((h == 0) & (d >= 0.5))                    # human vs dog-only (Step 23 objective)
    w = calltype_weights(ty, (h == 0) & (d >= 0.5), calltype_alpha)
    scaler = StandardScaler().fit(X[use])
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(
        scaler.transform(X[use]), h[use] >= 0.5, sample_weight=w[use])
    if calltype_alpha > 0:
        cfg = json.loads((OUT / "configs_v26.json").read_text())
        th = {c: cfg[f"barkw10_mute_{c}"][1] for c in LEVELS_V2.values()}
    else:
        th = json.loads((OUT / "human_detector_whisper" / "thresholds.json").read_text())["wenc+eat_dog"]
    out = args.output or (MODEL_V3 if calltype_alpha > 0 else MODEL_V2)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
             coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             levels=np.array(list(LEVELS_V2)), thresholds=np.array([th[LEVELS_V2[k]] for k in LEVELS_V2], np.float32),
             smooth=np.int32(5), hop_s=np.float32(HOP_S), efficientat_model=np.array("mn40_as"),
             features=np.array("whisper_small+mn40_as"), calltype_alpha=np.float32(calltype_alpha),
             trained_on=np.array(f"ELTE BARKS play sessions, {len({x.name[:4] for x in sessions})} participants, "
                                 f"{int(use.sum())} frames (human vs dog-only)"))
    print(f"saved {out}: thresholds {dict(zip(LEVELS_V2, [round(th[LEVELS_V2[k]], 3) for k in LEVELS_V2]))}")


if __name__ == "__main__":
    main()
