"""Step 15c: does listening with all three microphones find the human sounds better?

About 90% of what Whisper can still read after the no-label pipeline comes from
human events that the single-microphone detector (Step 9) mostly missed. The
raw recordings have three synchronized microphones; this trains the same kind
of detector (frozen VGGish + logistic regression, leave-one-participant-out,
identical frames and labels) with more inputs and compares:

* ``session_mic``  - Step 9 baseline: VGGish of the annotated session file
                      (+ previous / next frame)
* ``spatial``      - level, band level differences and GCC-PHAT delays
                      between the microphones only (no VGGish at all)
* ``session_mic+spatial``
* ``multimic``     - VGGish of every microphone (session mic, clip-on M1 and
                      the remaining one), each with previous / next frame
* ``multimic+spatial``

Microphones are arranged by role, not by name, so every participant looks the
same to the model: "session" (M2 for S sessions, M3 for G sessions), "clip-on"
(M1, always present) and "other" (the third; missing for P-01, P-05, P-06 -
then zeros and a has_other = 0 flag). Spatial features are normalized per
session (minus the session median), because microphone placement changes
between participants.

Target ``human`` (default) or ``dog`` (the hybrid's dog/no-dog router).
Operating points are reported at the SAME dog cost as the baseline: the
threshold at which each detector flags 6% / 25% of dog-only time (the
baseline's strict and balanced points), and the resulting human recall.

Usage (from inside speech_anonymization/), after extract_spatial_features and
extract_multimic_vggish:
    python -m scripts.train_multimic_detector
    python -m scripts.train_multimic_detector --target dog
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share, with_context
from vad_anonymization.elte_barks import load_sessions

OUT = Path(__file__).resolve().parent.parent / "outputs"
FEATURE_SETS = ["session_mic", "spatial", "session_mic+spatial", "multimic", "multimic+spatial"]
DOG_COSTS = [0.06, 0.249]  # baseline strict / balanced: share of dog-only frames flagged


def roles(session_name: str) -> tuple[str, str]:
    """(session mic, other mic) for a session name like P-05-G-P_1."""
    return ("M2", "M3") if session_name.split("-")[2] == "S" else ("M3", "M2")


def spatial_by_role(z, session_name: str) -> np.ndarray:
    """Reorder the spatial features from mic names to roles; minus the session median; NaN -> 0."""
    names = [str(n) for n in z["names"]]
    X = z["X"].astype(np.float32)
    col = {n: X[:, i] for i, n in enumerate(names)}
    s, o = roles(session_name)

    def pair(kind: str, a: str, b: str, band: str = "") -> np.ndarray:
        suffix = f"_{band}" if band else ""
        if f"{kind}_{a}{b}{suffix}" in col:
            return col[f"{kind}_{a}{b}{suffix}"]
        v = col[f"{kind}_{b}{a}{suffix}"]
        return v if kind == "gcc_peak" else -v

    feats, centre = [], []
    feats += [col[f"level_{s}"], col["level_M1"], col[f"level_{o}"]]
    centre += [True] * 3
    bands = ["100-500", "500-1500", "1500-4000", "4000-8000"]
    for a, b in [(s, "M1"), (s, o), ("M1", o)]:
        feats += [pair("ild", a, b, band) for band in bands]
        centre += [True] * 4
        feats += [pair("tdoa_ms", a, b)]
        centre += [True]
        feats += [pair("gcc_peak", a, b)]
        centre += [False]
    F = np.stack(feats, 1)
    for j, c in enumerate(centre):
        if c and np.isfinite(F[:, j]).any():
            F[:, j] -= np.nanmedian(F[:, j])
    has_other = np.full((len(F), 1), float(np.isfinite(col[f"level_{o}"]).any()), np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    # local contrast: how much each cue changes against the surrounding 5 s (the room is mostly background)
    k = int(round(5 / HOP_S))
    pad = np.pad(F, ((k, k), (0, 0)), mode="edge")
    cs = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), pad]), 0)
    local = (cs[2 * k + 1 :] - cs[: -2 * k - 1]) / (2 * k + 1)
    return np.hstack([with_context(F), F - local[: len(F)], has_other])


def vggish(name: str, mic: str, n: int) -> tuple[np.ndarray, bool]:
    f = OUT / "multimic_vggish" / f"{name}_{mic}.npz"
    if not f.exists():
        return np.zeros((n, 128), np.float32), False
    X = np.load(f)["X"].astype(np.float32)[:n]
    if len(X) < n:
        X = np.vstack([X, np.repeat(X[-1:], n - len(X), 0)])
    return X, True


def build(sess, feature_set: str, n: int) -> np.ndarray:
    base = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")["X"].astype(np.float32)
    parts = []
    if feature_set.startswith("session_mic"):
        parts.append(with_context(base))
    if feature_set.startswith("multimic"):
        s, o = roles(sess.name)
        m1, _ = vggish(sess.name, "M1", n)
        other, ok = vggish(sess.name, o, n)
        parts += [with_context(base), with_context(m1), with_context(other), np.full((n, 1), float(ok), np.float32)]
    if "spatial" in feature_set:
        z = np.load(OUT / "spatial_features" / f"{sess.name}.npz")
        sp = spatial_by_role(z, sess.name)[:n]
        if len(sp) < n:
            sp = np.vstack([sp, np.repeat(sp[-1:], n - len(sp), 0)])
        parts.append(sp)
    return np.hstack(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--target", choices=["human", "dog"], default="human")
    p.add_argument("--feature-sets", nargs="+", default=FEATURE_SETS)
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, default=OUT / "multimic_detector")
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sessions = load_sessions(args.data_root)
    meta = []
    for sess in sessions:
        c = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")["centres"]
        meta.append({"session": sess, "group": sess.name[:4], "centres": c,
                     "human": overlap_share(c, sess.human, HOP_S / 2), "dog": overlap_share(c, sess.dog, HOP_S / 2)})
    groups = sorted({m["group"] for m in meta})
    human = np.concatenate([m["human"] for m in meta])
    dog = np.concatenate([m["dog"] for m in meta])
    if args.target == "human":
        y_soft, pos, neg = human, human >= 0.5, human == 0
        cost_mask = (human == 0) & (dog >= 0.5)       # dog-only frames: flagging them loses dog sound
    else:
        y_soft, pos, neg = dog, dog >= 0.5, dog == 0
        cost_mask = (dog == 0) & (human >= 0.5)       # human-only frames sent to "subtract": privacy risk
    gid = np.concatenate([[m["group"]] * len(m["centres"]) for m in meta])

    rows = []
    out_dir = args.output_dir / args.target
    out_dir.mkdir(parents=True, exist_ok=True)
    for fs in args.feature_sets:
        X = np.vstack([build(m["session"], fs, len(m["centres"])) for m in meta])
        prob = np.zeros(len(X))
        for g in groups:
            tr = (gid != g) & (pos | neg)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000))
            clf.fit(X[tr], pos[tr].astype(int))
            prob[gid == g] = clf.predict_proba(X[gid == g])[:, 1]
        ev = pos | neg
        row = {"feature_set": fs, "n_features": X.shape[1],
               "roc_auc": round(float(roc_auc_score(pos[ev], prob[ev])), 3),
               "average_precision": round(float(average_precision_score(pos[ev], prob[ev])), 3)}
        for cost in DOG_COSTS:
            th = float(np.quantile(prob[cost_mask], 1 - cost))
            row[f"threshold@cost{cost}"] = round(th, 4)
            row[f"recall@cost{cost}"] = round(float((prob[pos] >= th).mean()), 3)
        for rec in (0.7, 0.9):
            th = float(np.quantile(prob[pos], 1 - rec))
            row[f"cost@recall{rec}"] = round(float((prob[cost_mask] >= th).mean()), 3)
        # per-participant AUC: is any gain carried by a few people?
        aucs = []
        for g in groups:
            sel = (gid == g) & ev
            if pos[sel].any() and (~pos[sel]).any():
                aucs.append(roc_auc_score(pos[sel], prob[sel]))
        row["median_participant_auc"] = round(float(np.median(aucs)), 3)
        rows.append(row)
        np.savez_compressed(out_dir / f"oof_{fs.replace('+', '_')}.npz", prob=prob, human=human, dog=dog)
        print(row, flush=True)
    with open(out_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "info.json").write_text(json.dumps({"target": args.target, "C": args.C, "groups": len(groups),
                                                     "evaluation": "leave-one-participant-out"}, indent=2))


if __name__ == "__main__":
    main()
