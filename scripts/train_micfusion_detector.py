"""Step 15d: one detector, every microphone ("listen with all three, trust the loudest opinion").

Step 15c showed that simply stacking the three microphones' features into one
long input does NOT help (AUC 0.727 vs 0.797): the setup differs between
participants (which microphone is missing, where each one stood), and a model
with 1,000+ inputs learns the training participants' setup. What does
generalize is to treat each microphone as one more *view* of the same moment:

1. train ONE single-microphone detector (VGGish + previous/next frame,
   logistic regression, exactly as Step 9) on the frames of all microphones
   of the training participants (3x the data, same labels - the microphones
   are synchronized);
2. run it on every available microphone of the unseen participant and take
   the maximum probability (a sound that is clear on any microphone counts);
3. smooth over 5 frames (+-0.48 s), like a short "hangover" in a VAD.

Everything is leave-one-participant-out. The ablation separates what each
step adds, and every variant is compared at the SAME dog cost as the Step 9
baseline (share of dog-only frames flagged: 6% strict, 25% balanced), plus
10% and 15% for a fuller curve. Outputs use the Step 9 / 11 file formats, so
``run_pipeline_real --human-oof/--dog-oof`` can use them directly.

Usage (from inside speech_anonymization/), after extract_multimic_vggish:
    python -m scripts.train_micfusion_detector                 # human detector
    python -m scripts.train_micfusion_detector --target dog    # dog router for the hybrid
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share, with_context
from scripts.train_multimic_detector import OUT, roles, vggish
from vad_anonymization.elte_barks import load_sessions

COSTS = [0.03, 0.06, 0.10, 0.15, 0.249]
SMOOTH = 5
FEATURES = ["vggish", "eat_emb", "eat_logits", "vggish+eat_emb"]   # Step 16: EfficientAT (extract_efficientat.py)


EAT_DIR = OUT / "efficientat"   # --eat-dir efficientat_mn40 for the larger model


def eat(name: str, mic: str, n: int, key: str) -> np.ndarray | None:
    f = EAT_DIR / f"{name}_{mic}.npz"
    if not f.exists():
        return None
    X = np.load(f)[key].astype(np.float32)[:n]
    return np.vstack([X, np.repeat(X[-1:], n - len(X), 0)]) if len(X) < n else X


def mic_views(sess, n: int, features: str) -> dict[str, np.ndarray]:
    """microphone -> frame features of that microphone, for the chosen feature type."""
    own, _ = roles(sess.name)
    views = {}
    if features in ("vggish", "vggish+eat_emb"):
        z = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")
        views[own] = with_context(z["X"].astype(np.float32))   # the session file itself = its own mic
        for m in ("M1", "M2", "M3"):
            if m != own:
                X, ok = vggish(sess.name, m, n)
                if ok:
                    views[m] = with_context(X)
    if features == "vggish":
        return views
    key = "logits" if features == "eat_logits" else "emb"
    eviews = {m: x for m in ("M1", "M2", "M3") if (x := eat(sess.name, m, n, key)) is not None}
    if features == "vggish+eat_emb":
        return {m: np.hstack([views[m], eviews[m]]) for m in views if m in eviews}
    return eviews


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--target", choices=["human", "dog"], default="human")
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--features", choices=FEATURES, default="vggish")
    p.add_argument("--eat-dir", default="efficientat", help="folder in outputs/ with the EfficientAT features")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    global EAT_DIR
    EAT_DIR = OUT / args.eat_dir
    suffix = "" if args.features == "vggish" else "_" + args.features.replace("+", "_")
    out = args.output_dir or OUT / (("human_detector_micfusion" if args.target == "human" else "hybrid_routing_micfusion") + suffix)
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    meta = []
    for sess in load_sessions(args.data_root):
        z = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")
        c, n = z["centres"], len(z["centres"])
        own, _ = roles(sess.name)
        views = mic_views(sess, n, args.features)
        h, d = overlap_share(c, sess.human, HOP_S / 2), overlap_share(c, sess.dog, HOP_S / 2)
        meta.append({"name": sess.name, "group": sess.name[:4], "n": n, "own": own, "views": views,
                     "y": h if args.target == "human" else d, "human": h, "dog": d})
    groups = sorted({m["group"] for m in meta})
    human = np.concatenate([m["human"] for m in meta])
    dog = np.concatenate([m["dog"] for m in meta])
    y = human if args.target == "human" else dog
    pos, neg = y >= 0.5, y == 0
    # the cost of a false alarm: dog-only frames muted (human target) / human-only frames sent to "subtract" (dog target)
    cost_mask = (human == 0) & (dog >= 0.5) if args.target == "human" else (dog == 0) & (human >= 0.5)

    variants = {k: np.zeros(len(y)) for k in ["baseline", "baseline+smooth", "pooled_own_mic", "pooled_max", "pooled_max+smooth"]}
    for g in groups:
        train = [m for m in meta if m["group"] != g]
        fits = {}
        for kind in ("single", "pooled"):
            Xs = [m["views"][m["own"]] for m in train]
            ys = [m["y"] for m in train]
            if kind == "pooled":
                Xs += [x for m in train for k, x in m["views"].items() if k != m["own"]]
                ys += [m["y"] for m in train for k in m["views"] if k != m["own"]]
            X, yy = np.vstack(Xs), np.concatenate(ys)
            use = (yy >= 0.5) | (yy == 0)
            fits[kind] = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000))
            fits[kind].fit(X[use], (yy[use] >= 0.5).astype(int))
        i = 0
        for m in meta:
            sl = slice(i, i + m["n"])
            i += m["n"]
            if m["group"] != g:
                continue
            base = fits["single"].predict_proba(m["views"][m["own"]])[:, 1]
            per_mic = [fits["pooled"].predict_proba(x)[:, 1] for x in m["views"].values()]
            fused = np.max(per_mic, 0)
            variants["baseline"][sl] = base
            variants["baseline+smooth"][sl] = uniform_filter1d(base, SMOOTH, mode="nearest")
            variants["pooled_own_mic"][sl] = fits["pooled"].predict_proba(m["views"][m["own"]])[:, 1]
            variants["pooled_max"][sl] = fused
            variants["pooled_max+smooth"][sl] = uniform_filter1d(fused, SMOOTH, mode="nearest")
        print(f"  fold {g} done", flush=True)

    ev = pos | neg
    gid = np.concatenate([[m["group"]] * m["n"] for m in meta])
    rows, thresholds = [], {}
    for name, prob in variants.items():
        row = {"variant": name, "roc_auc": round(float(roc_auc_score(pos[ev], prob[ev])), 3),
               "average_precision": round(float(average_precision_score(pos[ev], prob[ev])), 3)}
        aucs = [roc_auc_score(pos[(gid == g) & ev], prob[(gid == g) & ev]) for g in groups
                if pos[(gid == g) & ev].any() and (~pos[(gid == g) & ev]).any()]
        row["median_participant_auc"] = round(float(np.median(aucs)), 3)
        row["participants_improved_vs_baseline"] = ""
        if name != "baseline":
            base_aucs = [roc_auc_score(pos[(gid == g) & ev], variants["baseline"][(gid == g) & ev]) for g in groups
                         if pos[(gid == g) & ev].any() and (~pos[(gid == g) & ev]).any()]
            row["participants_improved_vs_baseline"] = f"{sum(a > b for a, b in zip(aucs, base_aucs))}/{len(aucs)}"
        thresholds[name] = {}
        for cost in COSTS:
            th = float(np.quantile(prob[cost_mask], 1 - cost))
            thresholds[name][str(cost)] = th
            row[f"recall@cost{cost}"] = round(float((prob[pos] >= th).mean()), 3)
        rows.append(row)
        print(row, flush=True)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    final = variants["pooled_max+smooth"]
    key = "prob"
    fname = "oof_predictions.npz" if args.target == "human" else "dog_oof_predictions.npz"
    np.savez_compressed(out / fname, **{key: final}, human=human, dog=dog)
    np.savez_compressed(out / "all_variants.npz", **{k.replace("+", "_"): v for k, v in variants.items()}, human=human, dog=dog)
    print(f"saved {out / fname}")


if __name__ == "__main__":
    main()
