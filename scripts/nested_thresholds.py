"""Step 19a: honest thresholds - chosen without ever looking at the test participant.

Steps 9-18 pick each detector's threshold on the pooled out-of-fold
predictions of ALL participants ("the threshold that flags 6% of dog-only
frames"). The models never saw the test participant, but the threshold
did. That is a mild optimism, and the same for every detector, so the
comparisons are fair, but the absolute numbers could be a little rosy.

Here the threshold is nested. For every test participant g, an inner
leave-one-participant-out loop runs over the 16 other participants only;
the threshold is set on THOSE predictions and then applied to g's
predictions from the model trained on the 16. So nothing about g influences
g's result. The recipe is the recommended one (EfficientAT mn40 class
scores, one microphone, logistic regression, 5-frame smoothing).

Outputs (``outputs/nested_thresholds/``):
* ``flags_c03.npz`` ... ``flags_c25.npz`` - 0/1 per frame (Step 9 order), usable by
  ``run_pipeline_real`` with threshold 0.5 (``configs_v19.json``);
* ``per_participant.csv`` - target vs actual dog cost and human recall, per person;
* ``summary.json``.

Usage (from inside speech_anonymization/):
    python -m scripts.nested_thresholds --worker 0 --n-workers 2     # and --worker 1 (inner fits, cached)
    python -m scripts.nested_thresholds --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share
from scripts.train_micfusion_detector import COSTS, SMOOTH
from scripts.train_multimic_detector import OUT, roles
from vad_anonymization.elte_barks import load_sessions

LABELS = {0.03: "c03", 0.06: "c06", 0.10: "c10", 0.15: "c15", 0.249: "c25"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--features", default="efficientat_mn40")
    p.add_argument("--outer", type=Path, default=OUT / "human_detector_micfusion_eat40_logits" / "all_variants.npz",
                   help="LOPO predictions of the same recipe (variant baseline_smooth)")
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    out = OUT / "nested_thresholds"
    (out / "inner").mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    meta = []
    for sess in load_sessions(args.data_root):
        c = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")["centres"]
        X = np.load(OUT / args.features / f"{sess.name}_{roles(sess.name)[0]}.npz")["logits"].astype(np.float32)[: len(c)]
        meta.append({"name": sess.name, "group": sess.name[:4], "X": X,
                     "human": overlap_share(c, sess.human, HOP_S / 2), "dog": overlap_share(c, sess.dog, HOP_S / 2)})
    groups = sorted({m["group"] for m in meta})

    # inner loop: for test participant g and inner held-out h (h != g), train on the other 15, predict h
    if not args.summarize_only:
        jobs = [(g, h) for g in groups for h in groups if h != g]
        for k, (g, h) in enumerate(jobs):
            f = out / "inner" / f"{g}_{h}.npy"
            if k % args.n_workers != args.worker or f.exists():
                continue
            train = [m for m in meta if m["group"] not in (g, h)]
            X = np.vstack([m["X"] for m in train])
            y = np.concatenate([m["human"] for m in train])
            use = (y >= 0.5) | (y == 0)
            sc = StandardScaler().fit(X[use])
            clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(sc.transform(X[use]), y[use] >= 0.5)
            preds = [uniform_filter1d(clf.predict_proba(sc.transform(m["X"]))[:, 1], SMOOTH, mode="nearest")
                     for m in meta if m["group"] == h]
            np.save(f, np.concatenate(preds))
            print(f"inner {g}/{h} ({k + 1}/{len(jobs)})", flush=True)

    missing = [(g, h) for g in groups for h in groups if h != g and not (out / "inner" / f"{g}_{h}.npy").exists()]
    if missing:
        print(f"{len(missing)} inner fits missing - summary skipped")
        return
    outer = np.load(args.outer)["baseline_smooth"]
    human = np.concatenate([m["human"] for m in meta])
    dog = np.concatenate([m["dog"] for m in meta])
    gid = np.concatenate([[m["group"]] * len(m["X"]) for m in meta])
    dog_only = (human == 0) & (dog >= 0.5)
    pos = human >= 0.5
    flags = {q: np.zeros(len(human), np.float32) for q in COSTS}
    rows = []
    for g in groups:
        # the inner predictions of the 16 other participants, in session order
        inner_p = np.concatenate([np.load(out / "inner" / f"{g}_{h}.npy") for h in groups if h != g])
        sel_inner = gid != g
        assert len(inner_p) == sel_inner.sum()
        inner_cost = dog_only[sel_inner]
        row = {"participant": g, "dog_only_frames": int(dog_only[gid == g].sum()), "human_frames": int(pos[gid == g].sum())}
        for q in COSTS:
            th = float(np.quantile(inner_p[inner_cost], 1 - q))
            f = outer[gid == g] >= th
            flags[q][gid == g] = f
            row[f"threshold_{LABELS[q]}"] = round(th, 4)
            row[f"dog_cost_{LABELS[q]}"] = round(float(f[dog_only[gid == g]].mean()), 3) if dog_only[gid == g].any() else ""
            row[f"recall_{LABELS[q]}"] = round(float(f[pos[gid == g]].mean()), 3) if pos[gid == g].any() else ""
        rows.append(row)
    with open(out / "per_participant.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pooled = json.loads((OUT / "human_detector_micfusion_eat40_logits" / "thresholds.json").read_text())["baseline+smooth"]
    summary = {}
    for q in COSTS:
        np.savez_compressed(out / f"flags_{LABELS[q]}.npz", prob=flags[q], human=human, dog=dog)
        f_pooled = outer >= pooled[str(q)]
        summary[LABELS[q]] = {
            "target_dog_cost": q,
            "nested": {"dog_cost": round(float(flags[q][dog_only].mean()), 3), "human_recall": round(float(flags[q][pos].mean()), 3),
                       "threshold_range": [round(min(r[f"threshold_{LABELS[q]}"] for r in rows), 3),
                                           round(max(r[f"threshold_{LABELS[q]}"] for r in rows), 3)]},
            "pooled": {"dog_cost": round(float(f_pooled[dog_only].mean()), 3), "human_recall": round(float(f_pooled[pos].mean()), 3),
                       "threshold": round(pooled[str(q)], 4)},
        }
        print(LABELS[q], summary[LABELS[q]])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cfg = {f"nested40_mute_{LABELS[q]}": ["trained", 0.5, None, f"outputs/nested_thresholds/flags_{LABELS[q]}.npz"] for q in COSTS}
    (OUT / "configs_v19.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
