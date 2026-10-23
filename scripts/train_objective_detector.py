"""Step 23a: train the detector for the trade-off we actually make.

What anonymization costs is dog sound. Muting background (no human, no dog)
costs nothing the lab analyses. But the detector of Steps 9-21 was trained
to separate human frames from EVERY other frame, background included, so
part of its capacity went into telling human sounds from room noise, which
never matters.

The fix is one line. Train it on human frames vs dog-only frames, and leave
background frames out of the training. The input stays the same (EfficientAT
mn40 class scores, one microphone, logistic regression, 5-frame smoothing,
leave one participant out) and so does the threshold rule (a fixed share of
dog-only frames flagged). This changes the objective, not the model, and
needs no tuning.

Writes ``outputs/human_detector_objective/`` (predictions, thresholds) and
``outputs/configs_v23.json`` for the pipeline.

Usage (from inside speech_anonymization/):
    python -m scripts.train_objective_detector
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share
from scripts.train_micfusion_detector import COSTS
from scripts.train_multimic_detector import OUT, roles
from vad_anonymization.elte_barks import load_sessions

LABELS = {0.03: "c03", 0.06: "c06", 0.10: "c10", 0.15: "c15", 0.249: "c25"}


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    out = OUT / "human_detector_objective"
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        c = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        X = np.load(OUT / "efficientat_mn40" / f"{s.name}_{roles(s.name)[0]}.npz")["logits"].astype(np.float32)[: len(c)]
        meta.append({"group": s.name[:4], "X": X, "h": overlap_share(c, s.human, HOP_S / 2), "d": overlap_share(c, s.dog, HOP_S / 2)})
    groups = sorted({m["group"] for m in meta})
    h = np.concatenate([m["h"] for m in meta])
    d = np.concatenate([m["d"] for m in meta])
    gid = np.concatenate([[m["group"]] * len(m["X"]) for m in meta])
    X = np.vstack([m["X"] for m in meta])
    pos, dog_only, background = h >= 0.5, (h == 0) & (d >= 0.5), (h == 0) & (d == 0)
    variants = {"human_vs_all": pos | dog_only | background, "human_vs_dog": pos | dog_only}
    thresholds, summary = {}, {}
    for name, use in variants.items():
        prob = np.zeros(len(h))
        for g in groups:
            tr = (gid != g) & use
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), pos[tr])
            raw = clf.predict_proba(sc.transform(X[gid == g]))[:, 1]
            # smooth per session (sessions of one participant are contiguous)
            i, parts = 0, []
            for m in [m for m in meta if m["group"] == g]:
                parts.append(uniform_filter1d(raw[i : i + len(m["X"])], 5, mode="nearest"))
                i += len(m["X"])
            prob[gid == g] = np.concatenate(parts)
        thresholds[name] = {LABELS[q]: float(np.quantile(prob[dog_only], 1 - q)) for q in COSTS}
        summary[name] = {"roc_auc_human_vs_rest": round(float(roc_auc_score(pos[pos | dog_only | background], prob[pos | dog_only | background])), 3),
                         "roc_auc_human_vs_dog": round(float(roc_auc_score(pos[pos | dog_only], prob[pos | dog_only])), 3),
                         **{f"recall@{LABELS[q]}": round(float((prob[pos] >= thresholds[name][LABELS[q]]).mean()), 3) for q in COSTS},
                         **{f"background_flagged@{LABELS[q]}": round(float((prob[background] >= thresholds[name][LABELS[q]]).mean()), 3) for q in COSTS}}
        np.savez_compressed(out / f"oof_{name}.npz", prob=prob, human=h, dog=d)
        print(name, summary[name], flush=True)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cfg = {f"obj40_mute_{c}": ["trained", round(t, 4), None, "outputs/human_detector_objective/oof_human_vs_dog.npz"]
           for c, t in thresholds["human_vs_dog"].items()}
    (OUT / "configs_v23.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
