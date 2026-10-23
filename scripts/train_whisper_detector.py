"""Step 25b: does Whisper's encoder find human sounds better than AudioSet scores?

Same setup as Steps 17 and 23, only the input changes: leave one participant
out, logistic regression (C=0.05, balanced), 5-frame smoothing, threshold at a
fixed share of dog-only frames flagged. Frames are the usual 0.24 s hop /
0.96 s window; the Whisper features are the mean of the 20 ms encoder frames
inside each window (``extract_whisper_encoder``).

Inputs compared:

* ``eat``      - EfficientAT mn40 class scores (the current default, Step 17);
* ``wenc``     - Whisper small encoder, mean over the window;
* ``wenc+eat`` - both side by side.

Each is trained human vs everything (Steps 17-22) and human vs dog-only
(Step 23).

Writes ``outputs/human_detector_whisper/`` (``oof_<input>_<objective>.npz``,
``summary.json``, ``thresholds.json``) and ``outputs/configs_v25.json``.

Usage (from inside speech_anonymization/):
    python -m scripts.train_whisper_detector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share
from scripts.train_micfusion_detector import COSTS
from scripts.train_multimic_detector import OUT, roles
from vad_anonymization.elte_barks import load_sessions

LABELS = {0.03: "c03", 0.06: "c06", 0.10: "c10", 0.15: "c15", 0.249: "c25"}
WIN_S = 0.96


def pool_whisper(name: str, centres: np.ndarray) -> np.ndarray:
    """Mean of the 20 ms Whisper frames inside each 0.96 s window (features saved by extract_whisper_encoder)."""
    z = np.load(OUT / "whisper_enc" / f"{name}.npz")
    return pool_frames(z["feats"], z["times"], centres)


def pool_frames(feats: np.ndarray, t: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Mean of the 20 ms encoder frames inside the 0.96 s window around each centre."""
    f = feats.astype(np.float32)
    cs = np.vstack([np.zeros((1, f.shape[1]), np.float32), np.cumsum(f, 0)])
    a = np.searchsorted(t, centres - WIN_S / 2)
    b = np.maximum(a + 1, np.searchsorted(t, centres + WIN_S / 2))
    b = np.minimum(b, len(t))
    a = np.minimum(a, b - 1)
    return (cs[b] - cs[a]) / (b - a)[:, None]


def load(sessions) -> list[dict]:
    meta = []
    for s in sessions:
        c = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        eat = np.load(OUT / "efficientat_mn40" / f"{s.name}_{roles(s.name)[0]}.npz")["logits"].astype(np.float32)[: len(c)]
        meta.append({"name": s.name, "group": s.name[:4], "eat": eat, "wenc": pool_whisper(s.name, c),
                     "h": overlap_share(c, s.human, HOP_S / 2), "d": overlap_share(c, s.dog, HOP_S / 2)})
    return meta


def lopo(meta: list[dict], X: np.ndarray, pos: np.ndarray, use: np.ndarray, C: float = 0.05) -> np.ndarray:
    """Leave-one-participant-out probabilities, smoothed over 5 frames within each session."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gid = np.concatenate([[m["group"]] * len(m["h"]) for m in meta])
    sid = np.concatenate([[k] * len(m["h"]) for k, m in enumerate(meta)])
    prob = np.zeros(len(gid))
    for g in sorted(set(gid)):
        tr = (gid != g) & use
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), pos[tr])
        te = gid == g
        raw = clf.predict_proba(sc.transform(X[te]))[:, 1]
        for k in sorted(set(sid[te])):
            sel = sid[te] == k
            raw[sel] = uniform_filter1d(raw[sel], 5, mode="nearest")
        prob[te] = raw
    return prob


def main() -> None:
    from sklearn.metrics import roc_auc_score

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", default=["eat", "wenc", "wenc+eat"])
    args = p.parse_args()
    out = OUT / "human_detector_whisper"
    out.mkdir(parents=True, exist_ok=True)
    meta = load(load_sessions(PROJECT_ROOT / "data" / "elte_barks"))
    h = np.concatenate([m["h"] for m in meta])
    d = np.concatenate([m["d"] for m in meta])
    pos, dog_only, background = h >= 0.5, (h == 0) & (d >= 0.5), (h == 0) & (d == 0)
    objectives = {"all": pos | dog_only | background, "dog": pos | dog_only}
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    thresholds = json.loads((out / "thresholds.json").read_text()) if (out / "thresholds.json").exists() else {}
    for inp in args.inputs:
        X = np.hstack([np.vstack([m[k] for m in meta]) for k in inp.split("+")])
        for obj, use in objectives.items():
            name = f"{inp}_{obj}"
            prob = lopo(meta, X, pos, use)
            th = {LABELS[q]: float(np.quantile(prob[dog_only], 1 - q)) for q in COSTS}
            thresholds[name] = th
            summary[name] = {"auc_human_vs_dog": round(float(roc_auc_score(pos[pos | dog_only], prob[pos | dog_only])), 4),
                             **{f"recall@{LABELS[q]}": round(float((prob[pos] >= th[LABELS[q]]).mean()), 3) for q in COSTS},
                             **{f"background@{LABELS[q]}": round(float((prob[background] >= th[LABELS[q]]).mean()), 3) for q in COSTS}}
            np.savez_compressed(out / f"oof_{name}.npz", prob=prob, human=h, dog=d)
            print(name, summary[name], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    cfg = {f"{n.replace('+', '_')}_mute_{c}": ["trained", round(t, 4), None, f"outputs/human_detector_whisper/oof_{n}.npz"]
           for n, th in thresholds.items() for c, t in th.items()}
    (OUT / "configs_v25.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
