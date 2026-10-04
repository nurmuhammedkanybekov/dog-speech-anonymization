"""Step 11: can the hybrid decide "mute" vs "subtract" without the expert annotations?

The hybrid removal (Step 6) mutes human-only time and subtracts speech only
where a dog vocalizes at the same moment. So far the dog/no-dog decision came
from the expert annotations ("oracle hybrid"). A real anonymizer needs its
own dog detector. This trains one exactly like the human detector (frozen
VGGish features, logistic regression, leave-one-participant-out) and then
checks the routing inside the annotated human time:

* human-only time sent to "mute"       -> correct (private, no dog lost)
* human-only time sent to "subtract"   -> privacy risk (subtraction is weak on real audio)
* human + dog time sent to "subtract"  -> correct
* human + dog time sent to "mute"      -> that bit of dog sound is lost

Needs the cached features from ``train_human_detector``.

Usage (from inside speech_anonymization/):
    python -m scripts.eval_hybrid_routing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share, with_context
from vad_anonymization.elte_barks import load_sessions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--features", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "human_detector" / "features")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "hybrid_routing")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = []
    for sess in load_sessions(args.data_root):
        z = np.load(args.features / f"{sess.name}.npz")
        c = z["centres"]
        data.append({"group": sess.name[:4], "X": with_context(z["X"].astype(np.float32)),
                     "dog": overlap_share(c, sess.dog, HOP_S / 2), "human": overlap_share(c, sess.human, HOP_S / 2)})
    groups = sorted({d["group"] for d in data})
    for g in groups:
        train = [d for d in data if d["group"] != g]
        X = np.vstack([d["X"] for d in train])
        y = np.concatenate([d["dog"] for d in train])
        use = (y >= 0.5) | (y == 0)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000))
        clf.fit(X[use], (y[use] >= 0.5).astype(int))
        for d in data:
            if d["group"] == g:
                d["prob"] = clf.predict_proba(d["X"])[:, 1]
        print(f"  fold {g}")

    prob = np.concatenate([d["prob"] for d in data])
    dog = np.concatenate([d["dog"] for d in data])
    human = np.concatenate([d["human"] for d in data])
    clear = (dog >= 0.5) | (dog == 0)
    auc = roc_auc_score(dog[clear] >= 0.5, prob[clear])

    in_human = human >= 0.5
    h_only = in_human & (dog == 0)
    h_dog = in_human & (dog >= 0.5)
    rows = []
    for th in (0.3, 0.5, 0.7):
        says_dog = prob >= th
        rows.append({
            "dog_threshold": th,
            "dog_time_recall": round(float(says_dog[(dog >= 0.5) & ~in_human].mean()), 3),
            "no_dog_time_flagged": round(float(says_dog[(dog == 0) & ~in_human].mean()), 3),
            "human_only_routed_to_mute": round(float((~says_dog[h_only]).mean()), 3),
            "human_dog_routed_to_subtract": round(float(says_dog[h_dog].mean()), 3),
        })
    with open(args.output_dir / "routing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    info = {"dog_detector_roc_auc": round(float(auc), 3), "evaluation": f"leave-one-participant-out ({len(groups)})",
            "human_frames_without_dog": int(h_only.sum()), "human_frames_with_dog": int(h_dog.sum())}
    (args.output_dir / "summary.json").write_text(json.dumps(info, indent=2))
    np.savez_compressed(args.output_dir / "dog_oof_predictions.npz", prob=prob, dog=dog, human=human)
    print(json.dumps(info, indent=2))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
