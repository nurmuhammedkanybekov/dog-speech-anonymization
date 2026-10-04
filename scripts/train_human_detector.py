"""Step 9: a human-sound detector adapted to the ELTE recordings.

Silero VAD finds only ~5% of the annotated human time in the real play
sessions (Step 5): the human sounds are quiet, short, and often not speech.
This trains a small classifier on frozen VGGish embeddings (see
vad_anonymization/vggish_features.py) using the expert "Human" annotations.

Evaluation is leave-one-participant-out: the model never sees the person
(or dog) it is tested on, so the numbers say how it would do on a new
recording, not how well it memorized a voice. All numbers come from these
out-of-fold predictions.

Reported per operating point (threshold chosen to reach a target recall of
human time):
* ``human_time_recall``   - share of annotated human time that is flagged
* ``event_recall``        - share of human events with at least one flagged frame
* ``false_alarm_share``   - share of non-human time flagged
* ``dog_call_flagged``    - share of dog-call time (without human) flagged:
                            what removal would touch that it should not

Usage (from inside speech_anonymization/):
    python -m scripts.train_human_detector                 # extract (cached) + evaluate
    python -m scripts.train_human_detector --labels outputs/labeling/human_event_labels.csv --tiers A_speech
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions
from vad_anonymization.vggish_features import VGGishEmbedder, normalize_loudness

SAMPLE_RATE = 16_000
HOP_S = 0.24
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_RECALLS = [0.5, 0.7, 0.8, 0.9, 0.95]


def overlap_share(centres: np.ndarray, events, half: float) -> np.ndarray:
    """For each frame (centre +- half), the share covered by the events."""
    lo, hi = centres - half, centres + half
    cover = np.zeros_like(centres)
    for e in events:
        cover += np.clip(np.minimum(hi, e.end_s) - np.maximum(lo, e.start_s), 0, None)
    return np.clip(cover / (2 * half), 0, 1)


def extract(args) -> None:
    feat_dir = args.output_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    embedder = None
    t0 = time.time()
    for sess in load_sessions(args.data_root):
        path = feat_dir / f"{sess.name}.npz"
        if path.exists():
            continue
        embedder = embedder or VGGishEmbedder()
        audio = normalize_loudness(load_audio(sess.audio_path, SAMPLE_RATE))
        X, centres = embedder.embed(audio, hop_s=HOP_S)
        np.savez_compressed(path, X=X.astype(np.float16), centres=centres)
        print(f"  {sess.name}: {len(X)} frames ({time.time() - t0:.0f}s)", flush=True)


def with_context(X: np.ndarray) -> np.ndarray:
    """Append the previous and next frame's embedding (0.24 s either side)."""
    prev = np.vstack([X[:1], X[:-1]])
    nxt = np.vstack([X[1:], X[-1:]])
    return np.hstack([X, prev, nxt])


def load_dataset(args):
    keep_ids = None
    if args.labels:
        labels = {r["id"]: r["label"] for r in csv.DictReader(open(args.labels))}
        keep_ids = {i for i, lab in labels.items() if lab in args.tiers}
    data = []
    for sess in load_sessions(args.data_root):
        z = np.load(args.output_dir / "features" / f"{sess.name}.npz")
        X, c = z["X"].astype(np.float32), z["centres"]
        human = sess.human if keep_ids is None else [e for k, e in enumerate(sess.human) if f"{sess.name}_{k:03d}" in keep_ids]
        other_human = [e for e in sess.human if e not in human]
        data.append({
            "session": sess.name, "group": sess.name[:4], "X": with_context(X), "centres": c,
            "human": overlap_share(c, human, HOP_S / 2),
            "other_human": overlap_share(c, other_human, HOP_S / 2),
            "dog": overlap_share(c, sess.dog, HOP_S / 2),
            "events": human,
        })
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--labels", type=Path, default=None, help="CSV exported from label_human_events.html")
    p.add_argument("--tiers", nargs="+", default=["A_speech"], help="with --labels: which labels count as the target")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "human_detector")
    args = p.parse_args()
    torch.set_num_threads(1)
    extract(args)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_dataset(args)
    groups = sorted({d["group"] for d in data})
    for d in data:
        d["prob"] = np.zeros(len(d["X"]))
    for g in groups:  # leave one participant out
        train = [d for d in data if d["group"] != g]
        X = np.vstack([d["X"] for d in train])
        # train only on clear frames: mostly human (positive) or no human at all (negative)
        y_soft = np.concatenate([d["human"] for d in train])
        other = np.concatenate([d["other_human"] for d in train])
        use = ((y_soft >= 0.5) | (y_soft == 0)) & (other == 0)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000))
        clf.fit(X[use], (y_soft[use] >= 0.5).astype(int))
        for d in data:
            if d["group"] == g:
                d["prob"] = clf.predict_proba(d["X"])[:, 1]
        print(f"  fold {g}: trained on {use.sum()} frames")

    prob = np.concatenate([d["prob"] for d in data])
    human = np.concatenate([d["human"] for d in data])
    other = np.concatenate([d["other_human"] for d in data])
    dog = np.concatenate([d["dog"] for d in data])
    pos, neg = human >= 0.5, (human == 0) & (other == 0)
    dog_only = neg & (dog >= 0.5)
    auc = roc_auc_score(pos[pos | neg], prob[pos | neg])
    ap = average_precision_score(pos[pos | neg], prob[pos | neg])

    def event_recall(th: float) -> float:
        hit = n = 0
        for d in data:
            for e in d["events"]:
                frames = (d["centres"] + HOP_S / 2 > e.start_s) & (d["centres"] - HOP_S / 2 < e.end_s)
                n += 1
                hit += bool(frames.any() and (d["prob"][frames] >= th).any())
        return hit / n if n else float("nan")

    rows = []
    for target in TARGET_RECALLS:
        th = float(np.quantile(prob[pos], 1 - target))
        rows.append({
            "target_recall": target, "threshold": round(th, 4),
            "human_time_recall": round(float((prob[pos] >= th).mean()), 3),
            "event_recall": round(event_recall(th), 3),
            "false_alarm_share": round(float((prob[neg] >= th).mean()), 3),
            "dog_call_flagged": round(float((prob[dog_only] >= th).mean()), 3),
        })
    with open(args.output_dir / "operating_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    per_session = []
    for d in data:
        s_pos = d["human"] >= 0.5
        per_session.append({"session": d["session"], "n_human_frames": int(s_pos.sum()),
                            "mean_prob_human": round(float(d["prob"][s_pos].mean()), 3) if s_pos.any() else "",
                            "mean_prob_other": round(float(d["prob"][~s_pos].mean()), 3)})
    with open(args.output_dir / "per_session.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_session[0]))
        w.writeheader()
        w.writerows(per_session)
    np.savez_compressed(args.output_dir / "oof_predictions.npz", prob=prob, human=human, other=other, dog=dog)
    info = {"target": "all annotated human events" if not args.labels else f"labels {args.tiers}",
            "evaluation": f"leave-one-participant-out ({len(groups)} participants)", "hop_s": HOP_S,
            "n_frames": int(len(prob)), "n_human_frames": int(pos.sum()), "n_clean_negative_frames": int(neg.sum()),
            "roc_auc": round(float(auc), 3), "average_precision": round(float(ap), 3),
            "base_rate": round(float(pos.sum() / (pos | neg).sum()), 3)}
    (args.output_dir / "summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
