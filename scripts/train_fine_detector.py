"""Step 25c: decide what to mute every 20 ms instead of every 0.24 s.

About 40% of the dog sound lost at the standard setting lies within 1 s of a
human sound: the 0.24 s frames, the 0.96 s windows and the +-0.12 s padding
cut into the dog calls just before and after people speak. Whisper's encoder
gives a description every 20 ms, so the boundaries can be placed 12x more
finely.

Every 20 ms frame gets three views of its surroundings, all from the Whisper
small encoder: the mean over +-0.06 s (the sound itself), over +-0.5 s (its
context, like the coarse detector) and over +-1.5 s (the wider scene). A
logistic regression (leave one participant out, C=0.05, balanced, trained on
every 8th frame) gives a probability per frame, which is smoothed over
0.3 s. Labels: human if the frame lies inside an annotated human event,
negative otherwise (``--objective all``) or only inside dog-only calls
(``--objective dog``).

The muting rule (``fine_segments``) flags frames above the threshold, pads
each flagged run by ``PAD_S`` and merges runs closer than ``GAP_S``. The
thresholds are again set at a fixed share of dog-only frames flagged.

Writes ``outputs/human_detector_fine/oof_<objective>.npz`` (``prob`` over all
20 ms frames of all sessions, ``hop_s`` = 0.02, ``pad_s``, ``gap_s``),
``summary.json`` and ``outputs/configs_v25_fine.json`` for ``run_pipeline_real``.

Usage (from inside speech_anonymization/), after ``extract_whisper_encoder``:
    python -m scripts.train_fine_detector --objective all      # ~30 min on 2 cores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.train_multimic_detector import OUT
from vad_anonymization.elte_barks import load_sessions

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HOP_S = 0.02
CONTEXTS = (3, 25, 75)          # half-widths in frames: +-0.06 s, +-0.5 s, +-1.5 s
SMOOTH = 15                     # 0.3 s
PAD_S = 0.06
GAP_S = 0.2
TRAIN_EVERY = 8
LABELS = {0.03: "c03", 0.06: "c06", 0.10: "c10", 0.15: "c15", 0.249: "c25"}


def multiscale(feats: np.ndarray) -> np.ndarray:
    """Means of the encoder frames over several centred windows, side by side."""
    f = feats.astype(np.float32)
    cs = np.vstack([np.zeros((1, f.shape[1]), np.float32), np.cumsum(f, 0, dtype=np.float64).astype(np.float32)])
    n, idx, parts = len(f), np.arange(len(f)), []
    for w in CONTEXTS:
        a, b = np.maximum(0, idx - w), np.minimum(n, idx + w + 1)
        parts.append((cs[b] - cs[a]) / (b - a)[:, None])
    return np.hstack(parts)


def frame_labels(times: np.ndarray, events) -> np.ndarray:
    m = np.zeros(len(times), bool)
    for e in events:
        m[(times >= e.start_s) & (times < e.end_s)] = True
    return m


def fine_segments(times: np.ndarray, flags: np.ndarray, total_s: float, pad_s: float = PAD_S,
                  gap_s: float = GAP_S) -> list[tuple[float, float]]:
    """Flagged 20 ms frames -> padded segments; segments closer than gap_s are merged."""
    segs: list[list[float]] = []
    for t in times[flags]:
        a, b = max(0.0, t - HOP_S / 2 - pad_s), min(total_s, t + HOP_S / 2 + pad_s)
        if segs and a <= segs[-1][1] + gap_s:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    return [(a, b) for a, b in segs]


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", choices=["all", "dog"], default="all")
    args = p.parse_args()
    out = OUT / "human_detector_fine"
    out.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    meta = []
    for s in sessions:
        z = np.load(OUT / "whisper_enc" / f"{s.name}.npz")
        t = z["times"]
        hum, dog = frame_labels(t, s.human), frame_labels(t, s.dog)
        meta.append({"name": s.name, "group": s.name[:4], "t": t, "pos": hum, "dog_only": dog & ~hum, "bg": ~dog & ~hum})
    groups = sorted({m["group"] for m in meta})

    # training subset (every 8th frame, 0.16 s apart) of every session, kept in memory once
    sub = []
    for m in meta:
        X = multiscale(np.load(OUT / "whisper_enc" / f"{m['name']}.npz")["feats"])
        k = np.arange(0, len(X), TRAIN_EVERY)
        use = m["pos"][k] | m["dog_only"][k] | (m["bg"][k] if args.objective == "all" else False)
        sub.append((X[k][use], m["pos"][k][use]))
    probs: dict[str, np.ndarray] = {}
    for g in groups:
        tr = [i for i, m in enumerate(meta) if m["group"] != g]
        Xtr = np.vstack([sub[i][0] for i in tr])
        ytr = np.concatenate([sub[i][1] for i in tr])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
        del Xtr
        for m in meta:
            if m["group"] == g:
                X = multiscale(np.load(OUT / "whisper_enc" / f"{m['name']}.npz")["feats"])
                probs[m["name"]] = uniform_filter1d(clf.predict_proba(sc.transform(X))[:, 1], SMOOTH, mode="nearest")
        print(g, "done", flush=True)

    prob = np.concatenate([probs[m["name"]] for m in meta])
    pos = np.concatenate([m["pos"] for m in meta])
    dog_only = np.concatenate([m["dog_only"] for m in meta])
    bg = np.concatenate([m["bg"] for m in meta])
    th = {LABELS[q]: float(np.quantile(prob[dog_only], 1 - q)) for q in LABELS}
    from sklearn.metrics import roc_auc_score

    summ = {"auc_human_vs_dog": round(float(roc_auc_score(pos[pos | dog_only], prob[pos | dog_only])), 4),
            **{f"recall@{c}": round(float((prob[pos] >= v).mean()), 3) for c, v in th.items()},
            **{f"background@{c}": round(float((prob[bg] >= v).mean()), 3) for c, v in th.items()}}
    print(summ)
    np.savez_compressed(out / f"oof_{args.objective}.npz", prob=prob, hop_s=HOP_S, pad_s=PAD_S, gap_s=GAP_S,
                        lengths=np.array([len(m["t"]) for m in meta]))
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    summary[args.objective] = {"thresholds": th, **summ}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cfg_file = OUT / "configs_v25_fine.json"
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    cfg.update({f"fine_{args.objective}_mute_{c}": ["trained", round(v, 4), None, f"outputs/human_detector_fine/oof_{args.objective}.npz"]
                for c, v in th.items()})
    cfg_file.write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
