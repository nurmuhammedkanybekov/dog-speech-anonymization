"""Step 19b: sharper edges - classify EfficientAT's own 0.32 s columns instead of 0.96 s windows.

About a third of the dog time that detect-and-mute loses lies within 1 s of a
human sound (Step 17): the 0.96 s detector windows are wider than many human
sounds, so the muted segment spills over onto the neighbouring bark. Wider or
narrower padding only moves along the same trade-off curve.

EfficientAT's feature map has one column per 0.32 s. Here each column goes
through the model's head on its own (``extract_efficientat --columns-only``),
a logistic regression is trained on those column scores (leave one
participant out), smoothed over 3 columns, and a flagged column mutes exactly
its own 0.32 s. Both detectors are then scored on a 10 ms grid against the
expert annotations: share of human time muted vs share of dog-only time
muted, over all thresholds. Then, for the pipeline, each detector's mute
segments at matched dog cost are written out (``segments_<detector>_cXX.json``).

Usage (from inside speech_anonymization/), after extract_efficientat --model mn40_as --columns-only:
    python -m scripts.eval_column_detector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from scripts.train_human_detector import HOP_S, PROJECT_ROOT
from scripts.train_multimic_detector import OUT, roles
from vad_anonymization.elte_barks import load_sessions

COL_S = 0.32
GRID = 100                      # 10 ms scoring grid
TARGETS = {"c03": 0.03, "c06": 0.06, "c10": 0.10, "c15": 0.15, "c25": 0.249}


def interval_share(starts: np.ndarray, length: float, events) -> np.ndarray:
    cover = np.zeros(len(starts))
    for e in events:
        cover += np.clip(np.minimum(starts + length, e.end_s) - np.maximum(starts, e.start_s), 0, None)
    return np.clip(cover / length, 0, 1)


def masks_from(intervals: list[tuple[float, float]], T: int) -> np.ndarray:
    m = np.zeros(T, bool)
    for a, b in intervals:
        m[max(0, int(a * GRID)) : max(0, int(np.ceil(b * GRID)))] = True
    return m


def merge(intervals):
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--cols", default="efficientat_mn40_cols")
    p.add_argument("--window-oof", type=Path, default=OUT / "human_detector_micfusion_eat40_logits" / "oof_single_mic_smooth.npz")
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--pad", type=float, default=0.0, help="extra padding around a flagged column (s)")
    args = p.parse_args()
    out = OUT / "column_detector"
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sessions = load_sessions(args.data_root)
    meta = []
    for s in sessions:
        Xc = np.load(OUT / args.cols / f"{s.name}_{roles(s.name)[0]}.npz")["logits_cols"].astype(np.float32)
        starts = COL_S * np.arange(len(Xc))
        cen = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        T = int((max(starts[-1] + COL_S, cen[-1] + 0.6)) * GRID) + 1
        H, D = masks_from([(e.start_s, e.end_s) for e in s.human], T), masks_from([(e.start_s, e.end_s) for e in s.dog], T)
        meta.append({"s": s, "group": s.name[:4], "X": Xc, "starts": starts, "centres": cen, "T": T,
                     "h": interval_share(starts, COL_S, s.human), "H": H, "dog_only": D & ~H})
    groups = sorted({m["group"] for m in meta})
    for g in groups:
        tr = [m for m in meta if m["group"] != g]
        X = np.vstack([m["X"] for m in tr])
        y = np.concatenate([m["h"] for m in tr])
        use = (y >= 0.5) | (y == 0)
        sc = StandardScaler().fit(X[use])
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[use]), y[use] >= 0.5)
        for m in meta:
            if m["group"] == g:
                m["pc"] = uniform_filter1d(clf.predict_proba(sc.transform(m["X"]))[:, 1], args.smooth, mode="nearest")
        print(f"  fold {g}", flush=True)
    win = np.load(args.window_oof)["prob"]
    i = 0
    for m in meta:
        m["pw"] = win[i : i + len(m["centres"])]
        i += len(m["centres"])

    def segments(m, kind, th):
        if kind == "window":   # Step 16-18 rule: flagged 0.24 s frame -> centre +- (0.12 + 0.12)
            return merge([(c - HOP_S / 2 - 0.12, c + HOP_S / 2 + 0.12) for c in m["centres"][m["pw"] >= th]])
        return merge([(a - args.pad, a + COL_S + args.pad) for a in m["starts"][m["pc"] >= th]])

    def score(kind, th):
        hum = hm = dog = dm = 0
        for m in meta:
            M = masks_from(segments(m, kind, th), m["T"])
            hum += m["H"].sum(); hm += (m["H"] & M).sum(); dog += m["dog_only"].sum(); dm += (m["dog_only"] & M).sum()
        return hm / hum, dm / dog

    curves = {}
    for kind in ("window", "column"):
        allp = np.concatenate([m["pw" if kind == "window" else "pc"] for m in meta])
        ths = np.quantile(allp, np.linspace(0.2, 0.995, 40))
        curves[kind] = sorted(score(kind, t) + (float(t),) for t in ths)
        # threshold reaching each target dog-only time muted (10 ms grid), by bisection on the threshold
        res = {}
        for name, q in TARGETS.items():
            lo, hi = 0.0, 1.0
            for _ in range(25):
                mid = (lo + hi) / 2
                if score(kind, mid)[1] > q:
                    lo = mid
                else:
                    hi = mid
            h, d = score(kind, hi)
            res[name] = {"threshold": round(hi, 4), "human_time_muted": round(h, 3), "dog_only_time_muted": round(d, 3)}
            segs = {m["s"].name: [[round(a, 3), round(b, 3)] for a, b in segments(m, kind, hi)] for m in meta}
            (out / f"segments_{kind}_{name}.json").write_text(json.dumps(segs))
        curves[kind + "_matched"] = res
        print(kind, json.dumps(res), flush=True)
    (out / "curves.json").write_text(json.dumps(curves, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for kind, color, label in (("window", "#1baf7a", "0.96 s windows, every 0.24 s (Steps 16-18)"),
                               ("column", "#2a78d6", "0.32 s columns of the feature map (Step 19)")):
        h, d, _ = zip(*curves[kind])
        ax.plot(np.array(d) * 100, np.array(h) * 100, "-", color=color, lw=1.8, label=label)
    ax.set_xlim(0, 40)
    ax.set_ylim(40, 100)
    ax.set_xlabel("Dog-only vocalization time muted (%)", fontsize=9, color="#52514e")
    ax.set_ylabel("Annotated human time muted (%)", fontsize=9, color="#52514e")
    fig.suptitle("Where to cut: window vs column resolution (EfficientAT mn40, one mic)", x=0.01, ha="left", fontsize=11.5)
    ax.set_title("10 ms scoring grid, 17 participants each tested on a model that never saw them", loc="left", fontsize=8.5,
                 color="#52514e")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(color="#e4e3df", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "window_vs_column.png", dpi=200, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
