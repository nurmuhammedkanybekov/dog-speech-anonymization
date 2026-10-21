"""Step 17a: a small temporal model on top of the EfficientAT class scores.

Steps 9-16 classify every 0.24 s frame on its own (logistic regression) and
then smooth over 5 frames. Human sounds in the play sessions come in
patterns: a word, a pause, a laugh, the owner calling the dog's name. So a
model that looks a few seconds around each frame should find more of the quiet
ones and confuse fewer dog sounds.

Model (fixed in advance, not tuned on the test participants): per-frame
input = the 527 AudioSet class scores of one microphone (EfficientAT,
Step 16), standardized; 1x1 conv to 64 channels; 4 residual blocks of
dilated 1-D convolutions (kernel 3, dilation 1/2/4/8 -> sees +-15 frames,
i.e. +-3.6 s); 1x1 conv to one logit per frame. About 100k parameters.

Training, leave-one-participant-out like everything before: the 16 training
participants' sessions, every microphone as its own sequence (synchronized,
same labels), random 48 s crops, binary cross-entropy on clear frames only
(mostly human / no human), positive class weighted to balance. 600 steps of
Adam. Three seeds are averaged. The unseen participant's microphones are then
fused by the maximum, as in Step 15.

Outputs use the Step 9 / 16 formats (``oof_predictions.npz``, comparison,
thresholds), so ``run_pipeline_real`` can use them directly.

Usage (from inside speech_anonymization/), after extract_efficientat:
    python -m scripts.train_temporal_detector
    python -m scripts.train_temporal_detector --target dog
    python -m scripts.train_temporal_detector --features efficientat_mn40     # the larger EfficientAT
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.train_human_detector import HOP_S, PROJECT_ROOT, overlap_share
from scripts.train_micfusion_detector import COSTS
from scripts.train_multimic_detector import OUT, roles
from vad_anonymization.elte_barks import load_sessions


class TemporalCNN(nn.Module):
    def __init__(self, n_in: int, ch: int = 64, dilations=(1, 2, 4, 8), dropout: float = 0.2) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(n_in, ch, 1))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.ReLU(), nn.Conv1d(ch, ch, 3, padding=d, dilation=d), nn.ReLU(), nn.Dropout(dropout),
                          nn.Conv1d(ch, ch, 1)) for d in dilations)
        self.out = nn.Sequential(nn.ReLU(), nn.Conv1d(ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, n_in, T) -> (B, T)
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)[:, 0]


def load_views(name: str, n: int, feat_dir: Path) -> dict[str, np.ndarray]:
    views = {}
    for m in ("M1", "M2", "M3"):
        f = feat_dir / f"{name}_{m}.npz"
        if f.exists():
            X = np.load(f)["logits"].astype(np.float32)[:n]
            views[m] = np.vstack([X, np.repeat(X[-1:], n - len(X), 0)]) if len(X) < n else X
    return views


def train_fold(seqs: list[tuple[np.ndarray, np.ndarray]], n_in: int, seed: int, steps: int = 600,
               crop: int = 200, batch: int = 16) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allX = np.vstack([x for x, _ in seqs])
    mu, sd = allX.mean(0), allX.std(0) + 1e-3
    ys = np.concatenate([y for _, y in seqs])
    clear = (ys >= 0.5) | (ys == 0)
    pos_w = float(((ys == 0) & clear).sum() / max(1, (ys >= 0.5).sum()))
    model = TemporalCNN(n_in)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lens = np.array([len(x) for x, _ in seqs], dtype=float)
    p = lens / lens.sum()
    model.train()
    for step in range(steps):
        xb, yb, wb = [], [], []
        for k in rng.choice(len(seqs), batch, p=p):
            x, y = seqs[k]
            a = int(rng.integers(0, max(1, len(x) - crop + 1)))
            xs, ysl = x[a : a + crop], y[a : a + crop]
            if len(xs) < crop:
                pad = crop - len(xs)
                xs = np.vstack([xs, np.repeat(xs[-1:], pad, 0)])
                ysl = np.concatenate([ysl, np.full(pad, 0.25)])      # 0.25 = ignored (not clear)
            xb.append((xs - mu) / sd)
            yb.append(ysl)
        xb = torch.from_numpy(np.stack(xb)).transpose(1, 2).float()
        y = torch.from_numpy(np.stack(yb)).float()
        mask = ((y >= 0.5) | (y == 0)).float()
        target = (y >= 0.5).float()
        w = mask * torch.where(target > 0, torch.tensor(pos_w), torch.tensor(1.0))
        loss = (nn.functional.binary_cross_entropy_with_logits(model(xb), target, reduction="none") * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        for g in opt.param_groups:                                   # cosine decay
            g["lr"] = 1e-3 * 0.5 * (1 + np.cos(np.pi * (step + 1) / steps))
    model.eval()
    return model, mu, sd


@torch.no_grad()
def predict(model: nn.Module, mu: np.ndarray, sd: np.ndarray, x: np.ndarray) -> np.ndarray:
    xt = torch.from_numpy((x - mu) / sd).float().T[None]
    return torch.sigmoid(model(xt))[0].numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--target", choices=["human", "dog"], default="human")
    p.add_argument("--features", default="efficientat", help="folder in outputs/ with <session>_<mic>.npz (logits)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = args.output_dir or OUT / f"temporal_{args.target}_{args.features}"
    out.mkdir(parents=True, exist_ok=True)
    feat_dir = OUT / args.features

    from sklearn.metrics import average_precision_score, roc_auc_score

    meta = []
    for sess in load_sessions(args.data_root):
        c = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")["centres"]
        h, d = overlap_share(c, sess.human, HOP_S / 2), overlap_share(c, sess.dog, HOP_S / 2)
        meta.append({"name": sess.name, "group": sess.name[:4], "n": len(c), "own": roles(sess.name)[0],
                     "views": load_views(sess.name, len(c), feat_dir), "y": h if args.target == "human" else d,
                     "human": h, "dog": d})
    n_in = next(iter(meta[0]["views"].values())).shape[1]
    groups = sorted({m["group"] for m in meta})
    human = np.concatenate([m["human"] for m in meta])
    dog = np.concatenate([m["dog"] for m in meta])
    y = human if args.target == "human" else dog
    pos, neg = y >= 0.5, y == 0
    cost_mask = (human == 0) & (dog >= 0.5) if args.target == "human" else (dog == 0) & (human >= 0.5)
    variants = {k: np.zeros(len(y)) for k in ("temporal_own_mic", "temporal_max")}
    t0 = time.time()
    for g in groups:
        seqs = [(x, m["y"]) for m in meta if m["group"] != g for x in m["views"].values()]
        models = [train_fold(seqs, n_in, seed, args.steps) for seed in range(args.seeds)]
        i = 0
        for m in meta:
            sl = slice(i, i + m["n"])
            i += m["n"]
            if m["group"] != g:
                continue
            per_mic = {k: np.mean([predict(mo, mu, sd, x) for mo, mu, sd in models], 0) for k, x in m["views"].items()}
            variants["temporal_own_mic"][sl] = per_mic[m["own"]]
            variants["temporal_max"][sl] = np.max(list(per_mic.values()), 0)
        print(f"  fold {g} done ({time.time() - t0:.0f}s)", flush=True)

    ev = pos | neg
    gid = np.concatenate([[m["group"]] * m["n"] for m in meta])
    rows, thresholds = [], {}
    for name, prob in variants.items():
        aucs = [roc_auc_score(pos[(gid == g) & ev], prob[(gid == g) & ev]) for g in groups
                if pos[(gid == g) & ev].any() and (~pos[(gid == g) & ev]).any()]
        row = {"variant": name, "roc_auc": round(float(roc_auc_score(pos[ev], prob[ev])), 3),
               "average_precision": round(float(average_precision_score(pos[ev], prob[ev])), 3),
               "median_participant_auc": round(float(np.median(aucs)), 3)}
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
    for name, prob in variants.items():
        np.savez_compressed(out / f"oof_{name}.npz", prob=prob, human=human, dog=dog)
    (out / "info.json").write_text(json.dumps({"features": args.features, "seeds": args.seeds, "steps": args.steps,
                                                "target": args.target, "n_in": int(n_in)}, indent=2))


if __name__ == "__main__":
    main()
