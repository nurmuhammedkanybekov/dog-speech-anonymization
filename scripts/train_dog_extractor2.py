"""Step 27c: two microphones - the clip-on microphone is on the DOG.

Every session has the room microphone it was annotated on (M2 shotgun, or M3
cardioid for P-05/P-06) and the wireless clip-on M1. Comparing the levels of
the two shows where M1 is: in 25 of 30 sessions the pants are relatively
louder on M1 than the human sounds are (median +8 dB vs +4.5 dB, relative to
the room microphone). M1 sits on the dog (a collar or harness microphone).

So the level difference between M1 and the room microphone, per time-frequency
bin, tells dog from human: a bin where M1 is unusually loud comes from the dog.
The absolute difference depends on each session's microphone gains and
positions, so it is measured relative to the session's own median difference
per frequency (computed on the whole recording, no labels needed).

The model is the Step 27a extractor with a second input channel (the
session-normalized level difference M1 - room, per bin). Training mixtures
are made inside one session at a time: a dog window and the session's own
human-only events, both taken from BOTH microphones at their real levels, so
each source keeps its real position in the room. The loss regresses a
privacy-weighted Wiener mask on the room microphone (``--beta``: a bin is kept
only where the dog is about 10 dB above the speech). Participant folds as in
Step 27a.

Usage (from inside speech_anonymization/):
    python -m scripts.train_dog_extractor2 --train-group odd      # model for the even participants
    python -m scripts.train_dog_extractor2 --train-group even
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from scripts.finetune_gtcrn import HUMAN_MARGIN_S, participant_group
from scripts.train_dog_extractor import HOP, MODEL_DIR, N_FFT, SEG, SR, istft, stft
from scripts.train_multimic_detector import roles
from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNED = PROJECT_ROOT / "data" / "elte_barks" / "aligned_mics"


class DogExtractor2(nn.Module):
    """(room log power, normalized M1-room level difference) -> mask on the room microphone."""

    def __init__(self, n_freq: int = N_FFT // 2 + 1, hidden: int = 128, in_ch: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=(1, 2), padding=1), nn.GELU())
        f = n_freq
        for _ in range(3):
            f = (f + 1) // 2
        self.proj = nn.Linear(32 * f, 2 * hidden)
        self.rnn = nn.GRU(2 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden + in_ch * n_freq, n_freq)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, C, T, F) -> mask (B, T, F)."""
        B, C, T, F = feats.shape
        z = self.conv(feats).permute(0, 2, 1, 3).reshape(B, T, -1)
        z, _ = self.rnn(torch.nn.functional.gelu(self.proj(z)))
        skip = feats.permute(0, 2, 1, 3).reshape(B, T, C * F)
        return torch.sigmoid(self.out(torch.cat([z, skip], -1)))


def logpow(X: torch.Tensor) -> torch.Tensor:
    return torch.log10(X.abs() ** 2 + 1e-10)


def features2(X0: torch.Tensor, X1: torch.Tensor, ild_med: torch.Tensor) -> torch.Tensor:
    """X0, X1: (B, T, F) STFTs of room mic and M1; ild_med: (B, F) session median of lp1 - lp0."""
    l0, l1 = logpow(X0), logpow(X1)
    f0 = l0 - l0.mean(dim=(1, 2), keepdim=True)
    f1 = torch.clamp(l1 - l0 - ild_med[:, None, :], -3.0, 3.0)        # in bels: +-30 dB
    return torch.stack([f0, f1], 1)


def session_ild(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Median over the recording of log10|X1|^2 - log10|X0|^2 per frequency (frames above the 20th level percentile)."""
    with torch.no_grad():
        X0, X1 = stft(torch.from_numpy(x0)[None])[0], stft(torch.from_numpy(x1)[None])[0]
    l0, l1 = logpow(X0), logpow(X1)
    e = l0.mean(1)
    keep = e >= torch.quantile(e, 0.2)
    return torch.median(l1[keep] - l0[keep], dim=0).values.numpy().astype(np.float32)


def load_pair(s) -> tuple[np.ndarray, np.ndarray]:
    own, _ = roles(s.name)
    x0 = load_audio(s.audio_path, SR).numpy().astype(np.float32)
    x1, sr = sf.read(ALIGNED / f"{s.name}_M1.flac", dtype="float32")
    assert sr == SR
    n = min(len(x0), len(x1))
    return x0[:n], x1[:n]


def extract2(model: DogExtractor2, x0: np.ndarray, x1: np.ndarray, ild: np.ndarray, chunk_s: float = 20.0) -> np.ndarray:
    """Apply the two-microphone extractor to a whole session; returns the room-microphone output."""
    out = np.zeros_like(x0)
    n, step, ov = len(x0), int(chunk_s * SR), 2 * SR
    il = torch.from_numpy(ild)[None]
    with torch.no_grad():
        for a in range(0, n, step):
            lo, hi = max(0, a - ov), min(n, a + step + ov)
            X0 = stft(torch.from_numpy(x0[lo:hi])[None])
            X1 = stft(torch.from_numpy(x1[lo:hi])[None])
            y = istft(model(features2(X0, X1, il)) * X0, hi - lo)[0].numpy()
            out[a : min(n, a + step)] = y[a - lo : a - lo + min(step, n - a)]
    return out


class Banks2:
    """Per session: both microphones, human-free window starts, dog call centres, human-only events."""

    def __init__(self, sessions):
        self.items = []
        for s in sessions:
            x0, x1 = load_pair(s)
            ok = np.ones(len(x0), bool)
            for e in s.human:
                ok[max(0, int((e.start_s - HUMAN_MARGIN_S) * SR)) : int((e.end_s + HUMAN_MARGIN_S) * SR)] = False
            bad = np.concatenate([[0], np.cumsum(~ok)])
            starts = np.nonzero(bad[SEG:] - bad[:-SEG] == 0)[0] if len(x0) > SEG else np.array([], int)
            human = []
            for e in s.human:
                if any(min(d.end_s, e.end_s) > max(d.start_s, e.start_s) for d in s.dog):
                    continue
                a, b = max(0, int((e.start_s - 0.1) * SR)), min(len(x0), int((e.end_s + 0.1) * SR))
                if SEG > b - a > int(0.2 * SR):
                    human.append((a, b))
            if len(starts) and human:
                centres = np.array([int(((e.start_s + e.end_s) / 2) * SR) for e in s.dog], int)
                self.items.append({"name": s.name, "x0": x0, "x1": x1, "starts": starts, "centres": centres,
                                   "human": human, "ild": session_ild(x0, x1)})

    def example(self, rng: random.Random, p_no_human: float = 0.1):
        it = rng.choice(self.items)
        starts, centres = it["starts"], it["centres"]
        a = None
        if len(centres) and rng.random() < 0.8:
            c = int(rng.choice(list(centres)))
            ok = starts[(starts > c - SEG + SR // 2) & (starts < c - SR // 2)]
            if len(ok):
                a = int(ok[rng.randrange(len(ok))])
        if a is None:
            a = int(starts[rng.randrange(len(starts))])
        d = np.stack([it["x0"][a : a + SEG], it["x1"][a : a + SEG]]).copy()
        s = np.zeros((2, SEG), np.float32)
        if rng.random() >= p_no_human:
            for _ in range(1 if rng.random() < 0.6 else 2):
                ha, hb = rng.choice(it["human"])
                h = np.stack([it["x0"][ha:hb], it["x1"][ha:hb]]) * 10 ** (rng.uniform(-6, 6) / 20)
                o = rng.randrange(0, SEG - (hb - ha) + 1)
                s[:, o : o + hb - ha] += h
        g = 10 ** (rng.uniform(-10, 10) / 20)
        peak = np.abs(d + s).max() * g
        if peak > 0.99:
            g *= 0.99 / peak
        return (d * g).astype(np.float32), (s * g).astype(np.float32), it["ild"]


def batch_loss(model, ex, beta: float, c: float = 0.3):
    d = torch.from_numpy(np.stack([e[0] for e in ex]))       # (B, 2, N)
    s = torch.from_numpy(np.stack([e[1] for e in ex]))
    ild = torch.from_numpy(np.stack([e[2] for e in ex]))
    D0, S0 = stft(d[:, 0]), stft(s[:, 0])
    X1 = stft(d[:, 1] + s[:, 1])
    M = model(features2(D0 + S0, X1, ild))
    pd, ps = D0.abs() ** 2, S0.abs() ** 2
    target = pd / (pd + beta * ps + 1e-12)
    mix = (D0 + S0).abs()
    w = (mix / (mix.pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + 1e-8) + 1e-6) ** c
    return (w * (M - target) ** 2).mean(), M, D0, S0


def evaluate(model, val, beta: float) -> dict:
    model.eval()
    sl = sp = dk = dp = 0.0
    with torch.no_grad():
        for i in range(0, len(val), 16):
            _, M, D, S = batch_loss(model, val[i : i + 16], beta)
            sl += float(((M * S.abs()) ** 2).sum()); sp += float((S.abs() ** 2).sum())
            dk += float(((M * D.abs()) ** 2).sum()); dp += float((D.abs() ** 2).sum())
    model.train()
    return {"speech_left_db": round(10 * math.log10(sl / sp + 1e-12), 2), "dog_kept_db": round(10 * math.log10(dk / dp + 1e-12), 2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-group", choices=["odd", "even"], required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = random.Random(0)
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    train = Banks2([s for s in sessions if participant_group(s.name) == args.train_group])
    test = Banks2([s for s in sessions if participant_group(s.name) != args.train_group])
    vrng = random.Random(123)
    val = [test.example(vrng) for _ in range(160)]
    print(f"train sessions {len(train.items)}, test sessions {len(test.items)}", flush=True)
    model = DogExtractor2()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    name = f"dog_extractor2_{args.train_group}{args.tag}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        loss, *_ = batch_loss(model, [train.example(rng) for _ in range(args.batch)], args.beta)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            ev = evaluate(model, val, args.beta)
            log.append({"step": step, "loss": round(loss.item(), 5), **ev})
            torch.save({"model": model.state_dict(), "step": step}, MODEL_DIR / f"{name}_last.pt")
            print(f"step {step}: loss {loss.item():.4f} | held-out: {ev} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": model.state_dict(), "train_group": args.train_group, "beta": args.beta, "steps": args.steps},
               MODEL_DIR / f"{name}.pt")
    (MODEL_DIR / f"{name}.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=1))
    print(f"saved {MODEL_DIR / name}.pt")


if __name__ == "__main__":
    main()
