"""Step 27a: keep the dog when a person talks over it - a dog extractor.

Muting removes every dog sound that overlaps a human sound: 230 s of dog
sound (5.8% of all dog time; 223 of 2,055 calls, 178 of them pants) are lost
by design, about 40% of everything the standard setting loses. Step 14 tried
the other direction (estimate the SPEECH, subtract it): the speech estimate
removed only ~5 dB of far-field lab speech, so words stayed readable.

This model estimates the DOG instead, with privacy in the loss. It predicts a
time-frequency mask M (0..1) on the mixture's spectrogram X = D + S (dog +
background D, human sound S). Because the mask is applied to a sum, the
output splits exactly into M*D (the dog that is kept) and M*S (the speech that
leaks), and the loss weighs the two separately:

    loss = | |M*D|^c - |D|^c |^2  +  lambda * | |M*S|^c |^2      (c = 0.3, compressed magnitudes)

A large lambda means "when in doubt, remove": a bin is kept only where the dog
clearly dominates. Each example is normalized by its mixture level first (the
lab audio is quiet and its level varies by 20 dB; without this the quiet dog
sounds weigh almost nothing and the model learns to turn everything down).

Training data imitate the lab with the lab's own audio, from the training
participants only:

* dog + background: 4 s windows of the sessions with no annotated human sound
  within 0.5 s (80% centred on annotated dog calls, 20% anywhere: background);
* human: the annotated human events that have no dog on them (real voices,
  real room, real distance), at their natural level +-6 dB, and 30% LibriSpeech
  speech convolved with simulated room responses of the lab (more voices);
* 10% of the examples have no human sound (the mask must then keep everything).

Participants are split into two folds (odd / even numbers, as in Step 14): the
model trained on one fold is tested on the other, never on a dog or person it
heard.

Model (offline, non-causal, ~1M parameters): log-power spectrogram (16 kHz,
512-point STFT, 16 ms hop) -> 3 conv layers -> 2-layer bidirectional GRU ->
sigmoid mask over the 257 frequency bins.

Usage (from inside speech_anonymization/):
    python -m scripts.train_dog_extractor --train-group odd     # model for the even participants (~45 min)
    python -m scripts.train_dog_extractor --train-group even
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.signal import fftconvolve

from scripts.finetune_gtcrn import HUMAN_MARGIN_S, active_power, make_rirs, participant_group, speech_split
from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
SEG = 4 * SR
N_FFT, HOP = 512, 256
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "dog_extractor"


# ---------------------------------------------------------------- model

class DogExtractor(nn.Module):
    """Spectrogram -> mask (0..1) that keeps the dog and removes the human sound."""

    def __init__(self, n_freq: int = N_FFT // 2 + 1, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=(1, 2), padding=1), nn.GELU())
        f = n_freq
        for _ in range(3):
            f = (f + 1) // 2
        self.proj = nn.Linear(32 * f, 2 * hidden)
        self.rnn = nn.GRU(2 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden + n_freq, n_freq)

    def forward(self, logpow: torch.Tensor) -> torch.Tensor:
        """logpow: (B, T, F) normalized log power -> mask (B, T, F)."""
        B, T, F = logpow.shape
        z = self.conv(logpow[:, None])                         # (B, 32, T, F')
        z = z.permute(0, 2, 1, 3).reshape(B, T, -1)
        z, _ = self.rnn(torch.nn.functional.gelu(self.proj(z)))
        return torch.sigmoid(self.out(torch.cat([z, logpow], -1)))


def stft(x: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).transpose(-1, -2)  # (B,T,F)


def istft(X: torch.Tensor, length: int) -> torch.Tensor:
    return torch.istft(X.transpose(-1, -2), N_FFT, HOP, window=torch.hann_window(N_FFT), length=length)


def features(X: torch.Tensor) -> torch.Tensor:
    """Log power, normalized per example (the recording level varies a lot)."""
    lp = torch.log10(X.abs() ** 2 + 1e-10)
    return lp - lp.mean(dim=(1, 2), keepdim=True)


def extract(model: DogExtractor, x: np.ndarray, chunk_s: float = 20.0) -> np.ndarray:
    """Apply the extractor to a whole 16 kHz signal (in chunks with 2 s overlap)."""
    out = np.zeros_like(x)
    n, step, ov = len(x), int(chunk_s * SR), 2 * SR
    with torch.no_grad():
        for a in range(0, n, step):
            lo, hi = max(0, a - ov), min(n, a + step + ov)
            seg = torch.from_numpy(x[lo:hi].astype(np.float32))[None]
            X = stft(seg)
            y = istft(model(features(X)) * X, hi - lo)[0].numpy()
            out[a : min(n, a + step)] = y[a - lo : a - lo + min(step, n - a)]
    return out


# ---------------------------------------------------------------- data

def merged(intervals: list[tuple[float, float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class Banks:
    """Real lab audio of one participant group: dog/background windows and human-only events."""

    def __init__(self, sessions, librispeech: list[Path] | None, seed: int, tts: list[np.ndarray] | None = None,
                 p_libri: float = 0.3, p_tts: float = 0.0):
        self.dog_items, self.human = [], []
        for s in sessions:
            audio = load_audio(s.audio_path, SR).numpy().astype(np.float32)
            ok = np.ones(len(audio), bool)
            for e in s.human:
                ok[max(0, int((e.start_s - HUMAN_MARGIN_S) * SR)) : int((e.end_s + HUMAN_MARGIN_S) * SR)] = False
            bad = np.concatenate([[0], np.cumsum(~ok)])
            starts = np.nonzero(bad[SEG:] - bad[:-SEG] == 0)[0] if len(audio) > SEG else np.array([], int)
            if len(starts) == 0:
                continue
            dog_centres = [int(((e.start_s + e.end_s) / 2) * SR) for e in s.dog]
            self.dog_items.append((audio, starts, np.array(dog_centres, int)))
            for e in s.human:
                if any(min(d.end_s, e.end_s) > max(d.start_s, e.start_s) for d in s.dog):
                    continue                                   # only human sound without a dog on it
                a, b = max(0, int((e.start_s - 0.1) * SR)), min(len(audio), int((e.end_s + 0.1) * SR))
                if b - a > int(0.2 * SR):
                    self.human.append(audio[a:b].copy())
        self.libri = [load_audio(f, SR).numpy().astype(np.float32) for f in (librispeech or [])]
        self.tts = list(tts or [])                               # synthetic voices (Step 28, make_tts_speech)
        self.p_libri, self.p_tts = p_libri, (p_tts if self.tts else 0.0)
        self.rirs = make_rirs(40, seed) if (self.libri or self.tts) else []
        self.human_power = [active_power(h) for h in self.human]

    def dog_window(self, rng: random.Random) -> np.ndarray:
        audio, starts, centres = rng.choice(self.dog_items)
        if len(centres) and rng.random() < 0.8:
            c = int(rng.choice(list(centres)))
            ok = starts[(starts > c - SEG + SR // 2) & (starts < c - SR // 2)]
            if len(ok):
                a = int(ok[rng.randrange(len(ok))])
                return audio[a : a + SEG]
        a = int(starts[rng.randrange(len(starts))])
        return audio[a : a + SEG]

    def human_sound(self, rng: random.Random) -> np.ndarray:
        r = rng.random()
        if r < self.p_tts or (self.libri and r < self.p_tts + self.p_libri):
            dry = rng.choice(self.tts) if r < self.p_tts else rng.choice(self.libri)
            if len(dry) > 3 * SR:
                a = rng.randrange(len(dry) - 3 * SR)
                dry = dry[a : a + rng.randrange(SR // 2, 3 * SR)]
            s = fftconvolve(dry, rng.choice(self.rirs))[: len(dry) + SR // 4].astype(np.float32)
            target = rng.choice(self.human_power)                # level of a real lab voice
            p = active_power(s)
            return s * math.sqrt(target / p) if p > 0 else s
        return rng.choice(self.human).copy()


def make_example(rng: random.Random, banks: Banks, p_no_human: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(dog + background, human sound), both SEG samples; the mixture is their sum."""
    d = banks.dog_window(rng).copy()
    s = np.zeros(SEG, np.float32)
    if rng.random() >= p_no_human:
        for _ in range(1 if rng.random() < 0.6 else 2):
            h = banks.human_sound(rng)[:SEG] * 10 ** (rng.uniform(-6, 6) / 20)
            a = rng.randrange(0, SEG - len(h) + 1)
            s[a : a + len(h)] += h
    g = 10 ** (rng.uniform(-10, 10) / 20)                      # overall recording level
    peak = np.abs(d + s).max() * g
    if peak > 0.99:
        g *= 0.99 / peak
    return (d * g).astype(np.float32), (s * g).astype(np.float32)


# ---------------------------------------------------------------- training

LOSS = "signal"     # "signal": compressed dog error + lambda * leak; "mask": regress a privacy-weighted Wiener mask
BETA = 10.0         # mask target: |D|^2 / (|D|^2 + BETA |S|^2): a bin is kept only where the dog is ~10 dB above the speech


def losses(model: DogExtractor, d: torch.Tensor, s: torch.Tensor, c: float = 0.3):
    D, S = stft(d), stft(s)
    M = model(features(D + S))
    if LOSS == "mask":
        pd, ps = D.abs() ** 2, S.abs() ** 2
        target = pd / (pd + BETA * ps + 1e-12)
        mix = (D + S).abs()
        w = (mix / (mix.pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + 1e-8) + 1e-6) ** c   # louder bins count more
        err = w * (M - target) ** 2
        return err.mean(), (w * M * (ps / (pd + ps + 1e-12))).mean(), M, D, S
    # the lab recordings are very quiet and their level varies 20 dB: normalize each example by its
    # mixture level, otherwise quiet dog sounds weigh almost nothing against the leakage term
    ref = (D + S).abs().pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + 1e-8
    Dn, Sn = D.abs() / ref, S.abs() / ref
    eps = 1e-6                                                 # x**0.3 has an infinite slope at 0
    dog = ((M * Dn + eps) ** c - (Dn + eps) ** c) ** 2
    leak = (M * Sn + eps) ** (2 * c) - eps ** (2 * c)
    return dog.mean(), leak.mean(), M, D, S


def evaluate(model: DogExtractor, val: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """What matters on held-out mixtures: speech left (dB), dog kept (dB) and dog distortion."""
    model.eval()
    sl, dk, sp, dp, lsd = 0.0, 0.0, 0.0, 0.0, []
    with torch.no_grad():
        for i in range(0, len(val), 16):
            d = torch.from_numpy(np.stack([v[0] for v in val[i : i + 16]]))
            s = torch.from_numpy(np.stack([v[1] for v in val[i : i + 16]]))
            _, _, M, D, S = losses(model, d, s)
            sl += float(((M * S.abs()) ** 2).sum()); sp += float((S.abs() ** 2).sum())
            dk += float(((M * D.abs()) ** 2).sum()); dp += float((D.abs() ** 2).sum())
            act = D.abs() ** 2 > 1e-8
            lsd.append(float(((torch.log10((M * D.abs()) ** 2 + 1e-8) - torch.log10(D.abs() ** 2 + 1e-8)) ** 2)[act].mean().sqrt() * 10))
    model.train()
    return {"speech_left_db": round(10 * math.log10(sl / sp + 1e-12), 2),
            "dog_kept_db": round(10 * math.log10(dk / dp + 1e-12), 2),
            "dog_log_spectral_distance_db": round(float(np.mean(lsd)), 2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-group", choices=["odd", "even"], required=True)
    p.add_argument("--lam", type=float, default=4.0, help="weight of the speech leakage term")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--loss", choices=["signal", "mask"], default="mask")
    p.add_argument("--beta", type=float, default=10.0)
    args = p.parse_args()
    global LOSS, BETA
    LOSS, BETA = args.loss, args.beta
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = random.Random(0)
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    train_s = [s for s in sessions if participant_group(s.name) == args.train_group]
    test_s = [s for s in sessions if participant_group(s.name) != args.train_group]
    libri_train, libri_val = speech_split(PROJECT_ROOT / "data" / "librispeech_subset")
    train = Banks(train_s, libri_train, seed=1)
    test = Banks(test_s, libri_val, seed=2)
    vrng = random.Random(123)
    val = [make_example(vrng, test) for _ in range(160)]
    print(f"train: {len(train.dog_items)} sessions, {len(train.human)} human events; "
          f"test: {len(test.dog_items)} sessions, {len(test.human)} human events", flush=True)

    model = DogExtractor()
    print(f"{sum(q.numel() for q in model.parameters())} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    name = f"dog_extractor_{args.train_group}{args.tag}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        ex = [make_example(rng, train) for _ in range(args.batch)]
        d = torch.from_numpy(np.stack([e[0] for e in ex]))
        s = torch.from_numpy(np.stack([e[1] for e in ex]))
        ld, ll, *_ = losses(model, d, s)
        loss = ld if LOSS == "mask" else ld + args.lam * ll
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            torch.save({"model": model.state_dict(), "step": step}, MODEL_DIR / f"{name}_last.pt")
        if step % 250 == 0 or step == args.steps:
            ev = evaluate(model, val)
            log.append({"step": step, "loss_dog": round(ld.item(), 5), "loss_leak": round(ll.item(), 5), **ev})
            print(f"step {step}: dog {ld.item():.4f} leak {ll.item():.4f} | held-out: {ev} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": model.state_dict(), "train_group": args.train_group, "lam": args.lam, "steps": args.steps,
                "loss": LOSS, "beta": BETA},
               MODEL_DIR / f"{name}.pt")
    (MODEL_DIR / f"{name}.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=1))
    print(f"saved {MODEL_DIR / name}.pt")


if __name__ == "__main__":
    main()
