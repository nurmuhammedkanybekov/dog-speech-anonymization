"""Step 27d: a dog extractor that listens with Whisper too.

Steps 27a-c trained a mask network from the spectrogram (and a second
microphone). On mixtures of held-out participants it removed only 4-6 dB of
the speech: with ~150 real human events per participant group, the network
learns the training voices, not speech in general. The detector had the same
problem until Step 25 gave it Whisper's encoder, which has heard 680,000
hours of speech.

Here the mask network gets, next to the spectrogram, the Whisper small encoder
frames of the mixture (768 numbers every 20 ms): Whisper tells it WHEN and
WHAT is spoken, the spectrogram WHERE in frequency it is. The target is the
privacy-weighted Wiener mask of Step 27a (a bin is kept only where the dog is
about 10 dB above the speech).

Running Whisper for every random training mixture would be too slow on a CPU,
so a fixed set of mixtures is built once per participant group
(``--build``): 3,000 mixtures of 4 s (real dog/background windows + real
human-only events, 30% LibriSpeech in simulated lab rooms, as in Step 27a),
encoded six at a time in one 30 s Whisper window. At run time the session's
own Whisper frames (``outputs/whisper_enc``, Step 25a) are used.

Usage (from inside speech_anonymization/):
    python -m scripts.train_dog_extractor_w --build --group odd       # ~20 min, ~1.6 GB in outputs/extractor_data/
    python -m scripts.train_dog_extractor_w --build --group even
    python -m scripts.train_dog_extractor_w --train-group odd         # model for the even participants
    python -m scripts.train_dog_extractor_w --train-group even
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

from scripts.finetune_gtcrn import participant_group, speech_split
from scripts.train_dog_extractor import HOP, MODEL_DIR, N_FFT, SEG, SR, Banks, features, istft, make_example, stft
from vad_anonymization.elte_barks import load_sessions

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = Path(__file__).resolve().parent.parent / "outputs" / "extractor_data"
N_EX = 3000
PACK = 6                     # mixtures per 30 s Whisper window
GAP = SR // 2                # 0.5 s of silence between packed mixtures
W_FRAMES = SEG // 320        # 200 Whisper frames (20 ms) per 4 s mixture
BETA = 10.0


class DogExtractorW(nn.Module):
    """(log power spectrogram, Whisper frames) -> mask (0..1) keeping the dog."""

    def __init__(self, n_freq: int = N_FFT // 2 + 1, hidden: int = 128, w_dim: int = 768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=(1, 2), padding=1), nn.GELU())
        f = n_freq
        for _ in range(3):
            f = (f + 1) // 2
        self.proj = nn.Linear(32 * f, 2 * hidden)
        self.wproj = nn.Sequential(nn.LayerNorm(w_dim), nn.Linear(w_dim, hidden), nn.GELU())
        self.rnn = nn.GRU(3 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden + n_freq, n_freq)

    def forward(self, logpow: torch.Tensor, wfeat: torch.Tensor) -> torch.Tensor:
        """logpow (B, T, F); wfeat (B, T, 768) already on the STFT frame grid."""
        B, T, F = logpow.shape
        z = self.conv(logpow[:, None]).permute(0, 2, 1, 3).reshape(B, T, -1)
        z = torch.cat([torch.nn.functional.gelu(self.proj(z)), self.wproj(wfeat)], -1)
        z, _ = self.rnn(z)
        return torch.sigmoid(self.out(torch.cat([z, logpow], -1)))


class DogExtractorW2(nn.Module):
    """Step 28: the same inputs and output, with a frequency decoder (U-Net skips).

    DogExtractorW squeezes each spectrogram frame into one vector and predicts the 257 mask
    values with a single linear layer, so fine harmonic detail is lost: on held-out speech the
    mask stays near 0.25 even in bins where the speech clearly dominates. Here the GRU output is
    decoded back to full frequency resolution by transposed convolutions that see the encoder's
    feature maps at every scale."""

    def __init__(self, n_freq: int = N_FFT // 2 + 1, hidden: int = 128, w_dim: int = 768, ch: int = 16):
        super().__init__()
        self.enc = nn.ModuleList([nn.Conv2d(1, ch, 3, stride=(1, 2), padding=1),
                                  nn.Conv2d(ch, ch, 3, stride=(1, 2), padding=1),
                                  nn.Conv2d(ch, ch, 3, stride=(1, 2), padding=1)])
        fs = [n_freq]
        for _ in range(3):
            fs.append((fs[-1] + 1) // 2)                        # 257 -> 129 -> 65 -> 33
        self.fs, self.ch = fs, ch
        self.proj = nn.Linear(ch * fs[3], 2 * hidden)
        self.wproj = nn.Sequential(nn.LayerNorm(w_dim), nn.Linear(w_dim, hidden), nn.GELU())
        self.rnn = nn.GRU(3 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.dec_in = nn.Linear(2 * hidden, ch * fs[3])
        self.dec = nn.ModuleList([nn.ConvTranspose2d(2 * ch, ch, 3, stride=(1, 2), padding=1) for _ in range(3)])
        self.out = nn.Conv2d(ch + 1, 1, 3, padding=1)

    def forward(self, logpow: torch.Tensor, wfeat: torch.Tensor) -> torch.Tensor:
        B, T, F = logpow.shape
        h, skips = logpow[:, None], []
        for conv in self.enc:
            h = torch.nn.functional.gelu(conv(h))
            skips.append(h)                                     # (B, ch, T, f)
        z = torch.cat([torch.nn.functional.gelu(self.proj(h.permute(0, 2, 1, 3).reshape(B, T, -1))), self.wproj(wfeat)], -1)
        z, _ = self.rnn(z)
        u = torch.nn.functional.gelu(self.dec_in(z)).reshape(B, T, self.ch, self.fs[3]).permute(0, 2, 1, 3)
        for k, dec in enumerate(self.dec):
            u = torch.nn.functional.gelu(dec(torch.cat([u, skips[2 - k]], 1)))
        return torch.sigmoid(self.out(torch.cat([u, logpow[:, None]], 1)))[:, 0]


ARCHS = {"w": DogExtractorW, "w2": DogExtractorW2}


def load_w(path) -> nn.Module:
    """A saved Whisper-conditioned extractor of either architecture."""
    ck = torch.load(path, map_location="cpu")
    m = ARCHS[ck.get("arch", "w")]()
    m.load_state_dict(ck["model"])
    return m.eval()


def to_stft_grid(wf: torch.Tensor, n_frames: int, offset_s: float = 0.0) -> torch.Tensor:
    """Whisper frames (B, Tw, D), frame j centred at offset + (j + 0.5) * 20 ms -> nearest frame for each STFT frame."""
    t = torch.arange(n_frames) * (HOP / SR)                     # STFT frame centres (center=True)
    idx = torch.clamp(torch.round((t - offset_s) / 0.02 - 0.5).long(), 0, wf.shape[1] - 1)
    return wf[:, idx]


# ---------------------------------------------------------------- dataset

def load_tts(group: str) -> list[np.ndarray]:
    """Synthetic voices of one speaker half (``scripts.make_tts_speech``)."""
    z = np.load(Path(__file__).resolve().parent.parent / "outputs" / "tts_speech" / f"{group}.npz")
    return [z[k].astype(np.float32) for k in z.files if k.startswith("c")]


MIXES = {"": {}, "tts": dict(p_libri=0.2, p_tts=0.45), "clean": dict(p_libri=0.4, p_tts=0.6)}


def build(group: str, mix: str = "", n_ex: int = N_EX) -> None:
    """Mixtures of one participant group, saved to ``<group>[_<mix>]``. The human sound is

    * ``""``: 70% real lab human events, 30% LibriSpeech in simulated rooms (Step 27);
    * ``tts``: 35% real, 45% synthetic voices, 20% LibriSpeech (Step 28 training);
    * ``clean``: only synthetic voices and LibriSpeech in simulated rooms - no real human events, which also
      carry the room's background, so the speech left is measured on speech alone (Step 28 evaluation)."""
    from scripts.extract_whisper_encoder import WhisperEncoder

    out = DATA / (group + (f"_{mix}" if mix else ""))
    out.mkdir(parents=True, exist_ok=True)
    sessions = [s for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks") if participant_group(s.name) == group]
    libri_train, libri_val = speech_split(PROJECT_ROOT / "data" / "librispeech_subset")
    # the odd group's mixtures train the model for the even participants and vice versa; LibriSpeech speakers
    # are split the same way (training speakers for one group, held-out speakers for the other)
    extra = dict(tts=load_tts(group), **MIXES[mix]) if mix else {}
    banks = Banks(sessions, libri_train if group == "odd" else libri_val, seed=1 if group == "odd" else 2, **extra)
    rng = random.Random((7 if group == "odd" else 8) + {"": 0, "tts": 100, "clean": 200}[mix])
    d_mm = np.lib.format.open_memmap(out / "dog.npy", "w+", np.float16, (n_ex, SEG))
    s_mm = np.lib.format.open_memmap(out / "human.npy", "w+", np.float16, (n_ex, SEG))
    w_mm = np.lib.format.open_memmap(out / "whisper.npy", "w+", np.float16, (n_ex, W_FRAMES, 768))
    enc = WhisperEncoder(threads=2)
    t0 = time.time()
    for k0 in range(0, n_ex, PACK):
        ks = list(range(k0, min(n_ex, k0 + PACK)))
        window = np.zeros(30 * SR, np.float32)
        for j, k in enumerate(ks):
            d, s = make_example(rng, banks)
            d_mm[k], s_mm[k] = d.astype(np.float16), s.astype(np.float16)
            mix = d + s
            rms = float(np.sqrt((mix ** 2).mean())) + 1e-9
            a = GAP + j * (SEG + GAP)
            window[a : a + SEG] = mix * (0.03 / rms)                  # same level for every packed mixture
        f = enc(window)                                             # (1500, 768)
        for j, k in enumerate(ks):
            a = (GAP + j * (SEG + GAP)) // 320
            w_mm[k] = f[a : a + W_FRAMES].astype(np.float16)
        if k0 % 300 == 0:
            print(f"{out.name}: {k0 + len(ks)}/{n_ex} ({time.time() - t0:.0f}s)", flush=True)
    d_mm.flush(); s_mm.flush(); w_mm.flush()
    print(f"saved {out}")


# ---------------------------------------------------------------- training

class _Concat:
    """Index several memory-mapped arrays as one (rows)."""

    def __init__(self, arrays):
        self.arrays, self.offsets = arrays, np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = []
        for i in idx:
            j = int(np.searchsorted(self.offsets, i, side="right") - 1)
            out.append(self.arrays[j][i - self.offsets[j]])
        return np.stack(out)


def batch(data, idx):
    d = torch.from_numpy(data["dog"][idx].astype(np.float32))
    s = torch.from_numpy(data["human"][idx].astype(np.float32))
    w = torch.from_numpy(data["whisper"][idx].astype(np.float32))
    return d, s, w


LEAK_FLOOR = 1e-4   # -40 dB: below this the leak term stops pushing


def loss_fn(model, d, s, w, beta: float = BETA, c: float = 0.3, leak: float = 0.0):
    """Weighted squared error to the privacy-weighted Wiener mask; with ``leak`` > 0 (Step 28) plus
    ``leak`` x the speech left in dB (per mixture, floor -40 dB, divided by 40).

    The squared error hardly cares whether a speech bin keeps 10% or 3% of its amplitude
    (0.01 vs 0.001), so the mask settles around 0.2 in bins the speech dominates, which leaves the
    speech about 13 dB down - readable for a strong recognizer after level normalization. The dB
    term rewards every halving of the leaked speech energy equally."""
    D, S = stft(d), stft(s)
    X = D + S
    M = model(features(X), to_stft_grid(w, X.shape[1]))
    pd, ps = D.abs() ** 2, S.abs() ** 2
    target = pd / (pd + beta * ps + 1e-12)
    mix = X.abs()
    wt = (mix / (mix.pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + 1e-8) + 1e-6) ** c
    loss = (wt * (M - target) ** 2).mean()
    if leak > 0:
        sp = ps.sum((1, 2))
        has = sp > 1e-9
        if has.any():
            left = (M ** 2 * ps).sum((1, 2))[has] / sp[has]
            loss = loss + leak * (10 * torch.log10(left + LEAK_FLOOR)).mean() / 40
    return loss, M, D, S


def evaluate(model, data, beta: float, n: int = 240) -> dict:
    model.eval()
    sl = sp = dk = dp = 0.0
    msd, mdd = [], []
    with torch.no_grad():
        for i in range(0, n, 16):
            loss, M, D, S = loss_fn(model, *batch(data, np.arange(i, min(n, i + 16))), beta)
            pd, ps = D.abs() ** 2, S.abs() ** 2
            sl += float((M ** 2 * ps).sum()); sp += float(ps.sum()); dk += float((M ** 2 * pd).sum()); dp += float(pd.sum())
            msd.append(float(M[(ps > 10 * pd) & (ps > 1e-10)].mean())); mdd.append(float(M[(pd > 10 * ps) & (pd > 1e-10)].mean()))
    model.train()
    return {"speech_left_db": round(10 * math.log10(sl / sp + 1e-12), 2), "dog_kept_db": round(10 * math.log10(dk / dp + 1e-12), 2),
            "mask_speech_bins": round(float(np.nanmean(msd)), 3), "mask_dog_bins": round(float(np.nanmean(mdd)), 3)}


def extract_w(model: DogExtractorW, x: np.ndarray, wfeat: np.ndarray, chunk_s: float = 20.0,
              power: float = 1.0) -> np.ndarray:
    """Apply to a whole 16 kHz session; wfeat = its Whisper frames (outputs/whisper_enc, 20 ms).

    power > 1 makes the mask more conservative (M ** power: uncertain bins are removed more strongly)."""
    out = np.zeros_like(x)
    n, step, ov = len(x), int(chunk_s * SR), 2 * SR
    wf = torch.from_numpy(wfeat.astype(np.float32))[None]
    with torch.no_grad():
        for a in range(0, n, step):
            lo, hi = max(0, a - ov), min(n, a + step + ov)
            X = stft(torch.from_numpy(x[lo:hi])[None])
            g = to_stft_grid(wf, X.shape[1], offset_s=-lo / SR)
            y = istft(model(features(X), g) ** power * X, hi - lo)[0].numpy()
            out[a : min(n, a + step)] = y[a - lo : a - lo + min(step, n - a)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--group", choices=["odd", "even"])
    p.add_argument("--train-group", choices=["odd", "even", "all"], help="all: the deployable model (both groups)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--mix", choices=list(MIXES), default="", help="--build: human sound mix (Step 28: tts, clean)")
    p.add_argument("--n", type=int, default=N_EX, help="--build: number of mixtures")
    p.add_argument("--data", default="", help="training sets per group, e.g. ',_tts' = original + synthetic voices")
    p.add_argument("--init", default="", help="start from this model file (fine-tuning)")
    p.add_argument("--arch", choices=list(ARCHS), default="w", help="w2: with a frequency decoder (Step 28)")
    p.add_argument("--c", type=float, default=0.3, help="loss weight exponent on the mixture magnitude")
    p.add_argument("--resume", action="store_true", help="continue from <name>_last.pt (optimizer state restarts)")
    p.add_argument("--leak", type=float, default=0.0, help="weight of the speech-left-in-dB loss term (Step 28)")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.build:
        return build(args.group, args.mix, args.n)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    keys = ("dog", "human", "whisper")
    suffixes = args.data.split(",")
    groups = ("odd", "even") if args.train_group == "all" else (args.train_group,)
    parts = [{k: np.load(DATA / (g + sfx) / f"{k}.npy", mmap_mode="r") for k in keys} for g in groups for sfx in suffixes]
    train = parts[0] if len(parts) == 1 else {k: _Concat([p_[k] for p_ in parts]) for k in keys}
    # held-out log: the other group's original mixtures (real lab voices); in-sample for "all"
    other = {"odd": "even", "even": "odd", "all": "odd"}[args.train_group]
    test = {k: np.load(DATA / other / f"{k}.npy", mmap_mode="r") for k in keys}
    n_train = len(train["dog"])
    model = ARCHS[args.arch]()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    name = f"dog_extractor_w_{args.train_group}{args.tag}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log, t0, start = [], time.time(), 1
    if args.resume:                       # continue a run that was stopped: its _last checkpoint, same schedule
        ck = torch.load(MODEL_DIR / f"{name}_last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        start = ck["step"] + 1
        for _ in range(ck["step"]):
            opt.step()                    # no gradients yet: only advances the schedule
            sched.step()
        rng = np.random.default_rng(ck["step"])
        print(f"resumed at step {start}", flush=True)
    for step in range(start, args.steps + 1):
        idx = np.sort(rng.choice(n_train, args.batch, replace=False))
        d, s, w = batch(train, idx)
        if rng.random() < 0.5:                                   # level augmentation (the features are level-normalized)
            g = float(10 ** (rng.uniform(-10, 10) / 20))
            d, s = d * g, s * g
        loss, *_ = loss_fn(model, d, s, w, args.beta, args.c, args.leak)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            ev = evaluate(model, test, args.beta)
            log.append({"step": step, "loss": round(loss.item(), 5), **ev})
            torch.save({"model": model.state_dict(), "step": step, "arch": args.arch}, MODEL_DIR / f"{name}_last.pt")
            print(f"step {step}: loss {loss.item():.4f} | held-out: {ev} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": model.state_dict(), "train_group": args.train_group, "beta": args.beta, "steps": args.steps,
                "arch": args.arch},
               MODEL_DIR / f"{name}.pt")
    (MODEL_DIR / f"{name}.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=1))
    print(f"saved {MODEL_DIR / name}.pt")


if __name__ == "__main__":
    main()
