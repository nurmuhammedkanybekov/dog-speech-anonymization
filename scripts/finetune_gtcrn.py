"""Step 14: fine-tune the speech remover for far-field speech in the lab.

Why: on the real recordings the pretrained speech-enhancement models remove
only 2-4 dB of the human sounds (Step 6), against 26-35 dB on clean synthetic
mixtures - a domain shift. In the lab the person is far from the microphone,
quiet, and reverberant, which is not what DNS/VCTK training data looks like.

What: GTCRN (48k parameters, authors' DNS3 weights as the starting point) is
fine-tuned on mixtures that imitate the lab:

* speech: LibriSpeech, convolved with simulated room impulse responses of a
  5.46 m x 4.41 m room (the BARKS audio lab) with random RT60, speaker and
  microphone positions (pyroomacoustics), so it arrives far-field and
  reverberant;
* background: REAL audio from the ELTE sessions (dog panting, growls, room
  noise) at its real, very quiet absolute level, taken only from the
  training participants and away from annotated human sounds;
* speech-to-background ratio drawn from -10..+15 dB, and 15% of the examples
  contain no speech at all (the model must then output silence, i.e. leave
  the dog alone).

The target is the full reverberant speech at the microphone (all of it should
be removed). Participants are split into two folds (odd / even numbers) so
the model is always evaluated on dogs, people and sessions it never heard
(scripts/eval_finetuned_gtcrn.py).

Usage (from inside speech_anonymization/):
    python -m scripts.finetune_gtcrn --train-group odd    # model for testing on even participants
    python -m scripts.finetune_gtcrn --train-group even
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
from scipy.signal import fftconvolve

from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions
from vad_anonymization.enhancers import GTCRNTorchEnhancer

SR = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gtcrn"
ROOM_XY = (5.46, 4.41)
HUMAN_MARGIN_S = 0.5


def participant_group(session_name: str) -> str:
    return "odd" if int(session_name[2:4]) % 2 else "even"


def speech_split(speech_dir: Path, n_val_speakers: int = 8) -> tuple[list[Path], list[Path]]:
    files = sorted(speech_dir.glob("*.flac"))
    speakers = sorted({f.name.split("-")[0] for f in files})
    val = set(speakers[-n_val_speakers:])
    return [f for f in files if f.name.split("-")[0] not in val], [f for f in files if f.name.split("-")[0] in val]


def make_rirs(n: int, seed: int) -> list[np.ndarray]:
    """Room impulse responses of a lab-sized shoebox room, random geometry and RT60."""
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed)
    rirs = []
    while len(rirs) < n:
        h = rng.uniform(2.7, 3.2)
        dims = [ROOM_XY[0], ROOM_XY[1], h]
        rt60 = rng.uniform(0.15, 0.6)
        absorption, max_order = pra.inverse_sabine(rt60, dims)
        mic = [rng.uniform(0.3, dims[0] - 0.3), rng.uniform(0.3, dims[1] - 0.3), rng.uniform(0.2, 1.5)]
        dist, ang = rng.uniform(0.8, 4.0), rng.uniform(0, 2 * math.pi)
        src = [mic[0] + dist * math.cos(ang), mic[1] + dist * math.sin(ang), rng.uniform(0.9, 1.8)]
        if not (0.2 < src[0] < dims[0] - 0.2 and 0.2 < src[1] < dims[1] - 0.2):
            continue
        room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(absorption), max_order=min(max_order, 30))
        room.add_source(src)
        room.add_microphone(mic)
        room.compute_rir()
        rir = np.asarray(room.rir[0][0], dtype=np.float32)[: int(0.8 * SR)]
        rirs.append(rir / (np.abs(rir).max() + 1e-9))
    return rirs


class NoiseBank:
    """Random windows of real session audio that stay away from annotated human sounds."""

    def __init__(self, sessions, seg: int) -> None:
        self.items = []
        for s in sessions:
            audio = load_audio(s.audio_path, SR).numpy().astype(np.float32)
            ok = np.ones(len(audio), dtype=bool)
            for e in s.human:
                ok[max(0, int((e.start_s - HUMAN_MARGIN_S) * SR)) : int((e.end_s + HUMAN_MARGIN_S) * SR)] = False
            # valid window starts: the whole window is human-free (cumulative-sum test)
            bad = np.concatenate([[0], np.cumsum(~ok)])
            starts = np.nonzero(bad[seg:] - bad[:-seg] == 0)[0] if len(audio) > seg else np.array([], dtype=int)
            if len(starts):
                self.items.append((audio, starts))
        self.seg = seg

    def sample(self, rng: random.Random) -> np.ndarray:
        audio, starts = rng.choice(self.items)
        a = int(starts[rng.randrange(len(starts))])
        return audio[a : a + self.seg]


class SpeechBank:
    def __init__(self, files: list[Path], seg: int) -> None:
        self.clips = [load_audio(f, SR).numpy().astype(np.float32) for f in files]
        self.seg = seg

    def sample(self, rng: random.Random) -> np.ndarray:
        x = rng.choice(self.clips)
        if len(x) <= self.seg:
            return np.pad(x, (0, self.seg - len(x)))
        a = rng.randrange(len(x) - self.seg)
        return x[a : a + self.seg]


def active_power(x: np.ndarray, frame: int = 320, floor_db: float = 30.0) -> float:
    n = len(x) // frame
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(1)
    if e.max() <= 0:
        return 0.0
    return float(e[e >= e.max() * 10 ** (-floor_db / 10)].mean())


def make_example(rng: random.Random, speech: SpeechBank, noise: NoiseBank, rirs: list[np.ndarray],
                 p_no_speech: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    n = noise.sample(rng)
    seg = len(n)
    if rng.random() < p_no_speech:
        s = np.zeros(seg, dtype=np.float32)
    else:
        dry = speech.sample(rng)
        s = fftconvolve(dry, rng.choice(rirs))[:seg].astype(np.float32)
        ps, pn = active_power(s), float((n ** 2).mean()) + 1e-12
        snr = rng.uniform(-10, 15)
        if ps > 0:
            s *= math.sqrt(pn * 10 ** (snr / 10) / ps)
    x = n + s
    if rng.random() < 0.5:
        g = 10 ** (rng.uniform(-6, 6) / 20)
        x, s = x * g, s * g
    peak = np.abs(x).max()
    if peak > 0.99:
        x, s = x * 0.99 / peak, s * 0.99 / peak
    return x.astype(np.float32), s.astype(np.float32)


def freeze_batchnorm(model: torch.nn.Module) -> None:
    """Keep BatchNorm running statistics fixed (small batches of very quiet audio would corrupt them)."""
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()


def hybrid_loss(est: torch.Tensor, ref: torch.Tensor, est_wave: torch.Tensor, ref_wave: torch.Tensor) -> torch.Tensor:
    """GTCRN-style loss on power-compressed spectra (+ a small SI-SNR term when speech is present)."""
    def compress(z):
        mag = torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + 1e-12)
        return mag ** 0.3, z / mag[..., None] * (mag[..., None] ** 0.3)

    em, ec = compress(est)
    rm, rc = compress(ref)
    loss = 0.3 * ((ec - rc) ** 2).mean() + 0.7 * ((em - rm) ** 2).mean()
    has = ref_wave.pow(2).sum(-1) > 0
    if has.any():
        e, r = est_wave[has], ref_wave[has]
        r0 = r - r.mean(-1, keepdim=True)
        e0 = e - e.mean(-1, keepdim=True)
        proj = (e0 * r0).sum(-1, keepdim=True) / (r0.pow(2).sum(-1, keepdim=True) + 1e-12) * r0
        si = 10 * torch.log10(proj.pow(2).sum(-1) / ((e0 - proj).pow(2).sum(-1) + 1e-12) + 1e-12)
        loss = loss - 1e-4 * si.mean()
    return loss


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-group", choices=["odd", "even"], required=True)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--speech-dir", type=Path, default=PROJECT_ROOT / "data" / "librispeech_subset")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seg-s", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-rirs", type=int, default=200)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    seg = int(args.seg_s * SR)

    sessions = [s for s in load_sessions(args.data_root) if participant_group(s.name) == args.train_group]
    train_files, val_files = speech_split(args.speech_dir)
    t0 = time.time()
    noise = NoiseBank(sessions, seg)
    speech, speech_val = SpeechBank(train_files, seg), SpeechBank(val_files, seg)
    rirs, rirs_val = make_rirs(args.n_rirs, args.seed), make_rirs(40, args.seed + 1000)
    vrng = random.Random(12345)
    val = [make_example(vrng, speech_val, noise, rirs_val) for _ in range(48)]
    print(f"{len(sessions)} training sessions, {len(train_files)} train / {len(val_files)} val speech clips, "
          f"{len(rirs)} RIRs ({time.time() - t0:.0f}s setup)", flush=True)

    enh = GTCRNTorchEnhancer()
    model = enh.model
    model.train()
    freeze_batchnorm(model)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr)

    def batch_loss(examples):
        x = torch.from_numpy(np.stack([e[0] for e in examples]))
        s = torch.from_numpy(np.stack([e[1] for e in examples]))
        X, S = enh.spec(x), enh.spec(s)
        E = model(X)
        return hybrid_loss(E, S, enh.wave(E, seg), s)

    def evaluate():
        """Validation loss, plus what matters: speech left after subtraction and background change (dB)."""
        model.eval()
        with torch.no_grad():
            v = float(np.mean([batch_loss(val[i : i + 8]).item() for i in range(0, len(val), 8)]))
            left, bg = [], []
            for x, s in val:
                if not s.any():
                    continue
                y = x - enh.wave(model(enh.spec(torch.from_numpy(x)[None])), seg)[0].numpy()
                left.append(10 * np.log10(abs(float(np.dot(y, s))) / float(np.dot(s, s)) + 1e-12))
                n = x - s
                bg.append(10 * np.log10(float(((y - s) ** 2).mean()) / (float((n ** 2).mean()) + 1e-20) + 1e-20))
        model.train()
        freeze_batchnorm(model)
        return v, float(np.median(left)), float(np.median(bg))

    out = MODEL_DIR / f"gtcrn_ft_{args.train_group}.pt"
    best, left, bg = evaluate()
    log = [{"step": 0, "val": best, "speech_left_db": left, "background_change_db": bg}]
    print(f"step 0: val {best:.5f} speech left {left:.1f} dB, background {bg:+.2f} dB (pretrained)", flush=True)
    torch.save({"model": model.state_dict(), "step": 0, "val": best}, out)
    (MODEL_DIR / "steps").mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        loss = batch_loss([make_example(rng, speech, noise, rirs) for _ in range(args.batch)])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 250 == 0 or step == args.steps:
            v, left, bg = evaluate()
            log.append({"step": step, "train": round(loss.item(), 6), "val": v, "speech_left_db": left, "background_change_db": bg})
            torch.save({"model": model.state_dict(), "step": step, "val": v}, MODEL_DIR / "steps" / f"gtcrn_ft_{args.train_group}_{step}.pt")
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "step": step, "val": v}, out)
            print(f"step {step}: train {loss.item():.5f} val {v:.5f} speech left {left:.1f} dB, background {bg:+.2f} dB "
                  f"best {best:.5f} ({time.time() - t0:.0f}s)", flush=True)
    (MODEL_DIR / f"gtcrn_ft_{args.train_group}.json").write_text(json.dumps(
        {"train_group": args.train_group, "steps": args.steps, "batch": args.batch, "lr": args.lr,
         "train_sessions": [s.name for s in sessions], "best_val": best, "log": log}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
