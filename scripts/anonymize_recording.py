"""Steps 18b and 25d: anonymize a new recording - mute the human sounds, keep everything else.

The recommended pipeline as one command, for recordings the models have never
seen. It needs no annotations. Since Step 25 the default model
(``models/human_detector_v3.npz``, Step 26) listens with two frozen networks:

    Whisper small encoder (speech) + EfficientAT mn40 (527 sound classes)
    -> trained logistic regression -> 5-frame smoothing
    -> threshold for the chosen strictness -> mute the flagged time (+-0.12 s)

The muting is applied to the ORIGINAL file (its own sample rate and all its
channels), so the dog sounds that are kept are untouched. Next to the output
it writes ``<output>.segments.csv`` (what was muted, in seconds).

``--keep-dog safe|more`` (Step 27): inside the muted time, where a dog router
hears a dog call, the output of a dog extractor (human sound removed) is kept
instead of silence. ``safe`` was exactly as private as muting for two speech
recognizers on unseen participants; ``more`` keeps more dog. Off by default
until a human listening check confirms it.

``--keep-dog all`` (Step 28): nothing is muted any more - the extractor's output
fills all the flagged time, and where the router hears no dog, steady noise with
the room's own sound 10 dB above it covers what is left of the voice
(``--room-tone``; ``--room-tone 10`` alone puts room tone into plain muted gaps).

Strictness levels (thresholds measured on unseen participants, README Step 25):
``light``, ``standard`` (default) and ``strict``; stricter = fewer readable words, more dog time lost.

Other models (``--model``): ``models/human_detector_eat40.npz`` (Step 18, EfficientAT only),
``models/human_detector_eat40_aug2.npz`` / ``_aug.npz`` (Step 21, for recordings unlike ELTE's).

Setup (once): ``git clone https://github.com/fschmid56/EfficientAT models/EfficientAT``,
``python -m scripts.download_models --asr`` (the Whisper small encoder) and
``python -m scripts.train_final_detector`` (or copy ``models/human_detector_v3.npz``).

Usage (from inside speech_anonymization/):
    python -m scripts.anonymize_recording input.wav anonymized.wav
    python -m scripts.anonymize_recording input.wav anonymized.wav --strictness strict
    python -m scripts.anonymize_recording input.wav anonymized.wav --keep-dog all
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import uniform_filter1d

from scripts.extract_efficientat import load_model, embed
from scripts.train_final_detector import MODEL_V3

SR = 16_000
WIN_S, HOP_S, PAD_S = 0.96, 0.24, 0.12
EAT_DIR = Path(__file__).resolve().parent.parent / "models" / "EfficientAT"
ROOM_TONE_ALL_DB = 10.0     # --keep-dog all: room tone this many dB above the room's own (Step 28)


_WHISPER: list = []


def whisper_encoder():
    """The Whisper small encoder, loaded once (Step 25 models only)."""
    if not _WHISPER:
        from scripts.extract_whisper_encoder import WhisperEncoder

        _WHISPER.append(WhisperEncoder())
    return _WHISPER[0]


def human_probability(mono16: torch.Tensor, detector, eat, cache: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame probability of a human sound and the frame centres (s), on the Step 9 grid.

    ``cache`` (a dict) receives the frame features and the 20 ms Whisper frames, so the
    dog router and the dog extractor (Step 27) can reuse them."""
    total = len(mono16) / SR
    n = max(1, int((total - WIN_S) // HOP_S) + 1)
    centres = WIN_S / 2 + HOP_S * np.arange(n)
    _, logits = embed(mono16, *eat[:2], n)
    if str(detector.get("features", "mn40_as")) == "whisper_small+mn40_as":
        from scripts.extract_whisper_encoder import encode_file
        from scripts.train_whisper_detector import pool_frames

        feats, times = encode_file(mono16.numpy().astype(np.float32), whisper_encoder())
        logits = np.hstack([pool_frames(feats, times, centres), logits])
        if cache is not None:
            cache.update(X=logits, whisper=feats)
    z = (logits - detector["mean"]) / detector["scale"] @ detector["coef"] + detector["intercept"]
    prob = uniform_filter1d(1 / (1 + np.exp(-z)), int(detector["smooth"]), mode="nearest")
    return prob, centres


def linear_probability(X: np.ndarray, model) -> np.ndarray:
    z = (X - model["mean"]) / model["scale"] @ model["coef"] + model["intercept"]
    return uniform_filter1d(1 / (1 + np.exp(-z)), int(model["smooth"]), mode="nearest")


def keep_dog_signal(mono16: torch.Tensor, cache: dict, centres: np.ndarray, level: str,
                    mask_power: float = 3.0) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Step 27: the dog extractor's output (16 kHz) and the time where the router hears a dog."""
    from scripts.train_dog_extractor import MODEL_DIR
    from scripts.train_dog_extractor_w import DogExtractorW, extract_w
    from scripts.train_dog_router import ROUTER_PATH

    router = dict(np.load(ROUTER_PATH))
    rlevel = "more" if level == "all" else level         # "all" (Step 28): the router decides where room tone goes
    th = float(router["thresholds"][[str(v) for v in router["levels"]].index(rlevel)])
    rp = linear_probability(cache["X"], router)
    routed = [(c - HOP_S / 2, c + HOP_S / 2) for c in centres[rp >= th]]
    model = DogExtractorW()
    model.load_state_dict(torch.load(MODEL_DIR / "dog_extractor_w_all.pt", map_location="cpu")["model"])
    model.eval()
    y = extract_w(model, mono16.numpy().astype(np.float32), cache["whisper"], power=mask_power)
    return y, routed


def time_mask(segs, n: int, sr: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in segs:
        m[max(0, int(a * sr)) : min(n, int(np.ceil(b * sr)))] = True
    return m


def flagged_segments(prob: np.ndarray, centres: np.ndarray, threshold: float, total_s: float) -> list[tuple[float, float]]:
    segs: list[list[float]] = []
    for c in centres[prob >= threshold]:
        a, b = max(0.0, c - HOP_S / 2 - PAD_S), min(total_s, c + HOP_S / 2 + PAD_S)
        if segs and a <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    return [(a, b) for a, b in segs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--strictness", default="standard", choices=["light", "standard", "strict"])
    p.add_argument("--model", type=Path, default=MODEL_V3)
    p.add_argument("--keep-dog", default="off", choices=["off", "safe", "more", "all"],
                   help="Step 27: inside the muted time, keep the dog where the router hears one (dog extractor). "
                        "safe: as private as muting in our tests; more: keeps more dog, slightly more speech; "
                        "all (Step 28): the extractor's output in all the muted time, with room tone 10 dB above the "
                        "room's own where the router hears no dog (it covers what is left of the voice)")
    p.add_argument("--room-tone", default="auto",
                   help="Step 28: steady noise with the room's own sound in the muted time, this many dB above it "
                        "(e.g. 10), or 'off'. auto: 10 with --keep-dog all, off otherwise")
    args = p.parse_args()
    torch.set_num_threads(2)
    det = dict(np.load(args.model))
    levels = [str(x) for x in det["levels"]]
    threshold = float(det["thresholds"][levels.index(args.strictness)])

    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)       # (samples, channels), original rate
    import torchaudio

    mono = torch.from_numpy(audio.mean(1))
    mono16 = torchaudio.functional.resample(mono, sr, SR) if sr != SR else mono
    eat = load_model(EAT_DIR, str(det["efficientat_model"]))
    cache: dict = {}
    prob, centres = human_probability(mono16, det, eat, cache)
    total = audio.shape[0] / sr
    segs = flagged_segments(prob, centres, threshold, total)
    out = audio.copy()
    for a, b in segs:
        out[int(a * sr) : int(np.ceil(b * sr))] = 0.0
    kept = 0.0
    room_db = (ROOM_TONE_ALL_DB if args.keep_dog == "all" else None) if args.room_tone == "auto" else \
        (None if args.room_tone == "off" else float(args.room_tone))
    tone_gate = None
    if args.keep_dog != "off":
        if "whisper" not in cache:
            raise SystemExit("--keep-dog needs a Whisper-encoder detector (models/human_detector_v3.npz)")
        y16, routed = keep_dog_signal(mono16, cache, centres, args.keep_dog)
        y = torchaudio.functional.resample(torch.from_numpy(y16), SR, sr).numpy() if sr != SR else y16
        y = np.pad(y, (0, max(0, len(out) - len(y))))[: len(out)]
        flagged = time_mask(segs, len(out), sr)
        keep = flagged if args.keep_dog == "all" else flagged & time_mask(routed, len(out), sr)
        if args.keep_dog == "all":
            tone_gate = ~time_mask(routed, len(out), sr)        # room tone only where the router hears no dog
        ramp = int(0.01 * sr)                                   # 10 ms fades, only inside the muted time
        g = np.clip(np.convolve(keep.astype(np.float32), np.ones(ramp, np.float32) / ramp, mode="same"), 0, 1) * flagged
        out += (g * y)[:, None].astype(np.float32)
        kept = float(keep.sum()) / sr
    if room_db is not None and segs:
        from vad_anonymization.comfort_noise import room_tone

        x16 = mono16.numpy().astype(np.float32)
        n16 = room_tone(x16, time_mask(segs, len(x16), SR), room_db)
        nz = torchaudio.functional.resample(torch.from_numpy(n16), SR, sr).numpy() if sr != SR else n16
        nz = np.pad(nz, (0, max(0, len(out) - len(nz))))[: len(out)]
        flagged = time_mask(segs, len(out), sr)
        gate = flagged if tone_gate is None else flagged & tone_gate
        ramp = int(0.01 * sr)
        g = np.clip(np.convolve(gate.astype(np.float32), np.ones(ramp, np.float32) / ramp, mode="same"), 0, 1) * flagged
        out += (g * nz)[:, None].astype(np.float32)
    info = sf.info(args.input)
    sf.write(args.output, out, sr, subtype=info.subtype if info.subtype in ("PCM_16", "PCM_24", "PCM_32", "FLOAT") else "PCM_16")
    with open(args.output.with_suffix(args.output.suffix + ".segments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s"])
        w.writerows([[round(a, 3), round(b, 3)] for a, b in segs])
    muted = sum(b - a for a, b in segs)
    print(f"{args.input.name}: {len(segs)} segments muted, {muted:.1f} s of {total:.1f} s ({100 * muted / total:.1f}%), "
          f"strictness '{args.strictness}'" + (f", dog kept in {kept:.1f} s of it" if args.keep_dog != "off" else "")
          + (f", room tone +{room_db:g} dB" if room_db is not None else "")
          + f" -> {args.output}")


if __name__ == "__main__":
    main()
