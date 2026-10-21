"""Step 20: does the ELTE-trained anonymizer work on recordings it was not made for?

Everything so far was measured on the ELTE play sessions, each person tested
on a model that never saw them, but always the same lab, microphones,
language and kind of play. This step builds scenes that differ in all of
these and runs the deployable command (Step 18) on them, unchanged:

* speech: LibriSpeech, read English, 40 speakers (the participants spoke
  Hungarian to their dogs);
* dogs: 40 ESC-50 "dog" clips (other dogs, other microphones, mostly
  outdoors);
* rooms: simulated shoeboxes from 3 x 3 m to 9 x 7 m, RT60 0.2-0.8 s, the
  source 0.5-4 m from the microphone;
* background: pink noise, 15-35 dB below the sources.

Each 30 s scene has 3 utterances and 4 barks at random times (some
overlap). The ground truth comes from the dry sources, so speech and dog
time are exactly known. Compared, on the same scenes:

* ``eat40``  - the deployable detector (ELTE-trained mn40 + LR), thresholds as
  set on ELTE for light / standard / strict;
* ``silero`` - Silero VAD + mute, the generic speech detector (Step 1);
* ``oracle`` - mute the true speech time (upper bound).

Privacy: Whisper (multilingual small) on each utterance (+-0.3 s), scored
against the LibriSpeech reference text: share of utterances with 2+ words
recovered (correct words, not just any words), and share of the reference
words recovered. Dog: share of dog-only time muted.

Usage (from inside speech_anonymization/):
    python -m scripts.eval_out_of_domain --worker 0 --n-workers 2      # and --worker 1
    python -m scripts.eval_out_of_domain --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

from scripts.anonymize_recording import EAT_DIR, flagged_segments, human_probability
from scripts.run_asr_real import CONTEXT_S, loud
from scripts.train_final_detector import MODEL_PATH
from vad_anonymization.asr import WhisperASR, normalize_text, words_recovered
from vad_anonymization.audio_io import load_audio
from vad_anonymization.detection import SpeechDetector

SR = 16_000
SCENE_S = 30.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs" / "out_of_domain"


def rir(rng: np.random.Generator) -> np.ndarray:
    import pyroomacoustics as pra

    while True:
        dims = [rng.uniform(3, 9), rng.uniform(3, 7), rng.uniform(2.5, 3.5)]
        absorption, max_order = pra.inverse_sabine(rng.uniform(0.2, 0.8), dims)
        mic = [rng.uniform(0.3, dims[0] - 0.3), rng.uniform(0.3, dims[1] - 0.3), rng.uniform(0.5, 2.0)]
        d, a = rng.uniform(0.5, 4.0), rng.uniform(0, 2 * math.pi)
        src = [mic[0] + d * math.cos(a), mic[1] + d * math.sin(a), rng.uniform(0.3, 1.8)]
        if 0.2 < src[0] < dims[0] - 0.2 and 0.2 < src[1] < dims[1] - 0.2:
            break
    room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(absorption), max_order=min(max_order, 30))
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    h = np.asarray(room.rir[0][0], np.float32)[: int(1.0 * SR)]
    return h / (np.abs(h).max() + 1e-9)


def active(x: np.ndarray, frame: int = 160, floor_db: float = 35.0) -> np.ndarray:
    """10 ms frames where a dry source is active (within floor_db of its loudest frame)."""
    n = len(x) // frame
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12
    return e >= e.max() * 10 ** (-floor_db / 10)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    X[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(X, n)
    return (x / x.std()).astype(np.float32)


def make_scene(k: int, speech_files, transcripts, dog_files, n_speech: int = 3, n_dogs: int = 4):
    rng = np.random.default_rng(1000 + k)
    n = int(SCENE_S * SR)
    mix = np.zeros(n, np.float32)
    sp_act = np.zeros(n // 160, bool)
    dog_act = np.zeros(n // 160, bool)
    utts = []
    level = 0.05
    for j, f in enumerate(rng.choice(speech_files, n_speech, replace=False) if n_speech else []):
        dry = load_audio(f, SR).numpy()[: int(8 * SR)]
        t0 = int(rng.uniform(j * 10, j * 10 + 10 - len(dry) / SR - 0.2) * SR) if len(dry) < 9.5 * SR else j * 10 * SR
        wet = fftconvolve(dry, rir(rng))[: len(dry) + SR // 2]
        g = level * 10 ** (rng.uniform(-6, 6) / 20) / (np.sqrt((dry ** 2).mean()) + 1e-9)
        a = max(0, min(t0, n - len(wet)))
        mix[a : a + len(wet)] += g * wet[: n - a]
        act = active(np.pad(dry, (0, len(wet) - len(dry))))
        sp_act[a // 160 : a // 160 + len(act)] |= act[: len(sp_act) - a // 160]
        utts.append({"file": Path(f).stem, "start_s": a / SR, "end_s": (a + len(dry)) / SR, "ref": transcripts[Path(f).stem]})
    for f in rng.choice(dog_files, n_dogs, replace=False) if n_dogs else []:
        dry = load_audio(f, SR).numpy()
        wet = fftconvolve(dry, rir(rng))[: len(dry) + SR // 2]
        on = np.zeros(len(dry), bool)
        on[: len(active(dry)) * 160] = active(dry).repeat(160)
        g = level * 10 ** (rng.uniform(-3, 9) / 20) / (np.sqrt((dry[on] ** 2).mean()) + 1e-9)
        a = int(rng.uniform(0, SCENE_S - len(wet) / SR) * SR)
        mix[a : a + len(wet)] += g * wet
        act = active(np.pad(dry, (0, len(wet) - len(dry))))
        dog_act[a // 160 : a // 160 + len(act)] |= act[: len(dog_act) - a // 160]
    noise_db = rng.uniform(-35, -15)
    mix += pink(n, rng) * level * 10 ** (noise_db / 20)
    mix *= 10 ** (rng.uniform(-30, 0) / 20)                       # overall recording level varies too
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix, sp_act, dog_act, utts


def mask_from(segs, n_frames: int) -> np.ndarray:
    m = np.zeros(n_frames, bool)
    for a, b in segs:
        m[int(a * 100) : int(math.ceil(b * 100))] = True
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-scenes", type=int, default=40)
    p.add_argument("--scene-type", choices=["mixed", "speech", "dog"], default="mixed",
                   help="mixed: 3 utterances + 4 barks; speech: 3 utterances only; dog: 5 dog clips only")
    p.add_argument("--heldout-dogs", action="store_true", help="use only the ESC-50 dogs and LibriSpeech speakers not used in Step 20c/21a training")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    global OUT
    OUT = OUT / (args.scene_type + ("_heldout_dogs_speakers" if args.heldout_dogs else ""))
    counts = {"mixed": (3, 4), "speech": (3, 0), "dog": (0, 5)}[args.scene_type]
    (OUT / "scenes").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if not args.summarize_only:
        speech_dir, dog_dir = PROJECT_ROOT / "data" / "librispeech_subset", PROJECT_ROOT / "data" / "esc50_dog"
        transcripts = dict(line.strip().split(" ", 1) for line in open(speech_dir / "transcripts.txt") if line.strip())
        speech_files = sorted(speech_dir.glob("*.flac"))
        dog_files = sorted(dog_dir.glob("*.wav"))
        if args.heldout_dogs:            # the half of ESC-50 dogs that the Step 20c/21a detectors never trained on
            dog_files = dog_files[1::2]
            from scripts.train_overlap_augmented import split_speakers   # and the speakers Step 21a never trained on
            speech_files = split_speakers(speech_files)[1]
        det = dict(np.load(MODEL_PATH))
        extra_models = {name: dict(np.load(MODEL_PATH.parent / f))           # Steps 20c / 21a, if trained
                        for name, f in (("eat40aug", "human_detector_eat40_aug.npz"), ("eat40aug2", "human_detector_eat40_aug2.npz"))
                        if (MODEL_PATH.parent / f).exists()}
        from scripts.extract_efficientat import load_model

        eat = load_model(EAT_DIR, str(det["efficientat_model"]))
        levels = [str(x) for x in det["levels"]]
        silero = SpeechDetector(threshold=0.5)
        asr = WhisperASR("small", language="", num_threads=1)
        for k in range(args.n_scenes):
            f = OUT / "scenes" / f"scene_{k:03d}.json"
            if k % args.n_workers != args.worker or f.exists():
                continue
            mix, sp_act, dog_act, utts = make_scene(k, speech_files, transcripts, dog_files, *counts)
            x = torch.from_numpy(mix)
            nf = len(sp_act)
            prob, centres = human_probability(x, det, eat)
            methods = {f"eat40_{lv}": flagged_segments(prob, centres, float(det["thresholds"][i]), SCENE_S)
                       for i, lv in enumerate(levels)}
            for mname, mdet in extra_models.items():
                prob_a, _ = human_probability(x, mdet, eat)
                methods.update({f"{mname}_{lv}": flagged_segments(prob_a, centres, float(mdet["thresholds"][i]), SCENE_S)
                                for i, lv in enumerate(levels)})
            methods["silero"] = [(s.start_s, s.end_s) for s in silero.detect(x)]
            methods["oracle"] = [(u["start_s"], u["end_s"]) for u in utts]
            dog_only = dog_act & ~sp_act
            res = {"scene": k, "speech_s": float(sp_act.sum()) / 100, "dog_only_s": float(dog_only.sum()) / 100, "methods": {}}
            windows = [(max(0, int((u["start_s"] - CONTEXT_S) * SR)), min(len(mix), int((u["end_s"] + CONTEXT_S) * SR))) for u in utts]
            res["original"] = [asr.transcribe(loud(x[a:b])) for a, b in windows]
            for name, segs in methods.items():
                m = mask_from(segs, nf)
                y = x.clone()
                for a, b in segs:
                    y[int(a * SR) : int(math.ceil(b * SR))] = 0
                hyps = [asr.transcribe(loud(y[a:b])) if torch.any(y[a:b] != x[a:b]) else res["original"][i]
                        for i, (a, b) in enumerate(windows)]
                res["methods"][name] = {"speech_muted": float((m & sp_act).sum() / max(1, sp_act.sum())),
                                        "dog_only_muted": float((m & dog_only).sum() / max(1, dog_only.sum())),
                                        "hyps": hyps}
            res["refs"] = [u["ref"] for u in utts]
            f.write_text(json.dumps(res))
            print(f"scene {k}: " + ", ".join(f"{n} {v['speech_muted']:.2f}/{v['dog_only_muted']:.2f}" for n, v in res["methods"].items()),
                  flush=True)

    files = sorted((OUT / "scenes").glob("scene_*.json"))
    if len(files) < args.n_scenes:
        print(f"{len(files)}/{args.n_scenes} scenes done - summary skipped")
        return
    scenes = [json.loads(f.read_text()) for f in files]

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    def rec(ref, hyp):
        r = words_recovered(ref, hyp)
        return 0.0 if math.isnan(r) else r

    def n_correct(ref, hyp):
        return round(rec(ref, hyp) * len(normalize_text(ref).split()))

    rows = [{"method": "original (no processing)", "speech_time_muted": 0.0, "dog_only_time_muted": 0.0,
             "utterances_2plus_correct_words": round(mean([n_correct(r, h) >= 2 for s in scenes for r, h in zip(s["refs"], s["original"])]), 3),
             "reference_words_recovered": round(mean([rec(r, h) for s in scenes for r, h in zip(s["refs"], s["original"])]), 3)}]
    for name in scenes[0]["methods"]:
        sp = sum(s["methods"][name]["speech_muted"] * s["speech_s"] for s in scenes) / max(1e-9, sum(s["speech_s"] for s in scenes))
        dg = sum(s["methods"][name]["dog_only_muted"] * s["dog_only_s"] for s in scenes) / max(1e-9, sum(s["dog_only_s"] for s in scenes))
        pairs = [(r, h) for s in scenes for r, h in zip(s["refs"], s["methods"][name]["hyps"])]
        rows.append({"method": name, "speech_time_muted": round(sp, 3), "dog_only_time_muted": round(dg, 3),
                     "utterances_2plus_correct_words": round(mean([n_correct(r, h) >= 2 for r, h in pairs]), 3),
                     "reference_words_recovered": round(mean([rec(r, h) for r, h in pairs]), 3)})
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    if args.scene_type == "mixed":
        plot(rows, OUT / "privacy_vs_dog_loss.png")


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = {"eat40": ("ELTE only (Steps 17-19)", "#1baf7a"), "eat40aug": ("+ other dogs as negatives (20c)", "#2a78d6"),
                "eat40aug2": ("+ other dogs and speech-under-bark (21a)", "#eb6834")}
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    by = {r["method"]: r for r in rows}
    for fam, (label, color) in families.items():
        pts = [(by[f"{fam}_{lv}"]["dog_only_time_muted"], by[f"{fam}_{lv}"]["utterances_2plus_correct_words"])
               for lv in ("light", "standard", "strict") if f"{fam}_{lv}" in by]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(np.array(xs) * 100, np.array(ys) * 100, "-o", color=color, lw=1.8, ms=5, label=label)
    for name, marker, label in (("silero", "s", "Silero VAD (generic)"), ("oracle", "D", "mute the exact utterances")):
        if name in by:
            ax.scatter([by[name]["dog_only_time_muted"] * 100], [by[name]["utterances_2plus_correct_words"] * 100],
                       marker=marker, color="#52514e", s=40, label=label, zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 40)
    ax.set_xlabel("Dog-only time muted (%)", fontsize=9, color="#52514e")
    ax.set_ylabel("Utterances with 2+ correct words still readable (%)", fontsize=9, color="#52514e")
    fig.suptitle("Unseen speakers, unseen dogs, simulated rooms (speech and barks overlap)", x=0.01, ha="left", fontsize=11.5)
    ax.set_title("40 scenes x 3 utterances; each line = light / standard / strict, thresholds set on ELTE", loc="left",
                 fontsize=8.5, color="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(color="#e4e3df", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
