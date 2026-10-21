"""Step 14b: does the fine-tuned remover work better on real far-field speech?

Each fine-tuned GTCRN (scripts/finetune_gtcrn.py) is tested only on the
participant group it never saw during training (odd <-> even), and compared
with the pretrained DTLN and the pretrained GTCRN on exactly the same audio:

1. synthetic (ground truth known): held-out LibriSpeech speakers, simulated lab
   reverberation, REAL background from the test participants. Reported:
   speech left after subtraction (dB, lower = better), background/dog kept
   (projection gain onto the real background, 0 dB = fully kept) and SDR of the
   output against the ideal output = the background alone (higher = better).
2. real recordings, oracle detection (expert annotations): energy drop inside
   each human event and how many audible events reach the local noise floor
   (as in Step 6).
3. Whisper on the real events after oracle hybrid removal (as in Step 8):
   share of events with 2+ readable words, separately for events that
   overlap a dog (where the remover actually matters).

Usage (from inside speech_anonymization/), after both fine-tuning runs:
    python -m scripts.eval_finetuned_gtcrn --worker 0 --n-workers 2   # and --worker 1
    python -m scripts.eval_finetuned_gtcrn --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from scripts.finetune_gtcrn import MODEL_DIR, NoiseBank, SpeechBank, make_example, make_rirs, participant_group, speech_split
from scripts.run_asr_real import CONTEXT_S, MAX_S, loud, n_words
from scripts.run_oracle_real import PAD_S, energy_db, noise_floor_db, speech_estimate
from vad_anonymization.anonymize import gated_subtract, hybrid_remove
from vad_anonymization.asr import WhisperASR, words_recovered
from vad_anonymization.audio_io import load_audio
from vad_anonymization.detection import SpeechSegment
from vad_anonymization.elte_barks import load_sessions
from vad_anonymization.enhancers import DTLNEnhancer, GTCRNTorchEnhancer
from vad_anonymization.metrics import dog_retention_db, sdr_db, speech_leakage_db

SR = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs" / "finetuned_gtcrn"
METHODS = ["dtln", "gtcrn_pretrained", "gtcrn_finetuned", "gtcrn_finetuned_os2"]


class OverSubtract:
    """Spectral over-subtraction on top of a speech estimator.

    |Y| = max(|X| - alpha*|S_hat|, beta*|X|), phase of X. Removes more of the
    speech than plain waveform subtraction, at the cost of some distortion of
    what is left (the dog). Returned as an equivalent "speech estimate"
    x - y, so the gated / hybrid removal code can use it unchanged.
    """

    def __init__(self, base: GTCRNTorchEnhancer, alpha: float = 2.0, beta: float = 0.1) -> None:
        self.base, self.alpha, self.beta = base, alpha, beta

    @torch.no_grad()
    def enhance(self, wave: torch.Tensor) -> torch.Tensor:
        x = wave.float()
        X = torch.view_as_complex(self.base.spec(x[None]))[0]
        S = torch.view_as_complex(self.base.model(self.base.spec(x[None]))[0].contiguous())
        mag = torch.maximum(X.abs() - self.alpha * S.abs(), self.beta * X.abs())
        Y = mag * torch.exp(1j * torch.angle(X))
        y = self.base.wave(torch.view_as_real(Y), x.shape[-1])
        return x - y


def remover(method: str, test_group: str):
    if method == "dtln":
        return DTLNEnhancer()
    if method == "gtcrn_pretrained":
        return GTCRNTorchEnhancer()
    train_group = "even" if test_group == "odd" else "odd"
    ft = GTCRNTorchEnhancer(checkpoint=MODEL_DIR / f"gtcrn_ft_{train_group}.pt")
    return OverSubtract(ft, alpha=2.0) if method == "gtcrn_finetuned_os2" else ft


def synthetic_test(test_sessions, test_group: str, n: int = 60) -> list[dict]:
    _, val_files = speech_split(PROJECT_ROOT / "data" / "librispeech_subset")
    seg = 4 * SR
    noise = NoiseBank(test_sessions, seg)
    speech = SpeechBank(val_files, seg)
    rirs = make_rirs(40, seed=777)
    rng = random.Random(2024)
    examples = []
    for _ in range(n):
        x, s = make_example(rng, speech, noise, rirs, p_no_speech=0.0)
        examples.append((x, s, x - s))
    rows = []
    for m in METHODS:
        enh = remover(m, test_group)
        for i, (x, s, n_) in enumerate(examples):
            y = torch.from_numpy(x) - enh.enhance(torch.from_numpy(x))
            rows.append({"group": test_group, "method": m, "i": i,
                         "speech_left_db": round(speech_leakage_db(y, torch.from_numpy(s)), 2),
                         "background_kept_db": round(dog_retention_db(y, torch.from_numpy(n_)), 2),
                         "output_sdr_db": round(sdr_db(y, torch.from_numpy(n_)), 2)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    sessions = load_sessions(args.data_root)

    if not args.summarize_only:
        # synthetic test, one group per worker
        for k, group in enumerate(["odd", "even"]):
            f = OUT / f"synthetic_{group}.csv"
            if k % args.n_workers == args.worker and not f.exists():
                rows = synthetic_test([s for s in sessions if participant_group(s.name) == group], group)
                with open(f, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                print(f"synthetic {group} done", flush=True)

        before = {r["id"]: r["before"] for r in csv.DictReader(open(OUT.parent / "asr_real" / "per_event.csv", encoding="utf-8"))}
        asr = WhisperASR("small", language="", num_threads=1)
        cache: dict = {}
        t0 = time.time()
        for i, sess in enumerate(sessions):
            out_file = OUT / "sessions" / f"{sess.name}.json"
            if i % args.n_workers != args.worker or out_file.exists():
                continue
            group = participant_group(sess.name)
            audio = load_audio(sess.audio_path, SR)
            total = audio.shape[-1] / SR
            segs = [SpeechSegment(max(0.0, e.start_s - PAD_S), min(total, e.end_s + PAD_S)) for e in sess.human]
            dogs = [SpeechSegment(d.start_s, d.end_s) for d in sess.dog]
            events = []
            outputs = {}
            for m in METHODS:
                key = (m, group)
                if key not in cache:
                    cache[key] = remover(m, group)
                s_hat = speech_estimate(audio, segs, cache[key], False)
                outputs[m] = {"subtract": gated_subtract(audio, s_hat, segs), "hybrid": hybrid_remove(audio, s_hat, segs, dogs)}
            for k, e in enumerate(sess.human):
                eid = f"{sess.name}_{k:03d}"
                a, b = int(e.start_s * SR), int(e.end_s * SR)
                if b - a < int(0.05 * SR) or eid not in before:
                    continue
                lvl, floor = energy_db(audio[a:b]), noise_floor_db(audio, e.start_s, e.end_s)
                wa, wb = max(0.0, e.start_s - CONTEXT_S), min(total, e.end_s + CONTEXT_S, e.start_s - CONTEXT_S + MAX_S)
                ia, ib = int(wa * SR), int(wb * SR)
                row = {"id": eid, "group": group, "duration_s": round(e.duration_s, 3),
                       "dog_overlap": any(min(d.end_s, e.end_s) > max(d.start_s, e.start_s) for d in sess.dog),
                       "above_floor_before": round(lvl - floor, 1), "words_before": n_words(before[eid])}
                for m in METHODS:
                    row[f"{m}_change_db"] = round(energy_db(outputs[m]["subtract"][a:b]) - lvl, 1)
                    row[f"{m}_above_floor"] = round(energy_db(outputs[m]["subtract"][a:b]) - floor, 1)
                    if ib - ia >= int(0.1 * SR):
                        hyp = asr.transcribe(loud(outputs[m]["hybrid"][ia:ib]))
                        row[f"{m}_words_hybrid"] = n_words(hyp)
                        row[f"{m}_recovered_hybrid"] = round(words_recovered(before[eid], hyp), 3) if n_words(before[eid]) else None
                events.append(row)
            out_file.write_text(json.dumps(events, ensure_ascii=False))
            print(f"{sess.name}: {len(events)} events ({time.time() - t0:.0f}s)", flush=True)

    files = [OUT / "sessions" / f"{s.name}.json" for s in sessions]
    if not all(f.exists() for f in files) or not all((OUT / f"synthetic_{g}.csv").exists() for g in ("odd", "even")):
        print("not finished yet - summary skipped")
        return
    events = [e for f in files for e in json.loads(f.read_text())]
    synth = [r for g in ("odd", "even") for r in csv.DictReader(open(OUT / f"synthetic_{g}.csv"))]
    summary = []
    for m in METHODS:
        sm = [r for r in synth if r["method"] == m]
        audible = [e for e in events if e["above_floor_before"] >= 6]
        entry = {
            "method": m,
            "synthetic_speech_left_db": round(statistics.median(float(r["speech_left_db"]) for r in sm), 1),
            "synthetic_background_kept_db": round(statistics.median(float(r["background_kept_db"]) for r in sm), 2),
            "synthetic_output_sdr_db": round(statistics.median(float(r["output_sdr_db"]) for r in sm), 1),
            "real_median_change_db": round(statistics.median(e[f"{m}_change_db"] for e in events), 1),
            "real_audible_to_floor": round(sum(e[f"{m}_above_floor"] <= 3 for e in audible) / len(audible), 3),
        }
        for name, sel in [("all", events), ("overlap", [e for e in events if e["dog_overlap"]]),
                          ("human_only", [e for e in events if not e["dog_overlap"]])]:
            sel = [e for e in sel if f"{m}_words_hybrid" in e]
            entry[f"hybrid_2plus_words_{name}"] = round(sum(e[f"{m}_words_hybrid"] >= 2 for e in sel) / len(sel), 3)
        wb = [e for e in events if e["words_before"] >= 2 and e.get(f"{m}_recovered_hybrid") is not None]
        entry["hybrid_words_recovered"] = round(sum(e[f"{m}_recovered_hybrid"] for e in wb) / len(wb), 3)
        summary.append(entry)
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(OUT / "per_event.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(events[0]))
        w.writeheader()
        w.writerows(events)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
