"""Step 8: on the REAL recordings - which human sounds contain readable words,
and are they still readable after removal?

For every expert-annotated human event (+0.3 s), multilingual Whisper
"small" (language auto-detected) transcribes:

* ``before``  - the original audio,
* ``subtract`` - the audio after oracle removal by gated subtraction inside
                the annotated event (see run_oracle_real.py),
* ``hybrid``  - oracle hybrid removal: human-only time muted, subtraction
                only where a dog call overlaps (anonymize.hybrid_remove), and
* ``control`` - a background window of the same length from the same
                session with no human or dog annotation within 1 s.

Every window is normalized to the same loudness before transcription (an
attacker would turn the volume up), so quiet leftovers are not "private"
just because they are quiet. Whisper invents words on pure noise
sometimes; the control windows measure how often, so the before/after
numbers can be read against that floor.

Outputs (outputs/asr_real/): sessions/<session>.csv as each session finishes
(a rerun skips finished sessions), then per_event.csv (with transcripts -
keep local, these are real people's words) and summary.csv. Several workers
can share the sessions: ``--worker 0 --n-workers 2`` and ``--worker 1 --n-workers 2``.

Usage (from inside speech_anonymization/):
    python -m scripts.run_asr_real --method dtln
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch

from scripts.run_oracle_real import PAD_S, speech_estimate
from vad_anonymization.anonymize import gated_subtract, hybrid_remove
from vad_anonymization.asr import WhisperASR, normalize_text, words_recovered
from vad_anonymization.audio_io import load_audio
from vad_anonymization.detection import SpeechSegment
from vad_anonymization.elte_barks import load_sessions
from vad_anonymization.enhancers import ENHANCERS

SAMPLE_RATE = 16_000
CONTEXT_S = 0.3
MAX_S = 29.0  # Whisper reads at most 30 s at once
TARGET_RMS = 10 ** (-20 / 20)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIELDS = ["id", "session", "start_s", "duration_s", "dog_overlap", "words_before", "words_subtract", "words_hybrid",
          "words_control", "recovered_subtract", "recovered_hybrid", "before", "subtract", "hybrid", "control"]


def loud(x: torch.Tensor) -> torch.Tensor:
    rms = float(x.double().pow(2).mean().sqrt())
    return torch.tanh(x * min(1000.0, TARGET_RMS / rms)) if rms > 0 else x


def n_words(text: str) -> int:
    return len(normalize_text(text).split())


def control_window(rng: random.Random, total_s: float, dur: float, busy: list[tuple[float, float]]) -> tuple[float, float] | None:
    for _ in range(200):
        t0 = rng.uniform(0, max(0.0, total_s - dur))
        if all(t0 + dur <= a - 1.0 or t0 >= b + 1.0 for a, b in busy):
            return t0, t0 + dur
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--method", default="dtln", help="enhancer for removal, '_norm' suffix = loudness-normalized input")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--sessions", nargs="+", default=None, help="only these sessions (quick runs)")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "asr_real")
    args = p.parse_args()
    sess_dir = args.output_dir / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    asr = WhisperASR("small", language="", num_threads=args.threads)
    normalize = args.method.endswith("_norm")
    enhancer = ENHANCERS[args.method.removesuffix("_norm")]()

    t_start = time.time()
    sessions = [s for s in load_sessions(args.data_root) if not args.sessions or s.name in args.sessions]
    for i, sess in enumerate(sessions):
        if i % args.n_workers != args.worker or (sess_dir / f"{sess.name}.csv").exists():
            continue
        rng, rows = random.Random(sess.name), []
        audio = load_audio(sess.audio_path, SAMPLE_RATE)
        total = audio.shape[-1] / SAMPLE_RATE
        segs = [SpeechSegment(max(0.0, e.start_s - PAD_S), min(total, e.end_s + PAD_S)) for e in sess.human]
        dog_segs = [SpeechSegment(d.start_s, d.end_s) for d in sess.dog]
        s_hat = speech_estimate(audio, segs, enhancer, normalize)
        outputs = {"subtract": gated_subtract(audio, s_hat, segs), "hybrid": hybrid_remove(audio, s_hat, segs, dog_segs)}
        busy = [(e.start_s, e.end_s) for e in sess.human + sess.dog]
        for k, e in enumerate(sess.human):
            t0, t1 = max(0.0, e.start_s - CONTEXT_S), min(total, e.end_s + CONTEXT_S, e.start_s - CONTEXT_S + MAX_S)
            a, b = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
            if b - a < int(0.1 * SAMPLE_RATE):
                continue
            before = asr.transcribe(loud(audio[a:b]))
            after = {m: asr.transcribe(loud(y[a:b])) for m, y in outputs.items()}
            ctl = control_window(rng, total, t1 - t0, busy)
            control = asr.transcribe(loud(audio[int(ctl[0] * SAMPLE_RATE) : int(ctl[1] * SAMPLE_RATE)])) if ctl else ""
            rows.append({
                "id": f"{sess.name}_{k:03d}", "session": sess.name, "start_s": round(e.start_s, 3),
                "duration_s": round(e.duration_s, 3),
                "dog_overlap": any(min(d.end_s, e.end_s) > max(d.start_s, e.start_s) for d in sess.dog),
                "words_before": n_words(before), "words_subtract": n_words(after["subtract"]),
                "words_hybrid": n_words(after["hybrid"]), "words_control": n_words(control),
                "recovered_subtract": round(words_recovered(before, after["subtract"]), 3) if n_words(before) else "",
                "recovered_hybrid": round(words_recovered(before, after["hybrid"]), 3) if n_words(before) else "",
                "before": before, "subtract": after["subtract"], "hybrid": after["hybrid"], "control": control,
            })
        with open(sess_dir / f"{sess.name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{sess.name}: {len(sess.human)} events ({time.time() - t_start:.0f}s)", flush=True)

    missing = [s.name for s in sessions if not (sess_dir / f"{s.name}.csv").exists()]
    if missing:
        print(f"{len(missing)} sessions not finished yet (other workers?) - summary skipped")
        return
    rows = []
    for s in sessions:
        for r in csv.DictReader(open(sess_dir / f"{s.name}.csv", encoding="utf-8")):
            r["duration_s"] = float(r["duration_s"])
            r["dog_overlap"] = r["dog_overlap"] == "True"
            for key in FIELDS:
                if key.startswith("words_"):
                    r[key] = int(r[key])
                elif key.startswith("recovered_"):
                    r[key] = float(r[key]) if r[key] else ""
            rows.append(r)
    with open(args.output_dir / "per_event.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    def share(sel, key, k=1):
        return round(sum(r[key] >= k for r in sel) / len(sel), 3) if sel else float("nan")

    summary = []
    for name, sel in [("all", rows), ("> 1 s", [r for r in rows if r["duration_s"] > 1.0]),
                      ("human only", [r for r in rows if not r["dog_overlap"]]),
                      ("human + dog overlap", [r for r in rows if r["dog_overlap"]])]:
        with_words = [r for r in sel if r["words_before"] >= 2]
        entry = {"group": name, "enhancer": args.method, "n_events": len(sel)}
        for col in ("before", "subtract", "hybrid", "control"):
            entry[f"share_words_{col}"] = share(sel, f"words_{col}")
            entry[f"share_2plus_words_{col}"] = share(sel, f"words_{col}", 2)
        for col in ("subtract", "hybrid"):
            entry[f"mean_recovered_{col}_2plus"] = (
                round(sum(r[f"recovered_{col}"] for r in with_words) / len(with_words), 3) if with_words else float("nan"))
        summary.append(entry)
    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
