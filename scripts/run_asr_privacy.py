"""Step 7: independent privacy check - can Whisper still read the words?

Same scenes as Step 4 (LibriSpeech speech 0-5 s, dog 2.5-7.5 s, SNR 10/0/-10
dB). Every method's output (first 7.5 s = speech + overlapping dog) is
transcribed by Whisper small.en, an attacker that is much stronger than the
VAD used for detection. The reference is Whisper's transcript of the clean
speech clip, so the numbers measure what the anonymization took away, not
Whisper's own mistakes.

* ``wer``              - word error rate vs. the clean transcript (higher = more private;
                         around 1.0 = nothing readable).
* ``words_recovered``  - share of the clean transcript's words found in the output
                         (0 = no word can be read back).

Usage (from inside speech_anonymization/):
    python -m scripts.download_models --asr     # once
    python -m scripts.run_asr_privacy --speech-dir ../data/librispeech_subset
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch

from scripts.run_method_comparison import collect_dog_clips, collect_speech_clips
from vad_anonymization.anonymize import gated_subtract, subtract, vad_mute
from vad_anonymization.asr import WhisperASR, word_error_rate, words_recovered
from vad_anonymization.detection import SpeechDetector
from vad_anonymization.enhancers import ENHANCERS
from vad_anonymization.mixing import build_scene

SAMPLE_RATE = 16_000
ASR_SPAN_S = 7.5
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METHODS = ["passthrough", "vad_mute", "subtract_dtln", "gated_subtract_dtln", "gated_subtract_gtcrn",
           "gated_subtract_deepfilternet3"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speech", type=Path, default=DATA_DIR / "sample_speech.wav")
    p.add_argument("--speech-dir", type=Path, default=DATA_DIR / "librispeech_subset")
    p.add_argument("--per-speaker", type=int, default=2)
    p.add_argument("--dog-dirs", type=Path, nargs="+", default=[DATA_DIR / "esc50_dog", DATA_DIR / "audioset_candidates"])
    p.add_argument("--snr", type=float, nargs="+", default=[10, 0, -10])
    p.add_argument("--max-dogs", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "asr_privacy")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    detector = SpeechDetector()
    asr = WhisperASR("small.en", "en")
    enhancers = {n: ENHANCERS[n]() for n in ("dtln", "gtcrn", "deepfilternet3")}
    speech_clips = collect_speech_clips(args, detector)
    dog_clips = collect_dog_clips(args.dog_dirs)[: args.max_dogs]

    results_path = args.output_dir / "results.csv"
    fields = ["speech_clip", "dog_clip", "snr_db", "method", "wer", "words_recovered", "reference", "transcript"]
    done = set()
    if results_path.exists():
        done = {(r["dog_clip"], float(r["snr_db"]), r["method"]) for r in csv.DictReader(open(results_path))}
    f = open(results_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if not done:
        w.writeheader()

    span = int(ASR_SPAN_S * SAMPLE_RATE)
    ref_cache: dict[str, str] = {}
    t0 = time.time()
    for i, (dog_name, _source, dog) in enumerate(dog_clips):
        speech_name, speech = speech_clips[i % len(speech_clips)]
        if speech_name not in ref_cache:
            ref_cache[speech_name] = asr.transcribe(speech)
        ref = ref_cache[speech_name]
        for snr in args.snr:
            todo = [m for m in METHODS if (dog_name, float(snr), m) not in done]
            if not todo:
                continue
            x = build_scene(speech, dog, SAMPLE_RATE, snr_db=snr).waveform
            segments = detector.detect(x)
            s_hats: dict[str, torch.Tensor] = {}
            for m in todo:
                if m == "passthrough":
                    y = x
                elif m == "vad_mute":
                    y = vad_mute(x, segments)
                else:
                    gated, name = m.startswith("gated_"), m.split("subtract_")[1]
                    if name not in s_hats:
                        s_hats[name] = enhancers[name].enhance(x)
                    s_hat = s_hats[name]
                    y = gated_subtract(x, s_hat, segments) if gated else subtract(x, s_hat)
                hyp = asr.transcribe(y[:span])
                w.writerow({"speech_clip": speech_name, "dog_clip": dog_name, "snr_db": snr, "method": m,
                            "wer": round(word_error_rate(ref, hyp), 3), "words_recovered": round(words_recovered(ref, hyp), 3),
                            "reference": ref, "transcript": hyp})
            f.flush()
        print(f"  {i + 1}/{len(dog_clips)} dog clips ({time.time() - t0:.0f}s)", flush=True)
    f.close()

    rows = list(csv.DictReader(open(results_path)))
    summary = []
    for snr in args.snr:
        for m in METHODS:
            sel = [r for r in rows if float(r["snr_db"]) == snr and r["method"] == m]
            wers = [float(r["wer"]) for r in sel if math.isfinite(float(r["wer"]))]
            rec = [float(r["words_recovered"]) for r in sel if math.isfinite(float(r["words_recovered"]))]
            summary.append({
                "snr_db": snr, "method": m, "n": len(sel),
                "mean_wer": round(statistics.mean(wers), 3),
                "mean_words_recovered": round(statistics.mean(rec), 3),
                "share_no_word_recovered": round(sum(v == 0 for v in rec) / len(rec), 3),
                "share_empty_transcript": round(sum(not r["transcript"].strip() for r in sel) / len(sel), 3),
            })
    with open(args.output_dir / "summary.csv", "w", newline="") as g:
        sw = csv.DictWriter(g, fieldnames=list(summary[0]))
        sw.writeheader()
        sw.writerows(summary)
    print(f"\n{'method':32s} {'SNR':>4s} {'WER':>6s} {'words back':>10s} {'0 words':>8s}")
    for s in summary:
        print(f"{s['method']:32s} {s['snr_db']:4.0f} {s['mean_wer']:6.2f} {100 * s['mean_words_recovered']:9.0f}% "
              f"{100 * s['share_no_word_recovered']:7.0f}%")


if __name__ == "__main__":
    main()
