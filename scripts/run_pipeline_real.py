"""Step 12: the whole pipeline on real recordings, with NO expert labels.

Steps 6-8 used the expert annotations as a perfect detector ("oracle"), and
Step 11 measured the two trained detectors separately. Here everything is put
together the way a real anonymizer would run:

    trained human detector  ->  human segments
    trained dog detector    ->  dog segments
    hybrid removal          ->  mute human-only time, subtract speech (DTLN) where a dog is detected

Both detectors' outputs are the leave-one-participant-out predictions from
Steps 9 and 11, so every session is processed by models that never saw that
participant. The expert annotations are used ONLY to score the result:

* privacy - Whisper (multilingual, loudness-normalized, as in Step 8) on every
  annotated human event: share of events with >= 2 readable words, and share
  of the original words still readable;
* dog     - share of annotated dog-vocalization time that is left untouched,
  subtracted from (small loss), or muted (lost).

Configurations: the original approach (Silero VAD + mute), the deployable
hybrid at three human-detector operating points, the same detector with
mute-only (to show what the dog routing buys), and the oracle hybrid (Step 8)
as the upper bound.

Usage (from inside speech_anonymization/), after train_human_detector and
eval_hybrid_routing:
    python -m scripts.run_pipeline_real --worker 0 --n-workers 2   # and --worker 1 in parallel
    python -m scripts.run_pipeline_real --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_asr_real import CONTEXT_S, MAX_S, loud, n_words
from vad_anonymization.anonymize import hybrid_remove, segments_to_sample_mask, vad_mute
from vad_anonymization.asr import WhisperASR, words_recovered
from vad_anonymization.audio_io import load_audio
from vad_anonymization.detection import SpeechDetector, SpeechSegment
from vad_anonymization.elte_barks import load_sessions
from vad_anonymization.enhancers import ENHANCERS

SAMPLE_RATE = 16_000
HOP_S = 0.24
PAD_S = 0.12
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"

# name -> (detector, human threshold, dog threshold or None for mute-only)
CONFIGS = {
    "silero_mute": ("silero", 0.5, None),
    "hybrid_strict": ("trained", 0.7304, 0.5),
    "hybrid_balanced": ("trained", 0.409, 0.5),
    "hybrid_high_recall": ("trained", 0.1372, 0.5),
    "mute_balanced": ("trained", 0.409, None),
    "mute_strict": ("trained", 0.7304, None),
    "hybrid_balanced_dog07": ("trained", 0.409, 0.7),
}


def frames_to_segments(centres: np.ndarray, flags: np.ndarray, total_s: float, pad_s: float = PAD_S) -> list[SpeechSegment]:
    """Flagged frames (each covering centre +- hop/2) -> merged, padded time segments."""
    segs: list[list[float]] = []
    for c in centres[flags]:
        a, b = max(0.0, c - HOP_S / 2 - pad_s), min(total_s, c + HOP_S / 2 + pad_s)
        if segs and a <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    return [SpeechSegment(a, b) for a, b in segs]


def split_per_session(sessions, arr: np.ndarray) -> dict[str, np.ndarray]:
    out, i = {}, 0
    for s in sessions:
        n = len(np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"])
        out[s.name] = arr[i : i + n]
        i += n
    assert i == len(arr), "prediction length does not match the features"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--sessions", nargs="+", default=None, help="only these sessions (quick runs)")
    p.add_argument("--output-dir", type=Path, default=OUT / "pipeline_real")
    p.add_argument("--human-oof", type=Path, default=OUT / "human_detector" / "oof_predictions.npz",
                   help="human detector predictions (e.g. outputs/human_detector_micfusion/oof_predictions.npz, Step 15)")
    p.add_argument("--dog-oof", type=Path, default=OUT / "hybrid_routing" / "dog_oof_predictions.npz")
    p.add_argument("--extra-configs", type=Path, default=None,
                   help='JSON {name: ["trained", human_threshold, dog_threshold_or_null]} added to the built-in configs')
    p.add_argument("--remover", choices=["dtln", "gtcrn", "gtcrn_ft"], default="dtln",
                   help="speech estimate for the subtraction part (gtcrn_ft: fine-tuned model of the other participant group)")
    args = p.parse_args()
    if args.extra_configs:
        extra = {k: tuple(v) for k, v in json.loads(args.extra_configs.read_text()).items()}
        CONFIGS.update(extra)
        if args.configs == [c for c in CONFIGS if c not in extra]:  # default list: run only the new ones
            args.configs = list(extra)
    sess_dir = args.output_dir / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    sessions = load_sessions(args.data_root)

    if not args.summarize_only:
        human_p = split_per_session(sessions, np.load(args.human_oof)["prob"])
        # a config may name its own detector predictions as 4th entry (compare detectors in one run)
        other_p: dict = {}
        for c in args.configs:
            if len(CONFIGS[c]) > 3 and CONFIGS[c][3] not in other_p:
                z = np.load(OUT.parent / CONFIGS[c][3])
                if "hop_s" in z:  # Step 25c: 20 ms predictions on the Whisper-encoder grid
                    parts = np.split(z["prob"], np.cumsum(z["lengths"])[:-1])
                    other_p[CONFIGS[c][3]] = {"fine": (float(z["pad_s"]), float(z["gap_s"])),
                                              **{s.name: q for s, q in zip(sessions, parts)}}
                else:
                    other_p[CONFIGS[c][3]] = split_per_session(sessions, z["prob"])
        dog_p = split_per_session(sessions, np.load(args.dog_oof)["prob"])
        before = {r["id"]: r["before"] for r in csv.DictReader(open(OUT / "asr_real" / "per_event.csv", encoding="utf-8"))}
        asr = WhisperASR("small", language="", num_threads=args.threads)
        from vad_anonymization.enhancers import GTCRNTorchEnhancer

        def get_remover(session_name: str):
            if args.remover == "dtln":
                return ENHANCERS["dtln"]()
            if args.remover == "gtcrn":
                return GTCRNTorchEnhancer()
            other = "even" if int(session_name[2:4]) % 2 else "odd"  # never the participant's own group
            return GTCRNTorchEnhancer(checkpoint=Path(__file__).resolve().parent.parent / "models" / "gtcrn" / f"gtcrn_ft_{other}.pt")

        removers: dict = {}
        silero = SpeechDetector(threshold=0.5)
        t0 = time.time()
        for i, sess in enumerate(sessions):
            if args.sessions and sess.name not in args.sessions:
                continue
            out_file = sess_dir / f"{sess.name}.json"
            if i % args.n_workers != args.worker:
                continue
            result = json.loads(out_file.read_text()) if out_file.exists() else {"session": sess.name, "configs": {}}
            todo = [c for c in args.configs if c not in result["configs"]]
            if not todo:
                continue
            audio = load_audio(sess.audio_path, SAMPLE_RATE)
            asr_cache: dict = {}
            n, total = audio.shape[-1], audio.shape[-1] / SAMPLE_RATE
            centres = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")["centres"]
            dog_mask_true = segments_to_sample_mask([SpeechSegment(d.start_s, d.end_s) for d in sess.dog], n)
            human_mask_true = segments_to_sample_mask([SpeechSegment(e.start_s, e.end_s) for e in sess.human], n)
            dog_only_true = dog_mask_true & ~human_mask_true
            key = "x" if args.remover != "gtcrn_ft" else int(sess.name[2:4]) % 2
            if key not in removers and any(CONFIGS[c][2] is not None for c in todo):
                removers[key] = get_remover(sess.name)
            needs_remover = any(CONFIGS[c][2] is not None and CONFIGS[c][0] != "silero" for c in todo)
            s_hat = removers[key].enhance(audio) if needs_remover else None  # one speech estimate per session
            for name in todo:
                kind, th_h, th_d = CONFIGS[name][:3]
                probs = other_p[CONFIGS[name][3]] if len(CONFIGS[name]) > 3 else human_p
                if kind == "silero":
                    h_segs = silero.detect(audio)
                elif "fine" in probs:
                    from scripts.train_fine_detector import fine_segments

                    times = np.load(OUT / "whisper_enc" / f"{sess.name}.npz")["times"]
                    h_segs = [SpeechSegment(a, b) for a, b in
                              fine_segments(times, probs[sess.name] >= th_h, total, *probs["fine"])]
                else:
                    h_segs = frames_to_segments(centres, probs[sess.name] >= th_h, total)
                h_mask = segments_to_sample_mask(h_segs, n)
                if th_d is None:
                    y = vad_mute(audio, h_segs)
                    muted, subtracted = h_mask, torch.zeros_like(h_mask)
                else:
                    d_segs = frames_to_segments(centres, dog_p[sess.name] >= th_d, total, pad_s=0.0)
                    d_mask = segments_to_sample_mask(d_segs, n)
                    y = hybrid_remove(audio, s_hat, h_segs, d_segs)
                    muted, subtracted = h_mask & ~d_mask, h_mask & d_mask
                events = []
                for k, e in enumerate(sess.human):
                    a_s, b_s = max(0.0, e.start_s - CONTEXT_S), min(total, e.end_s + CONTEXT_S, e.start_s - CONTEXT_S + MAX_S)
                    a, b = int(a_s * SAMPLE_RATE), int(b_s * SAMPLE_RATE)
                    eid = f"{sess.name}_{k:03d}"
                    if b - a < int(0.1 * SAMPLE_RATE) or eid not in before:
                        continue
                    ref = before[eid]
                    if h_mask[a:b].any():
                        # identical audio in this window (another config muted the same samples): reuse
                        key_w = (a, b, hashlib.md5(y[a:b].numpy().tobytes()).hexdigest())
                        if key_w not in asr_cache:
                            asr_cache[key_w] = asr.transcribe(loud(y[a:b]))
                        hyp = asr_cache[key_w]
                    else:
                        hyp = ref  # untouched window = identical audio: reuse the original transcript
                    ea, eb = int(e.start_s * SAMPLE_RATE), int(e.end_s * SAMPLE_RATE)
                    events.append({
                        "id": eid, "duration_s": round(e.duration_s, 3),
                        "dog_overlap": bool(dog_mask_true[ea:eb].any()),
                        "covered": round(float(h_mask[ea:eb].float().mean()), 3) if eb > ea else 0.0,
                        "words_before": n_words(ref), "words_after": n_words(hyp),
                        "recovered": round(words_recovered(ref, hyp), 3) if n_words(ref) else None,
                        "after": hyp,
                    })
                dt = float(dog_only_true.sum()) / SAMPLE_RATE
                result["configs"][name] = {
                    "events": events,
                    "dog_s": dt,
                    "dog_muted_s": float((dog_only_true & muted).sum()) / SAMPLE_RATE,
                    "dog_subtracted_s": float((dog_only_true & subtracted).sum()) / SAMPLE_RATE,
                    "processed_s": float(h_mask.sum()) / SAMPLE_RATE,
                    "total_s": total,
                }
            out_file.write_text(json.dumps(result, ensure_ascii=False))
            print(f"{sess.name}: {len(sess.human)} events ({time.time() - t0:.0f}s)", flush=True)

    files = [sess_dir / f"{s.name}.json" for s in sessions]
    missing = [f.stem for f in files if not f.exists()]
    if missing:
        print(f"{len(missing)} sessions not finished yet - summary skipped")
        return
    data = [json.loads(f.read_text()) for f in files]
    asr_real = list(csv.DictReader(open(OUT / "asr_real" / "per_event.csv", encoding="utf-8")))

    def privacy(rows, words_key, rec_key):
        with_words = [r for r in rows if int(r["words_before"]) >= 2]
        rec = [float(r[rec_key]) for r in with_words if r[rec_key] not in ("", None)]
        return {
            "n_events": len(rows),
            "share_2plus_words": round(sum(int(r[words_key]) >= 2 for r in rows) / len(rows), 3),
            "share_2plus_words_long": round(sum(int(r[words_key]) >= 2 for r in rows if float(r["duration_s"]) > 1) /
                                            max(1, sum(float(r["duration_s"]) > 1 for r in rows)), 3),
            "words_recovered": round(sum(rec) / len(rec), 3) if rec else None,
        }

    summary = [
        {"config": "original (no processing)", **privacy(asr_real, "words_before", "words_before"),
         "words_recovered": 1.0, "human_events_covered": 0.0, "dog_untouched": 1.0, "dog_subtracted": 0.0, "dog_muted": 0.0,
         "time_processed": 0.0},
        {"config": "background floor (no human)", **privacy(asr_real, "words_control", "words_control"),
         "words_recovered": None, "human_events_covered": None, "dog_untouched": None, "dog_subtracted": None,
         "dog_muted": None, "time_processed": None},
    ]
    for name in CONFIGS:
        if not all(name in d["configs"] for d in data):  # not computed for every session (yet)
            continue
        rows = [ev for d in data for ev in d["configs"][name]["events"]]
        dog_s = sum(d["configs"][name]["dog_s"] for d in data)
        muted = sum(d["configs"][name]["dog_muted_s"] for d in data)
        sub = sum(d["configs"][name]["dog_subtracted_s"] for d in data)
        proc = sum(d["configs"][name]["processed_s"] for d in data)
        tot = sum(d["configs"][name]["total_s"] for d in data)
        summary.append({
            "config": name, **privacy(rows, "words_after", "recovered"),
            "human_events_covered": round(sum(r["covered"] for r in rows) / len(rows), 3),
            "dog_untouched": round(1 - (muted + sub) / dog_s, 3), "dog_subtracted": round(sub / dog_s, 3),
            "dog_muted": round(muted / dog_s, 3), "time_processed": round(proc / tot, 3),
        })
    oracle = {"config": "oracle hybrid (expert labels)", **privacy(asr_real, "words_hybrid", "recovered_hybrid"),
              "human_events_covered": 1.0, "dog_untouched": None, "dog_subtracted": None, "dog_muted": None, "time_processed": None}
    summary.append(oracle)
    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(args.output_dir / "per_event.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "id", "duration_s", "dog_overlap", "covered", "words_before", "words_after", "recovered", "after"])
        w.writeheader()
        for d in data:
            for name, c in d["configs"].items():
                for ev in c["events"]:
                    w.writerow({"config": name, **ev})
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
