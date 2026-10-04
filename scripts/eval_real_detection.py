"""Step 5a: how well does Silero VAD find human voices in REAL lab recordings?

Uses the ELTE BARKS Lab play sessions (30 sessions, ~2 h), where experts
annotated every dog sound and every human vocalization. Silero VAD is run
on each full session and compared with the annotations on a 10 ms grid:

* recall          - share of annotated human-voice time that VAD flags
* event recall    - share of human vocalizations with >= 50% of their time flagged
* precision       - share of VAD-flagged time that lies on/near (+-0.3 s) a
                    human annotation
* dog false alarm - share of dog-sound time (no human within 0.5 s) that VAD
                    flags as speech, overall and per dog sound type. With
                    detect-and-mute, that is dog data that gets deleted.

Usage (from inside speech_anonymization/):
    python -m scripts.eval_real_detection
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

from vad_anonymization.audio_io import load_audio
from vad_anonymization.detection import SpeechDetector
from vad_anonymization.elte_barks import events_to_mask, load_sessions

SAMPLE_RATE = 16_000
FRAME_S = 0.01
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "real_detection")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(args.data_root)
    logger.info("%d sessions", len(sessions))

    totals = {t: defaultdict(float) for t in args.thresholds}
    per_type = {t: defaultdict(lambda: [0.0, 0.0]) for t in args.thresholds}  # type -> [flagged s, total s]
    session_rows = []
    detectors = {t: SpeechDetector(threshold=t) for t in args.thresholds}

    for i, sess in enumerate(sessions, 1):
        audio = load_audio(sess.audio_path, SAMPLE_RATE)
        n = int(audio.shape[-1] / SAMPLE_RATE / FRAME_S)
        human = events_to_mask(sess.human, n, FRAME_S)
        human_tol = events_to_mask(sess.human, n, FRAME_S, pad_s=0.3)
        near_human = events_to_mask(sess.human, n, FRAME_S, pad_s=0.5)
        dog = events_to_mask(sess.dog, n, FRAME_S)
        dog_only = dog & ~near_human

        for t, det in detectors.items():
            segs = det.detect(audio)
            vad = torch.zeros(n, dtype=torch.bool)
            for s in segs:
                vad[int(s.start_s / FRAME_S) : int(round(s.end_s / FRAME_S))] = True
            tot = totals[t]
            tot["human_s"] += float(human.sum()) * FRAME_S
            tot["human_flagged_s"] += float((vad & human).sum()) * FRAME_S
            tot["vad_s"] += float(vad.sum()) * FRAME_S
            tot["vad_near_human_s"] += float((vad & human_tol).sum()) * FRAME_S
            tot["dog_only_s"] += float(dog_only.sum()) * FRAME_S
            tot["dog_only_flagged_s"] += float((vad & dog_only).sum()) * FRAME_S
            tot["unannotated_flagged_s"] += float((vad & ~human_tol & ~dog).sum()) * FRAME_S
            for e in sess.human:
                a, b = int(e.start_s / FRAME_S), max(int(e.start_s / FRAME_S) + 1, int(round(e.end_s / FRAME_S)))
                tot["human_events"] += 1
                tot["human_events_hit"] += float(vad[a:b].float().mean() >= 0.5) if b <= n else 0.0
            for e in sess.dog:
                a, b = int(e.start_s / FRAME_S), int(round(e.end_s / FRAME_S))
                region = dog_only[a:b]
                if region.sum() == 0:
                    continue
                per_type[t][e.label.strip()][0] += float((vad[a:b] & region).sum()) * FRAME_S
                per_type[t][e.label.strip()][1] += float(region.sum()) * FRAME_S
            session_rows.append({
                "session": sess.name, "threshold": t,
                "human_s": round(float(human.sum()) * FRAME_S, 2),
                "recall": round(float((vad & human).sum()) / max(1.0, float(human.sum())), 4),
                "vad_s": round(float(vad.sum()) * FRAME_S, 2),
                "dog_only_false_alarm": round(float((vad & dog_only).sum()) / max(1.0, float(dog_only.sum())), 4),
            })
        logger.info("[%d/%d] %s", i, len(sessions), sess.name)

    summary, type_rows = [], []
    for t in args.thresholds:
        tot = totals[t]
        summary.append({
            "threshold": t,
            "human_s": round(tot["human_s"], 1),
            "recall": round(tot["human_flagged_s"] / tot["human_s"], 4),
            "event_recall": round(tot["human_events_hit"] / tot["human_events"], 4),
            "precision": round(tot["vad_near_human_s"] / tot["vad_s"], 4) if tot["vad_s"] else float("nan"),
            "vad_s": round(tot["vad_s"], 1),
            "dog_only_s": round(tot["dog_only_s"], 1),
            "dog_only_false_alarm": round(tot["dog_only_flagged_s"] / tot["dog_only_s"], 4),
            "unannotated_flagged_s": round(tot["unannotated_flagged_s"], 1),
        })
        for label, (flagged, total) in sorted(per_type[t].items(), key=lambda kv: -kv[1][1]):
            type_rows.append({"threshold": t, "dog_sound": label, "dog_only_s": round(total, 1),
                              "flagged_as_speech_s": round(flagged, 1), "false_alarm_rate": round(flagged / total, 4)})

    for name, rows in [("summary.csv", summary), ("false_alarms_by_dog_sound.csv", type_rows), ("per_session.csv", session_rows)]:
        with open(args.output_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (args.output_dir / "config.json").write_text(json.dumps({
        "dataset": "ELTE BARKS Lab Dog play pant (Zenodo 18972388), expert annotations",
        "n_sessions": len(sessions), "frame_s": FRAME_S, "precision_tolerance_s": 0.3,
        "dog_only_excludes_within_s_of_human": 0.5, "thresholds": args.thresholds,
    }, indent=2))

    logger.info("\n%-9s %7s %8s %9s %10s %12s", "threshold", "recall", "events", "precision", "dog FA", "unannot. s")
    for s in summary:
        logger.info("%-9.1f %6.1f%% %7.1f%% %8.1f%% %9.1f%% %12.1f", s["threshold"], 100 * s["recall"], 100 * s["event_recall"],
                    100 * s["precision"], 100 * s["dog_only_false_alarm"], s["unannotated_flagged_s"])
    logger.info("\nFalse alarms by dog sound (threshold 0.5):")
    for r in type_rows:
        if r["threshold"] == 0.5:
            logger.info("  %-12s %7.1f s of sound, %5.1f%% flagged as speech", r["dog_sound"], r["dog_only_s"], 100 * r["false_alarm_rate"])
    logger.info("\nSaved to %s", args.output_dir)


if __name__ == "__main__":
    main()
