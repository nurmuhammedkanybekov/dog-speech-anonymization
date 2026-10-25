"""Step 22c: a stronger attacker - Whisper large-v3-turbo instead of Whisper small.

All privacy numbers so far used Whisper small (244M parameters) as the
"attacker" that tries to read words back. A stronger recognizer might read
words that small cannot. Here the same 581 human-event windows (event +-0.3 s,
loudness-normalized, exactly as in Steps 8-21) are read by Whisper
large-v3-turbo (809M parameters, the 32-layer large-v3 encoder), in the
original and after the main pipeline settings.

* A window whose samples are all muted cannot contain speech; it is scored
  as empty without running the model.
* Floor: a background window of the same length, from the same session, at
  least 1 s away from any human event, for every third event (Whisper
  invents words on noise too).

Usage (from inside speech_anonymization/), after ``download_models``:
    python -m scripts.attack_stronger_asr --worker 0 --n-workers 2      # and --worker 1 (~2-3 h on 2 cores)
    python -m scripts.attack_stronger_asr --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from scripts.run_asr_real import CONTEXT_S, MAX_S, loud
from scripts.run_pipeline_real import frames_to_segments
from vad_anonymization.anonymize import vad_mute
from vad_anonymization.asr import make_asr, normalize_text, words_recovered
from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"
CONFIGS = ["vgg1_mute_c06", "eat40_1_mute_c06", "eat40_1_mute_c10", "obj40_mute_c10"]


def background_window(rng: random.Random, total: float, dur: float, human) -> tuple[float, float] | None:
    for _ in range(300):
        a = rng.uniform(0, max(0.0, total - dur))
        if all(a + dur <= e.start_s - 1.0 or a >= e.end_s + 1.0 for e in human):
            return a, a + dur
    return None


def n_words(t: str) -> int:
    return len(normalize_text(t).split())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="turbo")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--configs", nargs="*", default=[], help="more pipeline configs to attack (Step 25)")
    args = p.parse_args()
    out = OUT / f"attack_{args.model}"
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    cfg = {}
    for f in ("configs_v16.json", "configs_v17.json", "configs_v23.json", "configs_v25.json", "configs_v25_fine.json", "configs_v26.json"):
        if (OUT / f).exists():
            cfg.update(json.loads((OUT / f).read_text()))
    CONFIGS[:] = CONFIGS + [c for c in args.configs if c not in CONFIGS]
    cfg = {k: cfg[k] for k in CONFIGS}

    if not args.summarize_only:
        import torch

        torch.set_num_threads(1)
        asr = make_asr(args.model, num_threads=args.threads)
        from scripts.train_fine_detector import fine_segments
        from vad_anonymization.detection import SpeechSegment

        flags, fine = {}, {}
        for name, (_, th, _, path) in cfg.items():
            z = np.load(PROJECT_ROOT / "speech_anonymization" / path)
            arr = z["prob"]
            flags[name] = {}
            if "hop_s" in z:   # Step 25c: 20 ms predictions
                fine[name] = (float(z["pad_s"]), float(z["gap_s"]))
                for s, q in zip(sessions, np.split(arr, np.cumsum(z["lengths"])[:-1])):
                    flags[name][s.name] = q >= th
                continue
            i = 0
            for s in sessions:
                n = len(np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"])
                flags[name][s.name] = arr[i : i + n] >= th
                i += n
        t0 = time.time()
        for k, s in enumerate(sessions):
            f = out / "sessions" / f"{s.name}.json"
            done = json.loads(f.read_text()) if f.exists() else None
            if k % args.n_workers != args.worker or (done and all(c in e for e in done["events"][:1] for c in cfg)):
                continue
            x = load_audio(s.audio_path, SR)
            total = len(x) / SR
            centres = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
            versions = {"original": x}
            for name in cfg:
                if name in fine:
                    t = np.load(OUT / "whisper_enc" / f"{s.name}.npz")["times"]
                    segs = [SpeechSegment(a, b) for a, b in fine_segments(t, flags[name][s.name], total, *fine[name])]
                else:
                    segs = frames_to_segments(centres, flags[name][s.name], total)
                versions[name] = vad_mute(x, segs)
            cache: dict = {}
            rng = random.Random(zlib.crc32(s.name.encode()))
            events = []
            for j, e in enumerate(s.human):
                a_s = max(0.0, e.start_s - CONTEXT_S)
                b_s = min(total, e.end_s + CONTEXT_S, e.start_s - CONTEXT_S + MAX_S)
                a, b = int(a_s * SR), int(b_s * SR)
                if b - a < int(0.1 * SR):
                    continue
                row = {"id": f"{s.name}_{j:03d}", "duration_s": round(e.duration_s, 3)}
                prev = next((r for r in done["events"] if r["id"] == row["id"]), None) if done else None
                if prev:   # extending an earlier run: keep what was already transcribed
                    row.update(prev)
                for name, y in versions.items():
                    if name in row:
                        continue
                    seg = y[a:b]
                    if not torch.any(seg != 0):
                        row[name] = ""
                        continue
                    h = hashlib.md5(seg.numpy().tobytes()).hexdigest()
                    if h not in cache:
                        cache[h] = asr.transcribe(loud(seg))
                    row[name] = cache[h]
                if j % 3 == 0 and "background" not in row:
                    w = background_window(rng, total, b_s - a_s, s.human)
                    row["background"] = asr.transcribe(loud(x[int(w[0] * SR) : int(w[1] * SR)])) if w else None
                events.append(row)
            f.write_text(json.dumps({"session": s.name, "events": events}, ensure_ascii=False))
            print(f"{s.name}: {len(events)} events, {len(cache)} transcribed ({time.time() - t0:.0f}s)", flush=True)

    files = [out / "sessions" / f"{s.name}.json" for s in sessions]
    if not all(f.exists() and all(c in e for e in json.loads(f.read_text())["events"][:1] for c in CONFIGS) for f in files):
        print(f"{sum(f.exists() for f in files)}/{len(files)} sessions done - summary skipped")
        return
    ev = [e for f in files for e in json.loads(f.read_text())["events"]]
    small = {}
    for f in sorted((OUT / "pipeline_real_v16" / "sessions").glob("*.json")):
        for name, c in json.loads(f.read_text())["configs"].items():
            for e in c["events"]:
                small[(name, e["id"])] = e["after"]
    before_small = {r["id"]: r["before"] for r in csv.DictReader(open(OUT / "asr_real" / "per_event.csv", encoding="utf-8"))}
    rows = []
    for name in ["original"] + CONFIGS:
        readable = [n_words(e[name]) >= 2 for e in ev]
        ref = [e for e in ev if n_words(e["original"]) >= 2]
        rec = [words_recovered(e["original"], e[name]) for e in ref]
        s_text = [before_small[e["id"]] if name == "original" else small.get((name, e["id"]), "") for e in ev]
        rows.append({"version": name, "events": len(ev),
                     f"{args.model}_readable_2plus": round(float(np.mean(readable)), 3),
                     f"{args.model}_words_recovered": round(float(np.mean(rec)), 3) if rec else None,
                     "small_readable_2plus": round(float(np.mean([n_words(t) >= 2 for t in s_text])), 3)})
    bg = [e["background"] for e in ev if e.get("background") is not None]
    rows.append({"version": "background (no human)", "events": len(bg),
                 f"{args.model}_readable_2plus": round(float(np.mean([n_words(t) >= 2 for t in bg])), 3),
                 # chance level: words of the event's original transcript that also turn up in background audio
                 f"{args.model}_words_recovered": round(float(np.mean([
                     words_recovered(e["original"], e["background"]) for e in ev
                     if e.get("background") is not None and n_words(e["original"]) >= 2])), 3),
                 "small_readable_2plus": None})
    pairs = [("vgg1_mute_c06", "eat40_1_mute_c06"), ("eat40_1_mute_c10", "wenc_eat_dog_mute_c06")]
    for a_name, b_name in [(a, b) for a, b in pairs if a in CONFIGS and b in CONFIGS]:
        ra = [n_words(e[a_name]) >= 2 for e in ev]
        rb = [n_words(e[b_name]) >= 2 for e in ev]
        only_a, only_b = sum(x and not y for x, y in zip(ra, rb)), sum(y and not x for x, y in zip(ra, rb))
        rows.append({"version": f"McNemar {a_name} vs {b_name}", "events": only_a + only_b,
                     f"{args.model}_readable_2plus": f"{only_a} vs {only_b}",
                     f"{args.model}_words_recovered": round(binomtest(min(only_a, only_b), only_a + only_b).pvalue, 4),
                     "small_readable_2plus": None})
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    confirmed_speech(ev, small, before_small, out)


def n_recovered(reference: str, hypothesis: str | None) -> int:
    """How many words of ``reference`` appear again in ``hypothesis``."""
    if hypothesis is None or n_words(reference) == 0:
        return 0
    return round(words_recovered(reference, hypothesis) * n_words(reference))


def confirmed_speech(ev: list[dict], small: dict, before_small: dict, out: Path) -> None:
    """Leaks on events where BOTH recognizers agree there is speech.

    Turbo invents text on a quarter of the background windows, so "2+ words"
    alone overstates what an attacker gets. An event counts as confirmed
    speech when Whisper small and Whisper turbo both read 2+ words in the
    original and at least half of turbo's words are also in small's text.
    After muting, the event still leaks if either recognizer reads back at
    least 2 of the words it read in the original.
    """
    conf = [e for e in ev if n_words(e["original"]) >= 2 and n_words(before_small[e["id"]]) >= 2
            and words_recovered(e["original"], before_small[e["id"]]) >= 0.5]
    leaks = {}
    for name in CONFIGS:
        leaks[name] = [e for e in conf if max(n_recovered(e["original"], e[name]),
                                              n_recovered(before_small[e["id"]], small.get((name, e["id"])))) >= 2]
    rows = [{"version": name, "confirmed_events": len(conf), "still_leaking": len(l),
             "share": round(len(l) / max(1, len(conf)), 3), "ids": " ".join(e["id"] for e in l)}
            for name, l in leaks.items()]
    a, b = ({e["id"] for e in leaks[n]} for n in ("vgg1_mute_c06", "eat40_1_mute_c06"))
    only_a, only_b = len(a - b), len(b - a)
    rows.append({"version": "McNemar vgg1_mute_c06 vs eat40_1_mute_c06", "confirmed_events": only_a + only_b,
                 "still_leaking": f"{only_a} vs {only_b}", "share": None,
                 "ids": f"p = {binomtest(min(only_a, only_b), only_a + only_b).pvalue:.3f}" if only_a + only_b else ""})
    with open(out / "confirmed_speech.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"confirmed speech events (both recognizers read the same 2+ words): {len(conf)}")
    for r in rows:
        print(f"  {r['version']}: {r['still_leaking']} {r['share'] if r['share'] is not None else ''} {r['ids']}")


if __name__ == "__main__":
    main()
