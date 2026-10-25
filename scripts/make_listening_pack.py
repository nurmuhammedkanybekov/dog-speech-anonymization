"""Step 27f: a listening check for a person - dog calls under speech, before and after.

Automatic recognizers are one kind of attacker; a person is another. This
writes a small folder to listen to: for dog calls that overlap a human sound
(the ones muting loses), three versions of the same few seconds:

* ``a_original.wav``  - the recording (contains a real voice: stays local, never in git);
* ``b_muted.wav``     - the standard anonymization (Step 26: the flagged time is silent);
* ``c_dog_kept.wav``  - Step 27: inside the flagged time, the dog extractor's output where the
  dog router hears a dog (``--keep-dog more``), silence elsewhere;
* ``d_all_room_tone.wav`` - Step 28: the extractor's output in ALL the flagged time, with room
  tone 10 dB above the room's own where the router hears no dog (``--keep-dog all``).

Each participant is processed by models that never heard them (detector: leave
one participant out; router: leave one participant out; extractor: the other
participant group). Only calls where the router lets something through are used. ``listen.html``
plays them side by side; ``answers.csv`` has two questions per clip for the
listener, for c and for d: can you understand any words? Can you hear the dog?

Usage (from inside speech_anonymization/):
    python -m scripts.make_listening_pack --n 12
    open outputs/listening_overlap/listen.html
"""

from __future__ import annotations

import argparse
import csv
import html
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.eval_dog_extraction import (OUT, PROJECT_ROOT, SR, blend, load_config, load_extractor, mask_from,
                                         run_extractor)
from scripts.finetune_gtcrn import participant_group
from scripts.run_pipeline_real import frames_to_segments
from vad_anonymization.audio_io import load_audio
from vad_anonymization.comfort_noise import room_tone
from vad_anonymization.elte_barks import load_sessions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="barkw10_mute_c06")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--mask-power", type=float, default=3.0)
    p.add_argument("--kind", default="w")
    p.add_argument("--tag", default="")
    p.add_argument("--router", default="outputs/human_detector_whisper/oof_dog_router_wenc_eat_w5.npz")
    p.add_argument("--route-share", type=float, default=0.05)
    args = p.parse_args()
    out = OUT / "listening_overlap"
    out.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")
    th, path = load_config(args.config)
    prob = np.load(PROJECT_ROOT / "speech_anonymization" / path)["prob"]
    z = np.load(PROJECT_ROOT / "speech_anonymization" / args.router)
    rprob = z["prob"]
    rth = float(np.quantile(rprob[(z["human"] >= 0.5) & (z["dog"] == 0)], 1 - args.route_share))
    cands, i = [], 0
    for s in sessions:
        c = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        pr = prob[i : i + len(c)]
        rp = rprob[i : i + len(c)]
        i += len(c)
        for e in s.dog:
            hs = [h for h in s.human if min(h.end_s, e.end_s) > max(h.start_s, e.start_s)]
            if hs:
                a = max(0.0, min([e.start_s] + [h.start_s for h in hs]) - 0.5)
                b = max([e.end_s] + [h.end_s for h in hs]) + 0.5
                routed = (rp >= rth) & (pr >= th) & (c > e.start_s - 0.12) & (c < e.end_s + 0.12)
                if b - a <= 6.0 and routed.any():
                    cands.append((s, e, a, b, c, pr, rp))
    random.Random(0).shuffle(cands)
    picked, used = [], set()
    for cand in cands:                     # random calls, at most one clip per session
        if cand[0].name not in used:
            picked.append(cand)
            used.add(cand[0].name)
        if len(picked) == args.n:
            break
    models, rows = {}, []
    for k, (s, e, a, b, c, pr, rp) in enumerate(sorted(picked, key=lambda t: t[0].name), 1):
        x = load_audio(s.audio_path, SR).numpy().astype(np.float32)
        total = len(x) / SR
        flagged = mask_from([(g.start_s, g.end_s) for g in frames_to_segments(c, pr >= th, total)], len(x))
        grp = participant_group(s.name)
        if grp not in models:
            models[grp] = load_extractor(grp, args.tag, args.kind)
        y = run_extractor(models[grp], args.kind, x, s.name, args.mask_power)
        ia, ib = int(a * SR), int(b * SR)
        route = mask_from([(cc - 0.12, cc + 0.12) for cc in c[rp >= rth]], len(x))
        versions = {"a_original": x, "b_muted": np.where(flagged, 0.0, x),
                    "c_dog_kept": blend(x, np.where(route, y, 0.0).astype(np.float32), flagged),
                    "d_all_room_tone": blend(x, y + (1 - np.clip(np.convolve(route.astype(np.float32), np.ones(160, np.float32) / 160,
                                                                             mode="same"), 0, 1)) * room_tone(x, flagged, 10.0, k),
                                             flagged)}
        d = out / f"clip_{k:02d}"
        d.mkdir(exist_ok=True)
        peak = max(1e-6, float(np.abs(x[ia:ib]).max()))
        for name, v in versions.items():
            sf.write(d / f"{name}.wav", (v[ia:ib] / peak * 0.9).astype(np.float32), SR)   # same gain for all versions
        rows.append({"clip": f"clip_{k:02d}", "session": s.name, "start_s": round(a, 2), "end_s": round(b, 2),
                     "dog_call": e.label, "words_understandable_in_c (yes/no/some)": "",
                     "dog_audible_in_c (yes/no/partly)": "", "words_understandable_in_d (yes/no/some)": "",
                     "dog_audible_in_d (yes/no/partly)": "", "notes": ""})
        print(rows[-1]["clip"], s.name, e.label, flush=True)
    with open(out / "answers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cells = "".join(
        f"<tr><td>{r['clip']}<br><small>{html.escape(r['session'])} {r['start_s']}-{r['end_s']} s, {html.escape(r['dog_call'])}</small></td>"
        + "".join(f"<td><audio controls preload='none' src='{r['clip']}/{v}.wav'></audio></td>"
                  for v in ("a_original", "b_muted", "c_dog_kept", "d_all_room_tone")) + "</tr>"
        for r in rows)
    (out / "listen.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Listening check: dog under speech</title>"
        "<style>body{font-family:sans-serif;margin:24px}td{padding:6px 10px;border-bottom:1px solid #ddd}</style>"
        "<h2>Dog calls under speech: original / muted / dog kept</h2>"
        "<p>For each clip, listen to <b>c</b> and <b>d</b>. Note in answers.csv: can you understand any words? "
        "Can you hear the dog? a = original (contains a real voice), b = muted (what the standard setting does), "
        "c = dog kept where the router hears a dog (--keep-dog more), d = dog kept in all the muted time, with "
        "room tone where the router hears no dog (--keep-dog all).</p>"
        f"<table><tr><th>clip</th><th>a original</th><th>b muted</th><th>c dog kept</th><th>d all + room tone</th></tr>{cells}</table>")
    print(f"wrote {out}/listen.html and answers.csv")


if __name__ == "__main__":
    main()
