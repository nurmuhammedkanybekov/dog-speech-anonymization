"""Step 28: how loud are the dog calls compared with the room tone?

``--room-tone G`` adds steady noise G dB above the room's own tone inside the
processed time. A dog call stays clearly audible over it when the call is well
above that level. For every annotated dog call this measures its mean power
relative to the session's room tone (the quietest 20% of the frames outside
the detector's flagged time, as ``vad_anonymization.comfort_noise``), and
reports per call type the median and the share of calls more than G + 6 dB
above the room tone, for G = 10 and 20.

Writes ``outputs/dog_extraction/room_tone_levels.json``.

Usage (from inside speech_anonymization/):
    python -m scripts.room_tone_levels
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.eval_dog_extraction import load_config, mask_from
from scripts.run_pipeline_real import frames_to_segments
from vad_anonymization.audio_io import load_audio
from vad_anonymization.comfort_noise import N_FFT, room_tone_spectrum
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"
GAINS = (10, 20)


def main() -> None:
    th, path = load_config("barkw10_mute_c06")
    prob = np.load(PROJECT_ROOT / "speech_anonymization" / path)["prob"]
    per_type: dict = defaultdict(list)
    i = 0
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        centres = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        pr = prob[i : i + len(centres)]
        i += len(centres)
        x = load_audio(s.audio_path, SR).numpy().astype(np.float32)
        flagged = mask_from([(g.start_s, g.end_s) for g in frames_to_segments(centres, pr >= th, len(x) / SR)], len(x))
        # room tone power per sample (one-sided frame power spectrum, Hann window: Parseval)
        win = np.hanning(N_FFT + 1)[:-1]
        tone = 2 * room_tone_spectrum(x, flagged).sum() / (win ** 2).sum() / N_FFT
        for e in s.dog:
            a, b = int(e.start_s * SR), int(e.end_s * SR)
            if b - a < SR // 20:
                continue
            p = float((x[a:b].astype(np.float64) ** 2).mean())
            per_type[e.label].append(10 * np.log10(p / tone + 1e-12))
    res = {}
    for label, v in sorted(per_type.items(), key=lambda kv: -len(kv[1])):
        v = np.array(v)
        res[label] = {"calls": len(v), "median_db_above_room_tone": round(float(np.median(v)), 1),
                      **{f"share_above_rt{g}+6dB": round(float((v > g + 6).mean()), 3) for g in GAINS}}
        print(label, res[label])
    allv = np.concatenate([np.array(v) for v in per_type.values()])
    res["all"] = {"calls": len(allv), "median_db_above_room_tone": round(float(np.median(allv)), 1),
                  **{f"share_above_rt{g}+6dB": round(float((allv > g + 6).mean()), 3) for g in GAINS}}
    print("all", res["all"])
    (OUT / "dog_extraction" / "room_tone_levels.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
