"""Step 25: how sure are we that the new detector is better? Participant bootstrap.

Resamples the 17 participants with replacement (2,000 times) and recomputes,
for the old and new standard settings, the share of readable events and of
words still readable (Whisper small, recounted) and the dog-only time lost.
Reports 95% intervals of each and of the differences, and the share of
resamples where the new setting is at least as private AND loses less dog.

Usage (from inside speech_anonymization/):
    python -m scripts.bootstrap_step25
"""

from __future__ import annotations

import json

import numpy as np

from scripts.compare_detectors_pipeline import OUT, load_sessions_json

PAIRS = [("eat40_1_mute_c10", "wenc_eat_dog_mute_c06"),   # old standard vs new standard
         ("eat40_1_mute_c06", "wenc_eat_dog_mute_c03"),   # old light vs new light
         ("eat40_1_mute_c15", "wenc_eat_dog_mute_c10")]   # old strict vs new strict


def per_participant(sessions, name):
    """participant -> [readable events, events, words recovered sum, readable originals, dog muted s, dog s]"""
    out = {}
    for j in sessions:
        c = j["configs"][name]
        ev = c["events"]
        v = np.array([sum(e["words_after"] >= 2 for e in ev), len(ev),
                      sum(e["recovered"] for e in ev if e["words_before"] >= 2 and e["recovered"] is not None),
                      sum(e["words_before"] >= 2 and e["recovered"] is not None for e in ev), c["dog_muted_s"], c["dog_s"]])
        out[j["session"][:4]] = out.get(j["session"][:4], 0) + v
    return out


def rates(t):
    return t[0] / t[1], t[2] / max(t[3], 1), t[4] / t[5]


def main() -> None:
    sessions = load_sessions_json(OUT / "pipeline_real_v16")
    rng = np.random.default_rng(0)
    parts = sorted({j["session"][:4] for j in sessions})
    res = {}
    for old, new in PAIRS:
        a, b = per_participant(sessions, old), per_participant(sessions, new)
        point = {"old": rates(sum(a.values())), "new": rates(sum(b.values()))}
        draws = []
        for _ in range(2000):
            pick = rng.choice(parts, len(parts))
            ra, rb = rates(sum(a[p] for p in pick)), rates(sum(b[p] for p in pick))
            draws.append([*ra, *rb])
        d = np.array(draws)
        ci = lambda x: [round(float(np.percentile(x, 2.5)) * 100, 2), round(float(np.percentile(x, 97.5)) * 100, 2)]
        res[f"{old} vs {new}"] = {
            "old (readable, words, dog lost) %": [round(x * 100, 2) for x in point["old"]],
            "new (readable, words, dog lost) %": [round(x * 100, 2) for x in point["new"]],
            "readable diff new-old, 95% CI": ci(d[:, 3] - d[:, 0]),
            "words diff new-old, 95% CI": ci(d[:, 4] - d[:, 1]),
            "dog lost diff new-old, 95% CI": ci(d[:, 5] - d[:, 2]),
            "dog lost relative change, 95% CI": ci(d[:, 5] / d[:, 2] - 1),
            "share of resamples: new as private (words) and less dog lost": round(float(np.mean((d[:, 4] <= d[:, 1]) & (d[:, 5] < d[:, 2]))), 3)}
    (OUT / "detectors_pipeline" / "bootstrap_step25.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
