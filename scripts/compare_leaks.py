"""Step 28: what exactly does each version let through, compared with muting?

For every attacker and version of an ``eval_dog_extraction`` output folder this
lists the human events where the attacker reads back MORE of its own original
transcript than from the muted version, and sorts them into

* non-lexical: laughter and backchannels only ("ha ha", "mm-hmm", "uh", ...);
* typical recognizer hallucinations only ("you", "thank you", "bye"): Whisper writes
  these on noise and silence, so they may or may not be real leaks;
* lexical: at least one other real word more than muting.

It also gives the difference in words read back against muting with a 95%
bootstrap interval over participants (17 dogs; 2000 resamples).

Writes ``<folder>/leaks_vs_mute.json``.

Usage (from inside speech_anonymization/):
    python -m scripts.compare_leaks outputs/dog_extraction/barkw10_mute_c06_w_w2tts_p3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from vad_anonymization.asr import normalize_text, words_recovered

NON_LEXICAL = re.compile(r"^(ha|haha|hah|he|hehe|hi|hm|hmm|mm|mmm|mhm|mm-hmm|mmhmm|uh|um|uhm|oh|ah|aha|eh|oops)$")
HALLUCINATION = {"you", "thank", "thanks", "bye"}


def recovered_words(orig: str, hyp: str) -> list[str]:
    """Words of ``orig`` that also occur in ``hyp`` (as multisets)."""
    left = normalize_text(hyp).split()
    out = []
    for w in normalize_text(orig).split():
        if w in left:
            left.remove(w)
            out.append(w)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    args = p.parse_args()
    folder = Path(args.folder)
    data = [json.loads(f.read_text()) for f in sorted((folder / "sessions").glob("*.json"))]
    events = [(d["session"].split("-S-")[0].split("-G-")[0], e) for d in data for e in d["events"]]
    asrs = sorted({k.split(":")[0] for _, e in events for k in e if ":" in k})
    rng = np.random.default_rng(0)
    res = {}
    for a in asrs:
        ref = [(pid, e) for pid, e in events if len(normalize_text(e.get(f"{a}:original", "")).split()) >= 2]
        versions = sorted({k.split(":", 1)[1] for _, e in ref for k in e if k.startswith(a + ":")} - {"original", "mute"})
        pids = sorted({pid for pid, _ in ref})
        for v in versions:
            if not all(f"{a}:{v}" in e and f"{a}:mute" in e for _, e in ref):
                continue
            lexical, non_lexical, halluc, diff = [], [], [], {pid: [] for pid in pids}
            for pid, e in ref:
                o = e[f"{a}:original"]
                rv, rm = words_recovered(o, e[f"{a}:{v}"]), words_recovered(o, e[f"{a}:mute"])
                diff[pid].append(rv - rm)
                if rv > rm:
                    extra = recovered_words(o, e[f"{a}:{v}"])
                    for w in recovered_words(o, e[f"{a}:mute"]):
                        if w in extra:
                            extra.remove(w)
                    item = {"id": e["id"], "heard": e[f"{a}:{v}"][:80], "extra_words": extra}
                    if all(NON_LEXICAL.match(w) for w in extra):
                        non_lexical.append(item)
                    elif all(NON_LEXICAL.match(w) or w in HALLUCINATION for w in extra):
                        halluc.append(item)
                    else:
                        lexical.append(item)
            per = np.array([np.mean(diff[pid]) for pid in pids])
            n = np.array([len(diff[pid]) for pid in pids])
            boots = []
            for _ in range(2000):
                k = rng.integers(0, len(pids), len(pids))
                boots.append(float((per[k] * n[k]).sum() / n[k].sum()))
            mean = float((per * n).sum() / n.sum())
            more = len(lexical) + len(non_lexical) + len(halluc)
            res[f"{a}:{v}"] = {"events_more_than_mute": more,
                               "lexical": len(lexical), "hallucination_prone": len(halluc), "non_lexical": len(non_lexical),
                               "words_read_back_minus_mute": round(mean, 4),
                               "ci95": [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)],
                               "lexical_examples": lexical[:10]}
            print(f"{a:9s} {v:24s} more than mute: {more:2d} "
                  f"(lexical {len(lexical)}, you/thank-you {len(halluc)}) | words read back {mean:+.3f} "
                  f"[{res[f'{a}:{v}']['ci95'][0]:+.3f}, {res[f'{a}:{v}']['ci95'][1]:+.3f}] "
                  f"{[x['heard'][:30] for x in lexical[:4]]}")
    (folder / "leaks_vs_mute.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
