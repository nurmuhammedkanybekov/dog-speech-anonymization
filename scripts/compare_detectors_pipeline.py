"""Step 16c: detect -> mute with three human detectors, in one run, at matched dog cost.

``run_pipeline_real --extra-configs outputs/configs_v16.json --output-dir outputs/pipeline_real_v16``
runs, on the same 581 annotated human events and with the same Whisper:

* ``vgg1``  - Step 9 detector (VGGish, session microphone only);
* ``eat1``  - EfficientAT logits, session microphone only, smoothed (Step 16);
* ``eat3``  - EfficientAT logits, all microphones, max + smoothed (Steps 15 + 16);

each at the thresholds where it flags 3 / 6 / 10 / 15 / 25% of the dog-only
frames. Reported per point: readable events (Whisper, 2+ words), share of the
original words still readable, dog time muted, and an exact McNemar test
against ``vgg1`` on the same events. Then the dog loss each detector needs to
reach a privacy level, with a participant bootstrap of the saving.

Usage (from inside speech_anonymization/):
    python -m scripts.compare_detectors_pipeline
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from scripts.compare_micfusion_pipeline import at_privacy
from vad_anonymization.asr import normalize_text, words_recovered

OUT = Path(__file__).resolve().parent.parent / "outputs"
COSTS = ["c03", "c06", "c10", "c15", "c25"]
DETECTORS = {"vgg1": ("VGGish, one microphone (Step 9)", "#52514e"),
             "eat1": ("EfficientAT mn10, one microphone", "#2a78d6"),
             "eat3": ("EfficientAT mn10, three microphones", "#eda100"),
             "eat40_1": ("EfficientAT mn40, one microphone", "#1baf7a"),     # Step 17 (configs_v17.json)
             "eat40_3": ("EfficientAT mn40, three microphones", "#eb6834"),
             "nested40": ("EfficientAT mn40, one microphone, nested thresholds", "#8a8985"),   # Step 19 (configs_v19.json)
             "aug40": ("EfficientAT mn40, one mic, + ESC-50 dogs as negatives", "#2a78d6"),    # Step 20 (configs_v20.json)
             "aug2_40": ("EfficientAT mn40, one mic, + other dogs and speech under barks", "#eb6834")}  # Step 21 (configs_v21.json)
COSTS_BY_DET = {"aug40": ["c06", "c10", "c15"], "aug2_40": ["c06", "c10", "c15"]}
CURVE_DETS = [d for d in DETECTORS if d not in COSTS_BY_DET]
PLOTTED = ["vgg1", "eat1", "eat40_1", "eat40_3"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def n_words(text: str) -> int:
    return len(normalize_text(text).split())


def load_sessions_json(run: Path) -> list[dict]:
    """Session results, with every word count recomputed from the stored transcripts.

    The counts saved at run time used the text normalization of that time; recounting
    applies the current one (Step 19: sound tags cut off at the end, like "(wind blowing",
    are no longer counted as words), so every run is scored the same way.
    """
    before = {r["id"]: r["before"] for r in csv.DictReader(open(OUT / "asr_real" / "per_event.csv", encoding="utf-8"))}
    sessions = [json.loads(f.read_text()) for f in sorted((run / "sessions").glob("*.json"))]
    for j in sessions:
        for c in j["configs"].values():
            for e in c["events"]:
                ref = before[e["id"]]
                e["words_before"], e["words_after"] = n_words(ref), n_words(e["after"])
                e["recovered"] = words_recovered(ref, e["after"]) if e["words_before"] else None
    return sessions


def baseline_rates() -> tuple[float, float]:
    """(unprocessed readable share, Whisper's share on background-only windows), recounted."""
    rows = list(csv.DictReader(open(OUT / "asr_real" / "per_event.csv", encoding="utf-8")))
    return (sum(n_words(r["before"]) >= 2 for r in rows) / len(rows),
            sum(n_words(r["control"]) >= 2 for r in rows) / len(rows))


def totals(sessions: list[dict], name: str, keep=None) -> np.ndarray:
    """(readable events, events, words recovered sum, events with readable original, dog muted s, dog s)"""
    t = np.zeros(6)
    for j in sessions:
        if keep is not None and j["session"][:4] not in keep:
            continue
        c = j["configs"][name]
        ev = c["events"]
        t += [sum(e["words_after"] >= 2 for e in ev), len(ev),
              sum(e["recovered"] for e in ev if e["words_before"] >= 2 and e["recovered"] is not None),
              sum(e["words_before"] >= 2 and e["recovered"] is not None for e in ev), c["dog_muted_s"], c["dog_s"]]
    return t


def main() -> None:
    run = OUT / "pipeline_real_v16"
    sessions = load_sessions_json(run)
    assert len(sessions) == 30, f"only {len(sessions)} sessions finished"
    original, floor = baseline_rates()   # 19.8% and 1.0% with the Step 19 counting (22.9% / 4.1% before)
    print(f"unprocessed readable {original:.3f}, background-only {floor:.3f}")
    readable = {}
    rows = []
    for det in DETECTORS:
        for c in COSTS_BY_DET.get(det, COSTS):
            name = f"{det}_mute_{c}"
            t = totals(sessions, name)
            readable[name] = {e["id"]: e["words_after"] >= 2 for j in sessions for e in j["configs"][name]["events"]}
            row = {"detector": det, "dog_frames_flagged": c, "readable_events": round(t[0] / t[1], 3),
                   "words_recovered": round(t[2] / t[3], 3), "dog_muted": round(t[4] / t[5], 3),
                   "n_events": int(t[1])}
            if det != "vgg1":
                b, m = readable[f"vgg1_mute_{c}"], readable[name]
                only_b = sum(b[i] and not m[i] for i in b)
                only_m = sum(m[i] and not b[i] for i in b)
                row["readable_only_vgg1"], row["readable_only_this"] = only_b, only_m
                row["mcnemar_p_vs_vgg1"] = round(binomtest(min(only_b, only_m), only_b + only_m).pvalue, 4) if only_b + only_m else 1.0
            row["mcnemar_p_vs_mn10"] = ""
            if det.startswith("eat40"):                 # larger vs smaller EfficientAT, same microphones
                b, m = readable[f"{det.replace('eat40_', 'eat')}_mute_{c}"], readable[name]
                only_b = sum(b[i] and not m[i] for i in b)
                only_m = sum(m[i] and not b[i] for i in b)
                row["mcnemar_p_vs_mn10"] = round(binomtest(min(only_b, only_m), only_b + only_m).pvalue, 4) if only_b + only_m else 1.0
            rows.append(row)
            print(row)
    out = OUT / "detectors_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[-1])
    rows = [{k: r.get(k, "") for k in fields} for r in rows]
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def curve(det, keep=None):
        pts = []
        for c in COSTS:
            t = totals(sessions, f"{det}_mute_{c}", keep)
            pts.append((t[4] / t[5], t[0] / t[1]))
        return pts

    equal = {f"{tg:.3f}": {d: (round(v, 3) if (v := at_privacy(curve(d), tg)) is not None else None) for d in CURVE_DETS}
             for tg in (0.06, 0.04, 0.03, 0.02)}
    parts = sorted({j["session"][:4] for j in sessions})
    per_part = {f"{d}_mute_{c}": {p_: totals(sessions, f"{d}_mute_{c}", {p_}) for p_ in parts}
                for d in CURVE_DETS for c in COSTS}
    rng = np.random.default_rng(0)
    boot = {}
    for tg in (0.04, 0.03):
        diffs = {d: [] for d in CURVE_DETS if d != "vgg1"}
        for _ in range(1000):
            keep = list(rng.choice(parts, len(parts), replace=True))
            def cv(det):  # a participant drawn twice counts twice
                pts = []
                for c in COSTS:
                    t = sum(per_part[f"{det}_mute_{c}"][p_] for p_ in keep)
                    pts.append((t[4] / t[5], t[0] / t[1]))
                return pts
            x0 = at_privacy(cv("vgg1"), tg)
            for d in diffs:
                x1 = at_privacy(cv(d), tg)
                if x0 is not None and x1 is not None:
                    diffs[d].append(x0 - x1)
        boot[f"{tg:.3f}"] = {d: {"median_dog_loss_saved": round(float(np.median(v)), 3),
                                 "ci95": [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)],
                                 "share_better": round(float((np.array(v) > 0).mean()), 3), "n_valid": len(v)}
                             for d, v in diffs.items()}
    (out / "summary.json").write_text(json.dumps({"dog_muted_needed_for_privacy": equal, "participant_bootstrap": boot,
                                                  "background_floor": floor}, indent=2))
    print(json.dumps(equal, indent=2))
    print(json.dumps(boot, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), facecolor=SURFACE)
    for k, ax in enumerate(axes):
        ax.set_facecolor(SURFACE)
        if k == 0:
            ax.axhspan(0, floor * 100, color=GRID, alpha=0.6, zorder=0)
            ax.text(39, 0.5, "shaded: Whisper on unprocessed background-only audio", fontsize=7, color=MUTED, ha="right")
        for det in PLOTTED:
            label, color = DETECTORS[det]
            pts = []
            for c in COSTS:
                t = totals(sessions, f"{det}_mute_{c}")
                pts.append((t[4] / t[5], t[0] / t[1] if k == 0 else t[2] / t[3]))
            xs, ys = zip(*sorted(pts))
            ax.plot(np.array(xs) * 100, np.array(ys) * 100, "-o", color=color, lw=1.8, ms=5, label=label, zorder=3)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 12 if k == 0 else 35)
        ax.set_xlabel("Dog-vocalization time muted, i.e. lost (%)", fontsize=9, color=MUTED)
        ax.set_ylabel("Human events where Whisper reads 2+ words (%)" if k == 0 else
                      "Original words still readable (%), events with 2+ words", fontsize=9, color=MUTED)
        ax.set_title([f"Readable events (unprocessed: {original * 100:.1f}%)", "Words still readable (unprocessed: 100%)"][k],
                     loc="left", fontsize=9.5, color=INK)
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    fig.suptitle("Detect-and-mute with different human detectors, no expert labels", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.905, "581 annotated human events, 17 participants each tested on models that never saw them; "
             "points = thresholds flagging 3 / 6 / 10 / 15 / 25% of dog-only frames", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "privacy_vs_dog_loss_detectors.png", dpi=200, facecolor=SURFACE)
    print(f"saved {out / 'privacy_vs_dog_loss_detectors.png'}")


if __name__ == "__main__":
    main()
