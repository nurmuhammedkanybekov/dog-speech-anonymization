"""Step 15e: the no-label pipeline with the three-microphone detector vs the Step 9 detector.

Both detectors drive the same "detect -> mute" pipeline (Step 12) on the same
581 annotated human events; every model is leave-one-participant-out. Each
detector is run at the thresholds where it flags 3 / 6 / 10 / 15 / 25% of
the dog-only frames, so at every point the two cost about the same dog sound.

Reported: Whisper privacy (share of events with 2+ readable words) and dog
time actually muted, per point; an exact McNemar test per matched pair
(same events, readable or not); and the dog loss each detector needs to reach
a given privacy level (linear interpolation along its curve).

Usage (from inside speech_anonymization/), after run_pipeline_real with
--extra-configs outputs/configs_micfusion.json / configs_baseline_curve.json:
    python -m scripts.compare_micfusion_pipeline
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

OUT = Path(__file__).resolve().parent.parent / "outputs"
COSTS = ["c03", "c06", "c10", "c15", "c25"]
BASELINE_NAMES = {"c03": "mute_c03", "c06": "mute_strict", "c10": "mute_c10", "c15": "mute_c15", "c25": "mute_balanced"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GREEN, GOLD = "#1baf7a", "#eda100"


def load(dirs: list[Path]) -> tuple[dict, dict]:
    summary, events = {}, {}
    for d in dirs:
        for r in csv.DictReader(open(d / "summary.csv")):
            summary.setdefault(r["config"], r)
        for r in csv.DictReader(open(d / "per_event.csv", encoding="utf-8")):
            events.setdefault(r["config"], {})[r["id"]] = int(r["words_after"]) >= 2
    return summary, events


def at_privacy(curve: list[tuple[float, float]], target: float) -> float | None:
    """Dog loss (x) where the curve (x ascending, y = readable share) first reaches y <= target."""
    pts = sorted(curve)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 > target >= y1:
            return x0 + (x1 - x0) * (y0 - target) / (y0 - y1)
    return pts[0][0] if pts and pts[0][1] <= target else None


def per_participant(dirs: list[Path], configs: list[str]) -> dict:
    """participant -> config -> (readable events, events, dog-only seconds muted, dog-only seconds)."""
    res: dict = {}
    done: set = set()
    for d in dirs:
        for f in sorted((d / "sessions").glob("*.json")):
            j = json.loads(f.read_text())
            for name, c in j["configs"].items():
                if name not in configs or (j["session"], name) in done:   # first directory wins
                    continue
                done.add((j["session"], name))
                acc = res.setdefault(j["session"][:4], {}).setdefault(name, np.zeros(4))
                acc += [sum(e["words_after"] >= 2 for e in c["events"]), len(c["events"]), c["dog_muted_s"], c["dog_s"]]
    return res


def bootstrap_equal_privacy(res: dict, target: float, n_boot: int = 2000, seed: int = 0) -> dict:
    """Resample participants; dog loss each detector needs to reach `target` readable events."""
    rng = np.random.default_rng(seed)
    parts = sorted(res)
    diffs = []
    for _ in range(n_boot):
        sample = rng.choice(parts, len(parts), replace=True)
        x = {}
        for label, names in (("one", [BASELINE_NAMES[c] for c in COSTS]), ("three", [f"mf_mute_{c}" for c in COSTS])):
            curve = []
            for nm in names:
                t = sum(res[p_][nm] for p_ in sample)
                curve.append((t[2] / t[3], t[0] / t[1]))
            x[label] = at_privacy(curve, target)
        if x["one"] is not None and x["three"] is not None:
            diffs.append(x["one"] - x["three"])
    diffs = np.array(diffs)
    return {"target_readable": target, "n_valid": int(len(diffs)), "median_dog_loss_saved": round(float(np.median(diffs)), 3),
            "ci95": [round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)],
            "share_three_mics_better": round(float((diffs > 0).mean()), 3)}


def main() -> None:
    summary, events = load([OUT / "pipeline_real", OUT / "pipeline_real_baseline_curve", OUT / "pipeline_real_micfusion"])
    rows = []
    curves = {"one microphone (Step 9)": [], "three microphones (Step 15)": []}
    for c in COSTS:
        b, m = BASELINE_NAMES[c], f"mf_mute_{c}"
        if b not in summary or m not in summary:
            continue
        eb, em = events[b], events[m]
        ids = sorted(set(eb) & set(em))
        only_b = sum(eb[i] and not em[i] for i in ids)   # readable with one mic, not with three
        only_m = sum(em[i] and not eb[i] for i in ids)
        p = binomtest(min(only_b, only_m), only_b + only_m, 0.5).pvalue if only_b + only_m else 1.0
        row = {"dog_frames_flagged_target": c.replace("c", "0.").replace("0.25", "0.249"),
               "baseline_config": b, "baseline_readable": float(summary[b]["share_2plus_words"]),
               "baseline_dog_muted": float(summary[b]["dog_muted"]),
               "baseline_events_covered": float(summary[b]["human_events_covered"]),
               "micfusion_config": m, "micfusion_readable": float(summary[m]["share_2plus_words"]),
               "micfusion_dog_muted": float(summary[m]["dog_muted"]),
               "micfusion_events_covered": float(summary[m]["human_events_covered"]),
               "readable_only_baseline": only_b, "readable_only_micfusion": only_m, "mcnemar_p": round(p, 4)}
        rows.append(row)
        curves["one microphone (Step 9)"].append((row["baseline_dog_muted"], row["baseline_readable"]))
        curves["three microphones (Step 15)"].append((row["micfusion_dog_muted"], row["micfusion_readable"]))
    out = OUT / "micfusion_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # pooled McNemar over all five matched pairs (the same events appear five times: descriptive only)
    floor = float(summary["background floor (no human)"]["share_2plus_words"])
    equal_privacy = {}
    for target in (0.08, 0.06, 0.05, floor):
        equal_privacy[f"{target:.3f}"] = {k: (round(v, 3) if (v := at_privacy(c, target)) is not None else None)
                                          for k, c in curves.items()}
    names = [BASELINE_NAMES[c] for c in COSTS] + [f"mf_mute_{c}" for c in COSTS]
    res = per_participant([OUT / "pipeline_real", OUT / "pipeline_real_baseline_curve", OUT / "pipeline_real_micfusion"], names)
    boot = [bootstrap_equal_privacy(res, t) for t in (0.08, 0.06, 0.05)]
    for b in boot:
        print("participant bootstrap:", b)
    (out / "summary.json").write_text(json.dumps({"rows": rows, "dog_muted_needed_for_privacy": equal_privacy,
                                                  "participant_bootstrap": boot, "background_floor": floor}, indent=2))
    for r in rows:
        print(r)
    print(json.dumps(equal_privacy, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(0, floor * 100, color=GRID, alpha=0.6, zorder=0)
    ax.text(21, floor * 100 / 2, "Whisper's own hallucination level (no human present)", fontsize=7.5, color=MUTED, va="center")
    orig = float(summary["original (no processing)"]["share_2plus_words"]) * 100
    ax.axhline(orig, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(21, orig + 0.5, f"no processing: {orig:.1f}%", fontsize=7.5, color=MUTED)
    for (name, curve), color in zip(curves.items(), [GREEN, GOLD]):
        xs, ys = zip(*sorted(curve))
        ax.plot(np.array(xs) * 100, np.array(ys) * 100, "-o", color=color, lw=1.8, ms=5, label=name, zorder=3)
    for r in rows:
        if r["mcnemar_p"] < 0.05:
            ax.annotate(f"p={r['mcnemar_p']:.3f}", (r["micfusion_dog_muted"] * 100, r["micfusion_readable"] * 100),
                        (6, -10), textcoords="offset points", fontsize=7, color=GOLD)
    x_one = equal_privacy["0.060"]["one microphone (Step 9)"]
    x_three = equal_privacy["0.060"]["three microphones (Step 15)"]
    if x_one and x_three:
        ax.annotate("", (x_three * 100, 6.0), (x_one * 100, 6.0),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.0), zorder=4)
        ax.text(x_one * 100 + 0.6, 7.4, f"same privacy (6%):\n{x_one * 100:.0f}% -> {x_three * 100:.0f}% of dog time lost",
                fontsize=7.5, ha="left", va="center", color=INK)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 25)
    ax.set_xlabel("Dog-vocalization time muted, i.e. lost (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("Human events where Whisper reads 2+ words (%)", fontsize=9, color=MUTED)
    fig.suptitle("Detect-and-mute with one vs three microphones, no expert labels", x=0.01, ha="left", fontsize=12, color=INK)
    ax.set_title("581 annotated human events, 30 sessions; each point = threshold flagging 3/6/10/15/25% of dog-only frames",
                 loc="left", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.82), labelcolor=INK)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    fig.tight_layout()
    fig.savefig(out / "privacy_vs_dog_loss_micfusion.png", dpi=200, facecolor=SURFACE)
    print(f"saved {out / 'privacy_vs_dog_loss_micfusion.png'}")


if __name__ == "__main__":
    main()
