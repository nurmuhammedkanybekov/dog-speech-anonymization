"""Step 27b: does the dog extractor keep the dog under speech without letting the speech through?

On the real sessions, each participant processed by the extractor trained on
the OTHER participant group (Step 27a) and by the detector's leave-one-
participant-out predictions. Inside the time the detector flags, three
versions are compared:

* ``mute``     - silence (Steps 9-26);
* ``extract``  - the extractor's output (dog + background, human removed);
* ``oracle``   - the extractor's output only inside annotated dog calls,
  silence elsewhere: what a perfect "is there a dog?" decision would add.

Privacy: the attackers (``--asr``: Whisper small, Parakeet, ...) read the 581
human event windows of every version, as in Steps 8 and 25.

What is kept, measured on the real audio inside the flagged time:

* in human events with NO dog on them, the output should be near silence: the
  level of the output relative to the original (dB) is the real-data speech
  suppression;
* in dog calls that overlap a human sound (the 223 calls that muting loses),
  the level of the output relative to the original (dB): how much of that
  sound comes back (an upper bound for the dog, since it may contain speech);
* in dog-only calls inside flagged time (detector false alarms), the same:
  these are pure dog, so this measures how much dog the extractor keeps.

Usage (from inside speech_anonymization/), after train_dog_extractor for both groups:
    python -m scripts.eval_dog_extraction --config wenc_eat_dog_mute_c06 --asr small --worker 0 --n-workers 2
    python -m scripts.eval_dog_extraction --config wenc_eat_dog_mute_c06 --summarize-only
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

from scripts.finetune_gtcrn import participant_group
from scripts.run_asr_real import CONTEXT_S, MAX_S, loud
from scripts.run_pipeline_real import frames_to_segments
from scripts.train_dog_extractor import MODEL_DIR, DogExtractor, extract
from vad_anonymization.asr import make_asr, normalize_text, words_recovered
from vad_anonymization.audio_io import load_audio
from vad_anonymization.comfort_noise import adaptive_room_tone, room_tone
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"
CONFIG_FILES = ("configs_v17.json", "configs_v23.json", "configs_v25.json", "configs_v26.json")
VERSIONS = ("mute", "extract", "oracle")
RAMP = int(0.01 * SR)


def load_config(name: str) -> tuple[float, str]:
    cfg = {}
    for f in CONFIG_FILES:
        if (OUT / f).exists():
            cfg.update(json.loads((OUT / f).read_text()))
    return float(cfg[name][1]), cfg[name][3]


def load_extractor(group_of_session: str, tag: str, kind: str):
    other = "odd" if group_of_session == "even" else "even"      # trained on the other participants
    if kind == "w":
        from scripts.train_dog_extractor_w import load_w

        return load_w(MODEL_DIR / f"dog_extractor_w_{other}{tag}.pt")       # either architecture (Step 28: w2)
    else:
        m, f = DogExtractor(), MODEL_DIR / f"dog_extractor_{other}{tag}.pt"
    m.load_state_dict(torch.load(f, map_location="cpu")["model"])
    m.eval()
    return m


def run_extractor(model, kind: str, x: np.ndarray, session_name: str, power: float = 1.0) -> np.ndarray:
    if kind == "w":
        from scripts.train_dog_extractor_w import extract_w

        return extract_w(model, x, np.load(OUT / "whisper_enc" / f"{session_name}.npz")["feats"], power=power)
    return extract(model, x)


def mask_from(segs, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in segs:
        m[int(a * SR) : int(np.ceil(b * SR))] = True
    return m


def blend(x: np.ndarray, replacement: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Original outside the flagged time, the replacement inside, with 10 ms cross-fades."""
    g = (~flagged).astype(np.float32)
    k = np.ones(RAMP, np.float32) / RAMP
    g = np.clip(np.convolve(g, k, mode="same"), 0, 1)
    g[~flagged] = 1.0                                              # never touch audio outside the flagged time
    return g * x + (1 - g) * replacement


def level_db(y: np.ndarray, x: np.ndarray) -> float | None:
    px = float((x.astype(np.float64) ** 2).sum())
    return None if px <= 0 else 10 * np.log10(float((y.astype(np.float64) ** 2).sum()) / px + 1e-12)


def n_words(t: str) -> int:
    return len(normalize_text(t).split())


class DenoisedASR:
    """Adaptive attacker (Step 28): DeepFilterNet3 speech enhancement, then a recognizer.

    Strips steady noise (such as the room tone of ``--room-tone``) before reading."""

    def __init__(self, inner: str):
        from vad_anonymization.enhancers import DeepFilterNetEnhancer

        self.enh, self.asr = DeepFilterNetEnhancer(), make_asr(inner)

    def transcribe(self, wave: torch.Tensor) -> str:
        return self.asr.transcribe(loud(self.enh.enhance(wave)))


def attacker(name: str):
    return DenoisedASR(name.split("+", 1)[1]) if name.startswith("dfn+") else make_asr(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="wenc_eat_dog_mute_c06")
    p.add_argument("--tag", default="", help="extractor variant (file suffix)")
    p.add_argument("--kind", choices=["spec", "w"], default="w", help="w: Whisper-conditioned extractor (Step 27d)")
    p.add_argument("--mask-power", type=float, default=1.0, help="mask ** power: >1 = more conservative")
    p.add_argument("--asr", nargs="+", default=["small"])
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--router", default="", help="dog-presence predictions (npz with prob, human, dog on the 0.24 s grid)")
    p.add_argument("--router-name", default="routed", help="prefix of the routed versions' names")
    p.add_argument("--only", nargs="*", default=None, help="transcribe only these versions (e.g. for the slow turbo attacker)")
    p.add_argument("--room-tone", type=float, nargs="*", default=[],
                   help="Step 28: also versions with room-tone noise this many dB above the room's own, in the flagged time")
    p.add_argument("--adaptive-tone", type=float, nargs="*", default=[],
                   help="Step 28: versions 'extract_art<dB>': extractor output + room tone following the removed power (dB)")
    p.add_argument("--reuse-original", default="",
                   help="another output folder of this script: copy its transcripts of the original audio")
    p.add_argument("--route-share", type=float, nargs="*", default=[],
                   help="router thresholds: share of the human-only frames that are let through to the extractor")
    args = p.parse_args()
    torch.set_num_threads(1)
    out = OUT / "dog_extraction" / f"{args.config}_{args.kind}{args.tag}_p{args.mask_power:g}"
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(PROJECT_ROOT / "data" / "elte_barks")

    if not args.summarize_only:
        th, path = load_config(args.config)
        prob = np.load(PROJECT_ROOT / "speech_anonymization" / path)["prob"]
        asrs = {a: attacker(a) for a in args.asr}
        router_th, rprob = {}, None
        if args.router:
            z = np.load(PROJECT_ROOT / "speech_anonymization" / args.router)
            rprob = z["prob"]
            human_only = (z["human"] >= 0.5) & (z["dog"] == 0)
            router_th = {f"{args.router_name}{q:g}": float(np.quantile(rprob[human_only], 1 - q)) for q in args.route_share}
        models: dict = {}
        i, t0 = 0, time.time()
        for k, s in enumerate(sessions):
            centres = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
            pr = prob[i : i + len(centres)]
            rp = rprob[i : i + len(centres)] if rprob is not None else None
            i += len(centres)
            f = out / "sessions" / f"{s.name}.json"
            res = json.loads(f.read_text()) if f.exists() else None
            rt_names = [f"{v}_rt{g:g}" for g in args.room_tone for v in ("mute", "extract", *router_th)]
            rt_names += [f"extract_art{g:g}" for g in args.adaptive_tone]
            rt_names += [f"{v}_hyb{g:g}" for g in args.room_tone for v in router_th]
            wanted = tuple(v for v in ("original", *VERSIONS, *router_th, *rt_names) if args.only is None or v in args.only)
            if k % args.n_workers != args.worker or (res and all(f"{a}:{v}" in res["events"][0] for a in args.asr for v in wanted)
                                                     if res and res["events"] else res is not None):
                continue
            x = load_audio(s.audio_path, SR).numpy().astype(np.float32)
            n, total = len(x), len(x) / SR
            flagged = mask_from([(g.start_s, g.end_s) for g in frames_to_segments(centres, pr >= th, total)], n)
            grp = participant_group(s.name)
            if grp not in models:
                models[grp] = load_extractor(grp, args.tag, args.kind)
            y = run_extractor(models[grp], args.kind, x, s.name, args.mask_power)
            dogm = mask_from([(e.start_s, e.end_s) for e in s.dog], n)
            versions = {"mute": np.where(flagged, 0.0, x).astype(np.float32),
                        "extract": blend(x, y, flagged),
                        "oracle": blend(x, np.where(dogm, y, 0.0).astype(np.float32), flagged)}
            humm = mask_from([(e.start_s, e.end_s) for e in s.human], n)
            routed_stats = {}
            for v, th_r in router_th.items():         # extract only where the router hears a dog; mute elsewhere
                route = mask_from([(c - 0.12, c + 0.12) for c in centres[rp >= th_r]], n)
                versions[v] = blend(x, np.where(route, y, 0.0).astype(np.float32), flagged)
                routed_stats[v] = {"dog_under_human_s": float((flagged & dogm & humm).sum()) / SR,
                                   "dog_under_human_kept_s": float((flagged & dogm & humm & route).sum()) / SR,
                                   "dog_only_flagged_s": float((flagged & dogm & ~humm).sum()) / SR,
                                   "dog_only_flagged_kept_s": float((flagged & dogm & ~humm & route).sum()) / SR,
                                   "human_no_dog_s": float((flagged & humm & ~dogm).sum()) / SR,
                                   "human_no_dog_routed_s": float((flagged & humm & ~dogm & route).sum()) / SR}
            if res is None:
                # levels inside the flagged time (no ASR needed)
                lv = {"human_no_dog": [], "dog_under_human": [], "dog_only_flagged": []}
                for e in s.human:
                    if not any(min(d.end_s, e.end_s) > max(d.start_s, e.start_s) for d in s.dog):
                        a, b = int(e.start_s * SR), int(e.end_s * SR)
                        sel = flagged[a:b]
                        if sel.any():
                            lv["human_no_dog"].append((float((x[a:b][sel] ** 2).sum()), float((y[a:b][sel] ** 2).sum())))
                for e in s.dog:
                    a, b = int(e.start_s * SR), int(e.end_s * SR)
                    sel = flagged[a:b]
                    if not sel.any():
                        continue
                    over = any(min(h.end_s, e.end_s) > max(h.start_s, e.start_s) for h in s.human)
                    key = "dog_under_human" if over else "dog_only_flagged"
                    lv[key].append((float((x[a:b][sel] ** 2).sum()), float((y[a:b][sel] ** 2).sum()), e.label))
                res = {"session": s.name, "levels": lv, "events": [], "asr": []}
                prev = Path(args.reuse_original) / "sessions" / f"{s.name}.json" if args.reuse_original else None
                prev_ev = {e["id"]: e for e in json.loads(prev.read_text())["events"]} if prev and prev.exists() else {}
                for j, e in enumerate(s.human):
                    a_s = max(0.0, e.start_s - CONTEXT_S)
                    b_s = min(total, e.end_s + CONTEXT_S, e.start_s - CONTEXT_S + MAX_S)
                    if b_s - a_s >= 0.1:
                        ev_new = {"id": f"{s.name}_{j:03d}", "a": a_s, "b": b_s}
                        old = prev_ev.get(ev_new["id"])
                        if old and abs(old["a"] - a_s) < 1e-6 and abs(old["b"] - b_s) < 1e-6:
                            ev_new.update({k: v for k, v in old.items() if k.endswith(":original")})
                        res["events"].append(ev_new)
            seed = int(hashlib.md5(s.name.encode()).hexdigest()[:8], 16)
            tone_lv = {}
            for g in args.adaptive_tone:              # Step 28: room tone that follows what the extractor removed
                noise = adaptive_room_tone(x, y, flagged, rel_db=g, seed=seed)
                versions[f"extract_art{g:g}"] = blend(x, y + noise, flagged)
                tone_lv[f"extract_art{g:g}"] = noise
            for g in args.room_tone:                  # Step 28: room tone under the processed time
                noise = room_tone(x, flagged, g, seed)
                versions[f"mute_rt{g:g}"] = blend(x, noise, flagged)
                versions[f"extract_rt{g:g}"] = blend(x, y + noise, flagged)
                tone_lv[f"extract_rt{g:g}"] = noise
                for v in router_th:
                    route = mask_from([(c - 0.12, c + 0.12) for c in centres[rp >= router_th[v]]], n)
                    versions[f"{v}_rt{g:g}"] = blend(x, np.where(route, y, 0.0).astype(np.float32) + noise, flagged)
                    # hybrid: extractor output everywhere; room tone only where the router hears no dog (10 ms fades)
                    soft = np.clip(np.convolve(route.astype(np.float32), np.ones(RAMP, np.float32) / RAMP, mode="same"), 0, 1)
                    versions[f"{v}_hyb{g:g}"] = blend(x, y + (1 - soft) * noise, flagged)
                    tone_lv[f"{v}_hyb{g:g}"] = (1 - soft) * noise
            # dog calls in the flagged time: extractor output vs the added noise (energy), per call
            for v, noise in tone_lv.items():
                rows_ = []
                for e in s.dog:
                    a, b = int(e.start_s * SR), int(e.end_s * SR)
                    sel = flagged[a:b]
                    if sel.any():
                        over = any(min(h.end_s, e.end_s) > max(h.start_s, e.start_s) for h in s.human)
                        rows_.append(["dog_under_human" if over else "dog_only_flagged", e.label,
                                      float((y[a:b][sel].astype(np.float64) ** 2).sum()),
                                      float((noise[a:b][sel].astype(np.float64) ** 2).sum())])
                res.setdefault("tone_levels", {})[v] = rows_
            res.setdefault("routed", {}).update(routed_stats)
            for name, asr in asrs.items():
                cache: dict = {}
                for ev in res["events"]:
                    a, b = int(ev["a"] * SR), int(ev["b"] * SR)
                    for v in wanted:
                        if f"{name}:{v}" in ev:
                            continue
                        seg = x[a:b] if v == "original" else versions[v][a:b]
                        if not np.any(seg != 0):
                            ev[f"{name}:{v}"] = ""
                            continue
                        h = hashlib.md5(seg.tobytes()).hexdigest()
                        if h not in cache:
                            cache[h] = asr.transcribe(loud(torch.from_numpy(seg.copy())))
                        ev[f"{name}:{v}"] = cache[h]
                if name not in res["asr"]:
                    res["asr"].append(name)
            f.write_text(json.dumps(res, ensure_ascii=False))
            print(f"{s.name}: {len(res['events'])} events ({time.time() - t0:.0f}s)", flush=True)

    files = [out / "sessions" / f"{s.name}.json" for s in sessions]
    if not all(f.exists() for f in files):
        print(f"{sum(f.exists() for f in files)}/{len(files)} sessions done - summary skipped")
        return
    data = [json.loads(f.read_text()) for f in files]
    asr_names = sorted(set.intersection(*[set(d["asr"]) for d in data]))
    rows = []
    all_versions = sorted({k.split(":", 1)[1] for d in data for e in d["events"] for k in e if ":" in k},
                          key=lambda v: (v != "original", v))
    for name in asr_names:
        ev = [e for d in data for e in d["events"]]
        for v in [v for v in all_versions if all(f"{name}:{v}" in e for e in ev)]:
            ref = [e for e in ev if n_words(e[f"{name}:original"]) >= 2]
            rows.append({"asr": name, "version": v,
                         "readable_2plus": round(float(np.mean([n_words(e[f"{name}:{v}"]) >= 2 for e in ev])), 4),
                         "words_read_back": round(float(np.mean([words_recovered(e[f"{name}:original"], e[f"{name}:{v}"]) for e in ref])), 4)})
    levels = {}
    for key in ("human_no_dog", "dog_under_human", "dog_only_flagged"):
        items = [it for d in data for it in d["levels"][key]]
        xs, ys = sum(it[0] for it in items), sum(it[1] for it in items)
        per = [10 * np.log10(it[1] / it[0] + 1e-12) for it in items if it[0] > 0]
        levels[key] = {"events": len(items), "total_level_db": round(10 * np.log10(ys / xs + 1e-12), 2) if xs else None,
                       "median_event_level_db": round(float(np.median(per)), 2) if per else None}
    routed = {}
    for v in sorted({v for d in data for v in d.get("routed", {})}):
        have = [d for d in data if v in d.get("routed", {})]
        if all(v in d.get("routed", {}) or not d["events"] for d in data):   # sessions without human events: nothing flagged
            t = {k: sum(d["routed"][v][k] for d in have) for k in have[0]["routed"][v]}
            routed[v] = {"dog_under_human_kept": round(t["dog_under_human_kept_s"] / t["dog_under_human_s"], 3),
                         "dog_only_flagged_kept": round(t["dog_only_flagged_kept_s"] / t["dog_only_flagged_s"], 3),
                         "human_no_dog_routed": round(t["human_no_dog_routed_s"] / t["human_no_dog_s"], 3),
                         "dog_under_human_s": round(t["dog_under_human_s"], 1)}
    tone = {}
    for v in sorted({v for d in data for v in d.get("tone_levels", {})}):
        items = [it for d in data for it in d.get("tone_levels", {}).get(v, [])]
        tone[v] = {}
        for kind in ("dog_under_human", "dog_only_flagged"):
            for grp, test in (("pant", lambda l: l == "Pant"), ("other calls", lambda l: l != "Pant")):
                r = np.array([10 * np.log10(it[2] / it[3] + 1e-12) for it in items if it[0] == kind and test(it[1]) and it[3] > 0])
                if len(r):
                    tone[v][f"{kind}, {grp}"] = {"calls": len(r), "median_dog_to_noise_db": round(float(np.median(r)), 1),
                                                 "share_6db_above_noise": round(float((r > 6).mean()), 3)}
    summary = {"config": args.config, "tag": args.tag, "privacy": rows, "levels_in_flagged_time": levels,
               "routing": routed, "dog_vs_added_noise": tone}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
