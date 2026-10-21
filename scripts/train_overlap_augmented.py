"""Step 21a: teach the detector that speech under a bark is still speech.

Step 20c fixed the over-muting of unfamiliar dogs by adding dog-only scenes
as "not human" examples. But in scenes where speech and barks overlap, that
detector stopped muting the speech under the barks: it had only seen other
dogs as negatives, never a dog on top of a person. This adds the missing
case. Synthetic scenes with speech AND dogs (simulated rooms, noise) are
added to training, and every frame is labelled by the TRUE speech activity:

* speech for at least half of the frame -> human (also when a dog barks over it);
* no speech at all -> not human (the dog, the room);
* in between -> left out, as for ELTE.

Everything that is tested is held out from training: speakers are split by
person (LibriSpeech: 20 speakers for training, 20 others for testing), and
dog clips are split in half (ESC-50: 20 / 20), as in Step 20c.

Compared (thresholds always set on ELTE, leave one participant out):

* ``elte_only``         - Steps 17-19;
* ``dogs_negative``     - Step 20c (ELTE + dog-only scenes);
* ``dogs_and_overlap``  - ELTE + dog-only scenes + speech-and-dog scenes (this step).

Scored on 10 ms grids: ELTE human time found; on held-out scenes: unseen-dog
time muted (dog-only scenes); speech time muted, speech-under-bark time
muted, and dog-only time muted (mixed scenes).

Usage (from inside speech_anonymization/):
    python -m scripts.train_overlap_augmented --save-final     # also writes models/human_detector_eat40_aug2.npz
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from scripts.anonymize_recording import EAT_DIR, flagged_segments
from scripts.eval_out_of_domain import PROJECT_ROOT, SCENE_S, make_scene, mask_from
from scripts.extract_efficientat import embed, load_model
from scripts.train_final_detector import LEVELS
from scripts.train_human_detector import HOP_S, overlap_share
from scripts.train_multimic_detector import OUT, roles
from vad_anonymization.elte_barks import load_sessions

COSTS = {"light": 0.06, "standard": 0.10, "strict": 0.15}
N_FRAMES = int((SCENE_S - 0.96) // HOP_S) + 1
CENTRES = 0.48 + HOP_S * np.arange(N_FRAMES)


def split_speakers(files: list[Path]) -> tuple[list[Path], list[Path]]:
    speakers = sorted({f.name.split("-")[0] for f in files})
    train = set(speakers[0::2])
    return [f for f in files if f.name.split("-")[0] in train], [f for f in files if f.name.split("-")[0] not in train]


def frame_share(act10ms: np.ndarray) -> np.ndarray:
    """share of each 0.24 s frame (centre +- 0.12 s) that is active, from a 10 ms activity track"""
    return np.array([act10ms[max(0, int((c - 0.12) * 100)) : int((c + 0.12) * 100)].mean() for c in CENTRES])


def build(seeds, counts, speech_files, dog_files, transcripts, eat, cache: Path):
    if cache.exists():
        z = np.load(cache)
        return z["logits"], z["speech"], z["dog"]
    L, S, D = [], [], []
    for k in seeds:
        mix, sp, dg, _ = make_scene(k, speech_files, transcripts, dog_files, *counts)
        L.append(embed(torch.from_numpy(mix), eat[0], eat[1], N_FRAMES)[1])
        S.append(sp)
        D.append(dg)
    np.savez_compressed(cache, logits=np.stack(L), speech=np.stack(S), dog=np.stack(D))
    return np.stack(L), np.stack(S), np.stack(D)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-final", action="store_true")
    args = p.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(2)
    out = OUT / "overlap_augmented"
    out.mkdir(parents=True, exist_ok=True)
    speech_dir, dog_dir = PROJECT_ROOT / "data" / "librispeech_subset", PROJECT_ROOT / "data" / "esc50_dog"
    transcripts = dict(line.strip().split(" ", 1) for line in open(speech_dir / "transcripts.txt") if line.strip())
    tr_speech, te_speech = split_speakers(sorted(speech_dir.glob("*.flac")))
    dogs = sorted(dog_dir.glob("*.wav"))
    tr_dogs, te_dogs = dogs[0::2], dogs[1::2]
    with contextlib.redirect_stdout(io.StringIO()):
        eat = load_model(EAT_DIR, "mn40_as")[:2]

    aug_dog = build(range(5000, 5060), (0, 5), tr_speech, tr_dogs, transcripts, eat, out / "train_dog_scenes.npz")
    aug_mix = build(range(7000, 7060), (3, 4), tr_speech, tr_dogs, transcripts, eat, out / "train_mixed_scenes.npz")
    te_dog = build(range(1000, 1040), (0, 5), te_speech, te_dogs, transcripts, eat, out / "test_dog_scenes.npz")
    te_mix = build(range(3000, 3040), (3, 4), te_speech, te_dogs, transcripts, eat, out / "test_mixed_scenes.npz")

    def frames(scenes, speech_label: bool):
        L, S, _ = scenes
        X = np.vstack(L).astype(np.float32)
        y = np.concatenate([frame_share(s) for s in S]) if speech_label else np.zeros(len(X))
        return X, y

    Xd, yd = frames(aug_dog, False)
    Xm, ym = frames(aug_mix, True)

    meta = []
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        c = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        X = np.load(OUT / "efficientat_mn40" / f"{s.name}_{roles(s.name)[0]}.npz")["logits"].astype(np.float32)[: len(c)]
        meta.append({"group": s.name[:4], "X": X, "h": overlap_share(c, s.human, HOP_S / 2), "d": overlap_share(c, s.dog, HOP_S / 2)})
    groups = sorted({m["group"] for m in meta})
    h = np.concatenate([m["h"] for m in meta])
    d = np.concatenate([m["d"] for m in meta])
    pos, cost = h >= 0.5, (h == 0) & (d >= 0.5)

    def fit(X, y, extra):
        Xs, ys = [X], [y]
        for Xa, ya in extra:
            Xs.append(Xa)
            ys.append(ya)
        X, y = np.vstack(Xs), np.concatenate(ys)
        use = (y >= 0.5) | (y == 0)
        sc = StandardScaler().fit(X[use])
        return sc, LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[use]), y[use] >= 0.5)

    def prob(model, X):
        sc, clf = model
        return uniform_filter1d(clf.predict_proba(sc.transform(X))[:, 1], 5, mode="nearest")

    variants = {"elte_only": [], "dogs_negative": [(Xd, yd)], "dogs_and_overlap": [(Xd, yd), (Xm, ym)]}
    results = {}
    for name, extra in variants.items():
        oof = np.zeros(len(h))
        for g in groups:
            model = fit(np.vstack([m["X"] for m in meta if m["group"] != g]), np.concatenate([m["h"] for m in meta if m["group"] != g]), extra)
            i = 0
            for m in meta:
                if m["group"] == g:
                    oof[i : i + len(m["X"])] = prob(model, m["X"])
                i += len(m["X"])
        th = {k: float(np.quantile(oof[cost], 1 - q)) for k, q in COSTS.items()}
        final = fit(np.vstack([m["X"] for m in meta]), h, extra)
        r = {f"elte_recall_{k}": round(float((oof[pos] >= t).mean()), 3) for k, t in th.items()}
        for k, t in th.items():
            hit = tot = 0
            for X, dg in zip(te_dog[0], te_dog[2]):
                m = mask_from(flagged_segments(prob(final, X), CENTRES, t, SCENE_S), len(dg))
                hit += (m & dg).sum()
                tot += dg.sum()
            r[f"unseen_dogs_muted_{k}"] = round(float(hit / tot), 3)
            sp_hit = sp_tot = ov_hit = ov_tot = do_hit = do_tot = 0
            for X, sp, dg in zip(*te_mix):
                m = mask_from(flagged_segments(prob(final, X), CENTRES, t, SCENE_S), len(sp))
                ov, do = sp & dg, dg & ~sp
                sp_hit += (m & sp).sum(); sp_tot += sp.sum()
                ov_hit += (m & ov).sum(); ov_tot += ov.sum()
                do_hit += (m & do).sum(); do_tot += do.sum()
            r[f"mixed_speech_muted_{k}"] = round(float(sp_hit / sp_tot), 3)
            r[f"mixed_speech_under_bark_muted_{k}"] = round(float(ov_hit / ov_tot), 3)
            r[f"mixed_dog_only_muted_{k}"] = round(float(do_hit / do_tot), 3)
        results[name] = r
        print(name, json.dumps(r), flush=True)
        if name == "dogs_and_overlap" and args.save_final:
            sc, clf = final
            np.savez(Path(__file__).resolve().parent.parent / "models" / "human_detector_eat40_aug2.npz",
                     mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32), coef=clf.coef_[0].astype(np.float32),
                     intercept=np.float32(clf.intercept_[0]), levels=np.array(list(LEVELS)),
                     thresholds=np.array([th[k] for k in LEVELS], np.float32), smooth=np.int32(5), hop_s=np.float32(HOP_S),
                     efficientat_model=np.array("mn40_as"),
                     trained_on=np.array("ELTE BARKS (17 participants) + synthetic scenes: 20 ESC-50 dogs, 20 LibriSpeech speakers"))
            np.savez_compressed(out / "oof_dogs_and_overlap.npz", prob=oof, human=h, dog=d)
            (out / "thresholds_dogs_and_overlap.json").write_text(json.dumps(th, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
