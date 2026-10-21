"""Step 20c: teach the detector what OTHER dogs sound like (and test on dogs it never heard).

Step 20b showed that the ELTE-trained detector mutes 21-45% of the dog time in
scenes with unfamiliar dogs (ESC-50) and no human. A smaller input does not
fix this. The fix tried here is more varied negative examples: synthetic
scenes made from half of the ESC-50 dog clips (simulated rooms, noise, no
human) are added as "not human" frames to the training data of every fold.
The other half of the dog clips is used only for testing. So the test asks
whether the detector learned about dogs in general, not about these clips.

Scored:
* ELTE, leave one participant out (must not get worse): human time found at
  6 / 10 / 15% of ELTE dog-only frames flagged;
* out of domain, with the ELTE thresholds: dog time muted in 40 scenes built
  from the held-out dog clips only, and speech time muted in the 40 speech
  scenes of Step 20a.

Usage (from inside speech_anonymization/):
    python -m scripts.train_augmented_detector
    python -m scripts.train_augmented_detector --save-final     # also write models/human_detector_eat40_aug.npz
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


def scenes(kind: str, seeds, dog_files, speech_files, transcripts, eat, cache: Path):
    if cache.exists():
        z = np.load(cache)
        return list(z["logits"]), list(z["acts"])
    counts = {"dog": (0, 5), "speech": (3, 0)}[kind]
    L, A = [], []
    for k in seeds:
        mix, sp, dg, _ = make_scene(k, speech_files, transcripts, dog_files, *counts)
        L.append(embed(torch.from_numpy(mix), eat[0], eat[1], N_FRAMES)[1])
        A.append(dg if kind == "dog" else sp)
    np.savez_compressed(cache, logits=np.stack(L), acts=np.stack(A))
    return L, A


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weight", type=float, default=1.0, help="weight of the synthetic dog frames vs ELTE frames")
    p.add_argument("--save-final", action="store_true")
    args = p.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(2)
    out = OUT / "augmented_detector"
    out.mkdir(parents=True, exist_ok=True)
    speech_dir, dog_dir = PROJECT_ROOT / "data" / "librispeech_subset", PROJECT_ROOT / "data" / "esc50_dog"
    transcripts = dict(line.strip().split(" ", 1) for line in open(speech_dir / "transcripts.txt") if line.strip())
    speech_files, dogs = sorted(speech_dir.glob("*.flac")), sorted(dog_dir.glob("*.wav"))
    train_dogs, test_dogs = dogs[0::2], dogs[1::2]
    with contextlib.redirect_stdout(io.StringIO()):
        eat = load_model(EAT_DIR, "mn40_as")[:2]
    aug_L, aug_A = scenes("dog", range(5000, 5060), train_dogs, speech_files, transcripts, eat, out / "aug_train_dogs.npz")
    tst_L, tst_A = scenes("dog", range(1000, 1040), test_dogs, speech_files, transcripts, eat, out / "test_dogs.npz")
    sp_L, sp_A = scenes("speech", range(1000, 1040), dogs, speech_files, transcripts, eat, out / "test_speech.npz")
    Xaug = np.vstack(aug_L).astype(np.float32)          # every frame of a dog-only scene is "no human"

    meta = []
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        c = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        X = np.load(OUT / "efficientat_mn40" / f"{s.name}_{roles(s.name)[0]}.npz")["logits"].astype(np.float32)[: len(c)]
        meta.append({"group": s.name[:4], "X": X, "h": overlap_share(c, s.human, HOP_S / 2), "d": overlap_share(c, s.dog, HOP_S / 2)})
    groups = sorted({m["group"] for m in meta})
    h = np.concatenate([m["h"] for m in meta])
    d = np.concatenate([m["d"] for m in meta])
    pos, cost = h >= 0.5, (h == 0) & (d >= 0.5)

    def fit(X, y, aug: bool):
        use = (y >= 0.5) | (y == 0)
        X, yb, w = X[use], (y[use] >= 0.5), np.ones(use.sum())
        if aug:
            X = np.vstack([X, Xaug])
            yb = np.concatenate([yb, np.zeros(len(Xaug), bool)])
            w = np.concatenate([w, np.full(len(Xaug), args.weight)])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X), yb, sample_weight=w)
        return sc, clf

    def prob(model, X):
        sc, clf = model
        return uniform_filter1d(clf.predict_proba(sc.transform(X))[:, 1], 5, mode="nearest")

    results = {}
    for aug in (False, True):
        name = "elte_plus_esc50_dogs" if aug else "elte_only"
        oof = np.zeros(len(h))
        for g in groups:
            model = fit(np.vstack([m["X"] for m in meta if m["group"] != g]),
                        np.concatenate([m["h"] for m in meta if m["group"] != g]), aug)
            i = 0
            for m in meta:
                if m["group"] == g:
                    oof[i : i + len(m["X"])] = prob(model, m["X"])
                i += len(m["X"])
        th = {k: float(np.quantile(oof[cost], 1 - q)) for k, q in COSTS.items()}
        final = fit(np.vstack([m["X"] for m in meta]), h, aug)
        r = {f"elte_recall_{k}": round(float((oof[pos] >= t).mean()), 3) for k, t in th.items()}
        for kind, L, A in (("unseen_dog", tst_L, tst_A), ("speech", sp_L, sp_A)):
            for k, t in th.items():
                hit = tot = 0
                for X, act in zip(L, A):
                    m = mask_from(flagged_segments(prob(final, X), CENTRES, t, SCENE_S), len(act))
                    hit += (m & act).sum()
                    tot += act.sum()
                r[f"{kind}_time_muted_{k}"] = round(float(hit / tot), 3)
        results[name] = r
        print(name, r, flush=True)
        if aug and args.save_final:
            sc, clf = final
            np.savez(Path(__file__).resolve().parent.parent / "models" / "human_detector_eat40_aug.npz",
                     mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32), coef=clf.coef_[0].astype(np.float32),
                     intercept=np.float32(clf.intercept_[0]), levels=np.array(list(LEVELS)),
                     thresholds=np.array([th[k] for k in LEVELS], np.float32), smooth=np.int32(5), hop_s=np.float32(HOP_S),
                     efficientat_model=np.array("mn40_as"),
                     trained_on=np.array("ELTE BARKS play sessions (17 participants) + synthetic scenes of 20 ESC-50 dog clips"))
            np.savez_compressed(out / "oof_augmented.npz", prob=oof, human=h, dog=d)
            (out / "thresholds_augmented.json").write_text(json.dumps(th, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
