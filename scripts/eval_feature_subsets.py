"""Step 20b: why does the detector over-flag unfamiliar dogs, and does a smaller input fix it?

Step 20a: on scenes with ESC-50 dogs and NO human, the ELTE-trained detector
mutes 21-45% of the dog time, compared with 8-18% on the ELTE dogs. With all
527 AudioSet scores as input, the logistic regression can learn cues that are
specific to the ELTE room and dogs ("not like an ELTE bark -> human").
Here the same recipe is trained on fewer, meaningful inputs:

* ``all527``    - every AudioSet class (Steps 16-18);
* ``human_dog`` - the 72 "human sounds" classes (speech ... children playing) + the 9
                  animal / dog classes (animal ... whimper (dog)), 81 inputs;
* ``human``     - only the 72 human-sound classes;
* ``zero_shot`` - no training: EfficientAT's own maximum probability over the voice
                  classes (speech ... chuckle, indices 0-21 without "speech synthesizer").

Each is scored (1) on ELTE, leave one participant out, like Step 17, and
(2) out of domain with the threshold set on ELTE (dog-only cost 6 / 10 / 15%):
the share of dog time muted on 40 dog-only scenes and the share of speech time
muted on 40 speech-only scenes (Step 20a scenes, same seeds). Frame scoring on
a 10 ms grid; no Whisper needed.

Usage (from inside speech_anonymization/):
    python -m scripts.eval_feature_subsets
"""

from __future__ import annotations

import io
import json
import contextlib
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from scripts.anonymize_recording import EAT_DIR, flagged_segments
from scripts.eval_out_of_domain import PROJECT_ROOT, SCENE_S, make_scene, mask_from
from scripts.extract_efficientat import embed, load_model
from scripts.train_human_detector import HOP_S, overlap_share
from scripts.train_multimic_detector import OUT, roles
from vad_anonymization.elte_barks import load_sessions

SUBSETS = {"all527": list(range(527)), "human_dog": list(range(81)), "human": list(range(72))}
VOICE = [i for i in range(22) if i != 7]
COSTS = {"light": 0.06, "standard": 0.10, "strict": 0.15}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(2)
    out = OUT / "feature_subsets"
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        c = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        X = np.load(OUT / "efficientat_mn40" / f"{s.name}_{roles(s.name)[0]}.npz")["logits"].astype(np.float32)[: len(c)]
        meta.append({"group": s.name[:4], "X": X, "h": overlap_share(c, s.human, HOP_S / 2), "d": overlap_share(c, s.dog, HOP_S / 2)})
    groups = sorted({m["group"] for m in meta})
    h = np.concatenate([m["h"] for m in meta])
    d = np.concatenate([m["d"] for m in meta])
    pos, neg, cost = h >= 0.5, h == 0, (h == 0) & (d >= 0.5)

    def fit(X, y):
        use = (y >= 0.5) | (y == 0)
        sc = StandardScaler().fit(X[use])
        return sc, LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[use]), y[use] >= 0.5)

    def prob(model, X, cols):
        sc, clf = model
        return uniform_filter1d(clf.predict_proba(sc.transform(X[:, cols]))[:, 1], 5, mode="nearest")

    def zero_shot(X):
        return uniform_filter1d(sigmoid(X[:, VOICE]).max(1), 5, mode="nearest")

    # (1) in domain, leave one participant out
    results, final, thresholds = {}, {}, {}
    for name in list(SUBSETS) + ["zero_shot"]:
        oof = np.zeros(len(h))
        if name == "zero_shot":
            oof = np.concatenate([zero_shot(m["X"]) for m in meta])
        else:
            cols = SUBSETS[name]
            for g in groups:
                model = fit(np.vstack([m["X"][:, cols] for m in meta if m["group"] != g]),
                            np.concatenate([m["h"] for m in meta if m["group"] != g]))
                i = 0
                for m in meta:
                    if m["group"] == g:
                        oof[i : i + len(m["X"])] = prob(model, m["X"], cols)
                    i += len(m["X"])
            final[name] = fit(np.vstack([m["X"][:, cols] for m in meta]), h)
        ev = pos | neg
        thresholds[name] = {k: float(np.quantile(oof[cost], 1 - q)) for k, q in COSTS.items()}
        results[name] = {"roc_auc": round(float(roc_auc_score(pos[ev], oof[ev])), 3),
                         **{f"elte_recall_{k}": round(float((oof[pos] >= t).mean()), 3) for k, t in thresholds[name].items()}}
        print(name, results[name], flush=True)

    # (2) out of domain: the Step 20a scenes, same seeds
    with contextlib.redirect_stdout(io.StringIO()):
        model, mel, _ = load_model(EAT_DIR, "mn40_as")
    speech_dir, dog_dir = PROJECT_ROOT / "data" / "librispeech_subset", PROJECT_ROOT / "data" / "esc50_dog"
    transcripts = dict(line.strip().split(" ", 1) for line in open(speech_dir / "transcripts.txt") if line.strip())
    speech_files, dog_files = sorted(speech_dir.glob("*.flac")), sorted(dog_dir.glob("*.wav"))
    n = int((SCENE_S - 0.96) // HOP_S) + 1
    centres = 0.48 + HOP_S * np.arange(n)
    for kind, counts in (("dog", (0, 5)), ("speech", (3, 0))):
        cache = out / f"scenes_{kind}_logits.npz"
        if cache.exists():
            z = np.load(cache)
            L, acts = list(z["logits"]), list(z["acts"])
        else:
            L, acts = [], []
            for k in range(40):
                mix, sp, dg, _ = make_scene(k, speech_files, transcripts, dog_files, *counts)
                L.append(embed(torch.from_numpy(mix), model, mel, n)[1])
                acts.append(dg if kind == "dog" else sp)
            np.savez_compressed(cache, logits=np.stack(L), acts=np.stack(acts))
        for name in results:
            for lv, t in thresholds[name].items():
                hit = tot = 0
                for X, act in zip(L, acts):
                    p = zero_shot(X) if name == "zero_shot" else prob(final[name], X, SUBSETS[name])
                    m = mask_from(flagged_segments(p, centres, t, SCENE_S), len(act))
                    hit += (m & act).sum()
                    tot += act.sum()
                results[name][f"{kind}_scenes_muted_{lv}"] = round(float(hit / tot), 3)
        print(kind, {k: {kk: v for kk, v in r.items() if kk.startswith(kind)} for k, r in results.items()}, flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
