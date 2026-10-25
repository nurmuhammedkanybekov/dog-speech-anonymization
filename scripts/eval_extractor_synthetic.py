"""Step 27g: the dog extractor on held-out synthetic mixtures, where the true dog is known.

For each participant group, the extractor trained on the OTHER group processes
the first 400 mixtures of this group's dataset (``train_dog_extractor_w
--build``: real dog/background windows + real human-only events). Reported, for
mask powers 1 and 3:

* speech left: energy of the human sound that passes the mask (dB, relative to the human sound);
* dog kept: energy of the dog + background that passes (dB, relative to it);
* dog SI-SDR: how close the output's dog part is to the true dog (dB; the dog part
  of the output is the mask applied to the dog alone, so speech leakage is not counted here);
* output speech-to-dog ratio where the human sound is active: how audible the
  remaining speech is next to the dog.

Usage (from inside speech_anonymization/):
    python -m scripts.eval_extractor_synthetic
    python -m scripts.eval_extractor_synthetic --tag _tts      # Step 28 extractors
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from scripts.train_dog_extractor import MODEL_DIR, features, istft, stft
from scripts.train_dog_extractor_w import DATA, batch, load_w, to_stft_grid

N = 400


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-12)
    t = a * ref
    return 10 * torch.log10(t.pow(2).sum(-1) / ((est - t).pow(2).sum(-1) + 1e-12) + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="", help="extractor variant (file suffix), e.g. _tts (Step 28)")
    p.add_argument("--data", default="", help="mixture set suffix: '' (real lab voices) or _clean (Step 28: unseen synthetic"
                                          " and LibriSpeech voices only, no room background in the human part)")
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--groups", nargs="+", default=["odd", "even"], help="evaluated participant groups")
    p.add_argument("--powers", type=float, nargs="+", default=[1.0, 3.0], help="mask powers")
    args = p.parse_args()
    torch.set_num_threads(2)
    res = {}
    for group in args.groups:
        other = "even" if group == "odd" else "odd"
        m = load_w(MODEL_DIR / f"dog_extractor_w_{other}{args.tag}.pt")
        data = {k: np.load(DATA / (group + args.data) / f"{k}.npy", mmap_mode="r") for k in ("dog", "human", "whisper")}
        for power in args.powers:
            acc = {"sl": 0.0, "sp": 0.0, "dk": 0.0, "dp": 0.0, "spk": 0.0, "dpk": 0.0}
            sdr = []
            with torch.no_grad():
                for i in range(0, args.n, 16):
                    d, s, w = batch(data, np.arange(i, min(args.n, i + 16)))
                    D, S = stft(d), stft(s)
                    M = m(features(D + S), to_stft_grid(w, D.shape[1])) ** power
                    ps, pd = S.abs() ** 2, D.abs() ** 2
                    act = (ps.sum(-1, keepdim=True) > 1e-3 * ps.sum(-1, keepdim=True).amax(1, keepdim=True)).float()
                    acc["sl"] += float((M ** 2 * ps).sum()); acc["sp"] += float(ps.sum())
                    acc["dk"] += float((M ** 2 * pd).sum()); acc["dp"] += float(pd.sum())
                    acc["spk"] += float((M ** 2 * ps * act).sum()); acc["dpk"] += float((M ** 2 * pd * act).sum())
                    dog_out = istft(M * D, d.shape[-1])
                    sdr += si_sdr(dog_out, d).tolist()
            db = lambda a, b: round(10 * math.log10(a / b + 1e-12), 2)
            res[f"{group} (model trained on {other}), mask power {power:g}"] = {
                "speech_left_db": db(acc["sl"], acc["sp"]), "dog_kept_db": db(acc["dk"], acc["dp"]),
                "dog_si_sdr_db_median": round(float(np.median(sdr)), 2),
                "output_speech_to_dog_db_while_speaking": db(acc["spk"], acc["dpk"])}
            print(group, power, res[f"{group} (model trained on {other}), mask power {power:g}"], flush=True)
    out = DATA.parent / "dog_extraction" / f"synthetic_heldout{args.tag}{args.data}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
