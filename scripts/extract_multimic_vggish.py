"""Step 15b: VGGish embeddings of every microphone, not only the session microphone.

``extract_spatial_features.py --export-audio data/elte_barks/aligned_mics`` writes
each microphone's part of every play session, aligned sample-exactly with the
annotated session file. This embeds each of them with the same frozen VGGish
and the same 0.24 s grid as the human detector (Step 9), so a detector can
look at how a sound appears on the clip-on (M1), the shotgun (M2) and the
cardioid (M3) at the same moment.

Sanity check printed per session: for the session's own microphone the
embeddings must equal the Step 9 features (same audio).

Usage (from inside speech_anonymization/):
    python -m scripts.extract_multimic_vggish --worker 0 --n-workers 2   # and --worker 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions
from vad_anonymization.vggish_features import VGGishEmbedder, normalize_loudness

SR, HOP_S = 16_000, 0.24
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--output-dir", type=Path, default=OUT / "multimic_vggish")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    embedder = None
    t0 = time.time()
    for i, sess in enumerate(load_sessions(args.data_root)):
        if i % args.n_workers != args.worker:
            continue
        ref = np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")
        for mic in ("M1", "M2", "M3"):
            src = args.data_root / "aligned_mics" / f"{sess.name}_{mic}.flac"
            out = args.output_dir / f"{sess.name}_{mic}.npz"
            if not src.exists() or out.exists():
                continue
            embedder = embedder or VGGishEmbedder()
            X, centres = embedder.embed(normalize_loudness(load_audio(src, SR)), hop_s=HOP_S)
            n = min(len(X), len(ref["X"]))
            own = ("S" in sess.name.split("-")[2] and mic == "M2") or ("G" in sess.name.split("-")[2] and mic == "M3")
            note = ""
            if own:
                a, b = X[:n].astype(np.float32), ref["X"][:n].astype(np.float32)
                cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
                note = f" (own mic: median cosine to Step 9 features {np.median(cos):.4f})"
            np.savez_compressed(out, X=X.astype(np.float16), centres=centres)
            print(f"{sess.name} {mic}: {len(X)} frames{note} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
