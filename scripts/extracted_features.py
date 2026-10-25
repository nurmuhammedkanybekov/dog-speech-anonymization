"""Step 28a: listen to the extractor's OUTPUT - features for a better dog router.

The Step 27 router decides "is a dog calling here?" from the original audio,
where the person's voice dominates and hides the dog (AUC 0.78 inside human
frames). The dog extractor has already removed most of the voice, so in its
output a dog call should be much easier to hear, and remaining speech should
still be recognizable as speech by the same networks.

For every session this runs the dog extractor of the OTHER participant group
(mask power 3, as deployed) over the whole recording and describes its output
with the detector's two frozen networks, on the usual 0.24 s / 0.96 s grid:
Whisper small encoder (mean over the window) and EfficientAT mn40 class scores.

Writes ``outputs/extracted_feats/<session>.npz`` (``wenc``, ``eat``).

Usage (from inside speech_anonymization/):
    python -m scripts.extracted_features         # ~20 min on 2 cores
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

import numpy as np
import torch

from scripts.eval_dog_extraction import load_extractor, run_extractor
from scripts.extract_efficientat import embed, load_model
from scripts.extract_whisper_encoder import WhisperEncoder, encode_file
from scripts.finetune_gtcrn import participant_group
from scripts.train_whisper_detector import pool_frames
from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"
EAT_DIR = Path(__file__).resolve().parent.parent / "models" / "EfficientAT"
MASK_POWER = 3.0


def main() -> None:
    torch.set_num_threads(2)
    out = OUT / "extracted_feats"
    out.mkdir(parents=True, exist_ok=True)
    enc = WhisperEncoder(threads=2)
    with contextlib.redirect_stdout(io.StringIO()):
        eat = load_model(EAT_DIR, "mn40_as")
    models: dict = {}
    t0 = time.time()
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        f = out / f"{s.name}.npz"
        if f.exists():
            continue
        centres = np.load(OUT / "human_detector" / "features" / f"{s.name}.npz")["centres"]
        x = load_audio(s.audio_path, SR).numpy().astype(np.float32)
        g = participant_group(s.name)
        if g not in models:
            models[g] = load_extractor(g, "", "w")
        y = run_extractor(models[g], "w", x, s.name, MASK_POWER).astype(np.float32)
        feats, times = encode_file(y, enc)
        wenc = pool_frames(feats, times, centres)
        _, logits = embed(torch.from_numpy(y), *eat[:2], len(centres))
        np.savez_compressed(f, wenc=wenc.astype(np.float16), eat=np.asarray(logits, np.float16))
        print(f"{s.name}: {len(centres)} frames ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
