"""Step 16a: a stronger frozen audio model than VGGish (EfficientAT, AudioSet mAP 0.47).

VGGish (Step 9) is a 2017 model with AudioSet mAP of about 0.31 when fine-tuned.
EfficientAT's MobileNetV3 "mn10_as" (Schmid et al., ICASSP 2023; MIT licence)
reaches 0.471 with 4.9M parameters, so it runs on a laptop CPU. It also
predicts the 527 AudioSet classes directly, including Speech, Conversation,
Child speech, Laughter, Dog and Bark.

For every microphone of every session (the session file itself + the aligned
other microphones from Step 15), on exactly the Step 9 grid (0.96 s windows,
0.24 s hop), this saves:

* ``emb``    - the 960-d pooled feature vector (the model's own "embedding");
* ``logits`` - the 527 AudioSet class logits.

Same loudness normalization as Step 9 (the recordings are around -60 dBFS).
EfficientAT expects 32 kHz audio; our files are 16 kHz, so they are upsampled
(nothing above 8 kHz, as for VGGish).

Setup (once): ``git clone https://github.com/fschmid56/EfficientAT models/EfficientAT``
(the weights download themselves from the project's GitHub release).

Usage (from inside speech_anonymization/):
    python -m scripts.extract_efficientat --worker 0 --n-workers 2      # and --worker 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

from scripts.train_multimic_detector import roles
from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions
from vad_anonymization.vggish_features import normalize_loudness

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"
SR_IN, SR_MODEL = 16_000, 32_000
HOP_FRAMES, WIN_FRAMES = 24, 96          # 10 ms mel frames: 0.24 s hop, 0.96 s window (as VGGish)
SPEECH_LIKE = ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
               "Conversation", "Narration, monologue", "Babbling", "Whispering", "Laughter", "Baby laughter",
               "Giggle", "Chuckle, chortle", "Shout", "Yell", "Children shouting", "Screaming", "Singing",
               "Humming", "Whistling", "Clapping", "Hands", "Finger snapping", "Cheering", "Applause",
               "Walk, footsteps", "Breathing", "Sigh", "Cough", "Sneeze", "Sniff"]


def load_model(efficientat_dir: Path, name: str):
    sys.path.insert(0, str(efficientat_dir))
    cwd = os.getcwd()
    os.chdir(efficientat_dir)  # helpers.utils reads its label list from a relative path
    try:
        from helpers.utils import NAME_TO_WIDTH, labels  # noqa: E402
    finally:
        os.chdir(cwd)
    from models.mn.model import get_model  # noqa: E402
    from models.preprocess import AugmentMelSTFT  # noqa: E402
    import models.mn.model as mn  # noqa: E402

    mn.model_dir = str(efficientat_dir / "resources")
    model = get_model(width_mult=NAME_TO_WIDTH(name), pretrained_name=name).eval()
    mel = AugmentMelSTFT(n_mels=128, sr=SR_MODEL, win_length=800, hopsize=320).eval()
    return model, mel, list(labels)


@torch.no_grad()
def embed(audio16: torch.Tensor, model, mel, n_frames: int, chunk: int = 1000, margin: int = 256,
          sr_in: int = SR_IN, return_columns: bool = False):
    """Local embeddings and logits on the Step 9 grid, from the model's feature map on long inputs.

    The CNN was trained on 10 s clips; on 1 s inputs its zero padding dominates and the outputs
    saturate. So the whole session runs through the convolutional part in 10 s chunks (+-2.56 s of
    context, dropped afterwards). The feature map has one column per 32 mel frames (0.32 s); for each
    0.96 s window the overlapping columns are averaged (= the model's own global pooling, restricted
    to the window) and passed through the model's classification head.
    """
    x = normalize_loudness(audio16).float()
    if sr_in != SR_MODEL:
        x = torchaudio.functional.resample(x, sr_in, SR_MODEL)
    spec = mel(x[None])[0]                                   # (128, T), 10 ms frames
    T = spec.shape[1]
    stride = 32
    cols = {}                                                # global column index -> 960-d vector
    for s in range(0, T, chunk):
        a = max(0, s - margin)
        a -= a % stride                                      # keep columns on a global 32-frame grid
        b = min(T, s + chunk + margin)
        fmap = model.features(spec[None, None, :, a:b])[0].mean(1)   # (960, t')
        for k in range(fmap.shape[1]):
            g = a // stride + k
            if s <= g * stride < s + chunk:
                cols[g] = fmap[:, k]
    G = torch.stack([cols[g] for g in sorted(cols)])         # (n_cols, 960)
    head = model.classifier[2:]                              # Linear -> Hardswish -> Dropout -> Linear
    embs = []
    for i in range(n_frames):
        lo, hi = i * HOP_FRAMES, i * HOP_FRAMES + WIN_FRAMES
        c0, c1 = lo // stride, max(lo // stride + 1, -(-hi // stride))
        embs.append(G[min(c0, len(G) - 1) : min(c1, len(G))].mean(0))
    E = torch.stack(embs)
    if return_columns:   # Step 19: the model's own 0.32 s columns, each through the head (finer timing)
        return E.numpy(), head(E).numpy(), head(G).numpy()
    return E.numpy(), head(E).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--efficientat-dir", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "EfficientAT")
    p.add_argument("--model", default="mn10_as")
    p.add_argument("--output-dir", type=Path, default=OUT / "efficientat")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--columns-only", action="store_true",
                   help="save only the per-column (0.32 s) class scores of the session microphone (Step 19)")
    p.add_argument("--audio-32k", action="store_true",
                   help="read every microphone from data/elte_barks/aligned_mics_32k (export_aligned_audio.py), "
                        "i.e. the real 32 kHz band instead of upsampled 16 kHz audio")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, mel, labels = load_model(args.efficientat_dir.resolve(), args.model)
    (args.output_dir / "labels.txt").write_text("\n".join(labels))
    t0 = time.time()
    for i, sess in enumerate(load_sessions(args.data_root)):
        if i % args.n_workers != args.worker or (args.sessions and sess.name not in args.sessions):
            continue
        n = len(np.load(OUT / "human_detector" / "features" / f"{sess.name}.npz")["centres"])
        own, _ = roles(sess.name)
        for mic in ("M1", "M2", "M3"):
            if args.columns_only and mic != own:
                continue
            out = args.output_dir / f"{sess.name}_{mic}.npz"
            if args.audio_32k:
                src = args.data_root / "aligned_mics_32k" / f"{sess.name}_{mic}.flac"
            else:
                src = sess.audio_path if mic == own else args.data_root / "aligned_mics" / f"{sess.name}_{mic}.flac"
            if out.exists() or not Path(src).exists():
                continue
            sr = SR_MODEL if args.audio_32k else SR_IN
            if args.columns_only:
                _, logits, cols = embed(load_audio(src, sr), model, mel, n, sr_in=sr, return_columns=True)
                np.savez_compressed(out, logits_cols=cols.astype(np.float16))
                print(f"{sess.name} {mic}: {len(cols)} columns ({time.time() - t0:.0f}s)", flush=True)
                continue
            emb, logits = embed(load_audio(src, sr), model, mel, n, sr_in=sr)
            np.savez_compressed(out, emb=emb.astype(np.float16), logits=logits.astype(np.float16))
            top = labels[int(np.argmax(logits.mean(0)))]
            print(f"{sess.name} {mic}: {n} frames, most common class '{top}' ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
