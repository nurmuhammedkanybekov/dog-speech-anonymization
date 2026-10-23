"""Step 25a: Whisper's own encoder as a speech-aware feature extractor.

The privacy attacker is a speech recognizer (Whisper). The detector so far
looked at the audio through AudioSet models (VGGish, EfficientAT), which were
trained to name 527 kinds of sound, not to understand speech. Whisper's
encoder was trained on 680,000 hours of speech and turns every 20 ms of audio
into a 768-number description of what it hears. Speech, even quiet or under a
dog, should stand out in that description much more clearly than in AudioSet
class scores, and at a 12x finer time step (20 ms instead of 0.24 s).

The encoder is the int8 sherpa-onnx export of Whisper small that the privacy
check already uses (``download_models --asr``). Its output is the first
decoder layer's cross-attention keys: a fixed, full-rank linear map of the
encoder's final hidden states, so for a linear classifier it carries the same
information.

Audio is cut into 30 s windows (Whisper's fixed input length) every 20 s, and
every 20 ms frame is taken from the window in which it lies most centrally,
so each frame sees at least 5 s of context on both sides (except at the file
edges).

Writes ``outputs/whisper_enc/<session>.npz``: ``feats`` (frames x 768, float16)
and ``times`` (frame centres, seconds).

Usage (from inside speech_anonymization/):
    python -m scripts.extract_whisper_encoder            # all 30 sessions, ~10 min on 2 cores
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from vad_anonymization.audio_io import load_audio
from vad_anonymization.elte_barks import load_sessions

SR = 16_000
CHUNK = 30 * SR          # Whisper's input length
HOP = 20 * SR            # windows overlap by 10 s
FRAME_S = 0.02           # one encoder frame
N_FRAMES = 1500
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sherpa-onnx-whisper-small"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "whisper_enc"


def log_mel(x: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper's log-Mel spectrogram (as ``whisper.log_mel_spectrogram``) of exactly 30 s: (80, 3000)."""
    window = np.hanning(401)[:-1].astype(np.float32)               # periodic Hann, as torch.hann_window(400)
    pad = np.pad(x, (200, 200), mode="reflect")                    # torch.stft(center=True)
    idx = np.arange(400)[None, :] + 160 * np.arange(len(x) // 160 + 1)[:, None]
    spec = np.abs(np.fft.rfft(pad[idx] * window, axis=1)) ** 2    # (frames + 1, 201)
    m = mel_filters @ spec[:-1].T                                  # drop the last frame, as Whisper does
    log = np.log10(np.maximum(m, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)


class WhisperEncoder:
    """Whisper small encoder (sherpa-onnx int8): 30 s of 16 kHz audio -> 1500 x 768 frames."""

    def __init__(self, model_dir: Path = MODEL_DIR, threads: int = 2):
        import librosa
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_dir / "small-encoder.int8.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.mel = librosa.filters.mel(sr=SR, n_fft=400, n_mels=80).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.pad(x, (0, max(0, CHUNK - len(x))))[:CHUNK]
        k, _ = self.sess.run(None, {"mel": log_mel(x, self.mel)[None]})
        return k[0, 0]                                              # layer 0 cross-attention keys: (1500, 768)


def encode_file(x: np.ndarray, enc: WhisperEncoder) -> tuple[np.ndarray, np.ndarray]:
    """Frames for a whole recording; each frame from the window where it is most central."""
    n = int(np.ceil(len(x) / SR / FRAME_S))
    feats = np.zeros((n, 768), np.float16)
    best = np.full(n, -1.0)                                         # distance to the window edge, larger = better
    starts = list(range(0, max(1, len(x) - CHUNK + HOP), HOP))
    for a in starts:
        f = enc(x[a : a + CHUNK])
        i0 = int(round(a / SR / FRAME_S))
        j = np.arange(N_FRAMES)
        keep = (i0 + j) < n
        margin = np.minimum(j, N_FRAMES - 1 - j).astype(float)
        if a == 0:
            margin[: N_FRAMES // 2] = N_FRAMES                      # file start: nothing better exists
        if a == starts[-1]:
            margin[N_FRAMES // 2 :] = N_FRAMES                      # file end
        upd = keep & (margin > best[np.minimum(i0 + j, n - 1)])
        feats[i0 + j[upd]] = f[upd].astype(np.float16)
        best[i0 + j[upd]] = margin[upd]
    return feats, (np.arange(n) + 0.5) * FRAME_S


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    enc = WhisperEncoder(threads=args.threads)
    t0 = time.time()
    for s in load_sessions(PROJECT_ROOT / "data" / "elte_barks"):
        f = OUT / f"{s.name}.npz"
        if f.exists():
            continue
        x = load_audio(s.audio_path, SR).numpy().astype(np.float32)
        feats, times = encode_file(x, enc)
        np.savez(f, feats=feats, times=times)
        print(f"{s.name}: {len(times)} frames ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
