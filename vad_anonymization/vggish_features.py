"""VGGish embeddings on a fine time grid, for a domain-adapted human-sound detector.

VGGish (Google, Apache-2.0; PyTorch port github.com/harritaylor/torchvggish)
was trained on millions of YouTube clips (AudioSet), so its 128-d embedding
already "knows" speech, laughter, whistling, dogs, ... Here it is used as a
frozen feature extractor: a small classifier on top learns what the human
sounds in the ELTE play sessions look like.

The original VGGish takes 0.96 s patches with no overlap; human events here
are often shorter than 1 s, so patches are taken every ``hop_s`` (default
0.24 s) and each patch is labeled by its centre.

The recordings are very quiet (median human event about -60 dBFS), far below
the level VGGish expects (its log-mel uses log(mel + 0.01), so quiet input is
flattened to a constant). Each session is therefore normalized to a fixed
loudness first (``target_dbfs``), with a soft limiter for loud barks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vad_anonymization.enhancers import DEFAULT_MODEL_DIR

SAMPLE_RATE = 16_000
PATCH_FRAMES = 96   # 0.96 s of 10 ms log-mel frames
FRAME_S = 0.010
WEIGHTS = "vggish/vggish-10086976.pth"


def normalize_loudness(audio: torch.Tensor, target_dbfs: float = -30.0) -> torch.Tensor:
    """Scale to ``target_dbfs`` RMS (max +60 dB) and soft-limit peaks with tanh."""
    rms = float(audio.double().pow(2).mean().sqrt())
    gain = min(1000.0, 10 ** (target_dbfs / 20) / rms) if rms > 0 else 1.0
    return torch.tanh(audio * gain)


def log_mel(audio: torch.Tensor) -> np.ndarray:
    """VGGish's exact front-end: 64 mel bands, 25 ms window, 10 ms hop -> (frames, 64)."""
    from torchvggish import mel_features, vggish_params as vp

    return mel_features.log_mel_spectrogram(
        audio.numpy().astype(np.float64), audio_sample_rate=SAMPLE_RATE, log_offset=vp.LOG_OFFSET,
        window_length_secs=vp.STFT_WINDOW_LENGTH_SECONDS, hop_length_secs=vp.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vp.NUM_MEL_BINS, lower_edge_hertz=vp.MEL_MIN_HZ, upper_edge_hertz=vp.MEL_MAX_HZ,
    )


class VGGishEmbedder:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        from torchvggish.torchvggish import _vgg

        self.model = _vgg(postprocess=False)
        self.model.load_state_dict(torch.load(Path(model_dir) / WEIGHTS, map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, hop_s: float = 0.24, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Embeddings (n, 128) and the centre time (s) of each 0.96 s patch."""
        mel = log_mel(audio).astype(np.float32)
        hop = max(1, round(hop_s / FRAME_S))
        starts = np.arange(0, max(1, mel.shape[0] - PATCH_FRAMES + 1), hop)
        out = []
        for i in range(0, len(starts), batch):
            patches = np.stack([mel[s : s + PATCH_FRAMES] for s in starts[i : i + batch]])
            out.append(self.model(torch.from_numpy(patches)[:, None]).numpy())
        centres = (starts + PATCH_FRAMES / 2) * FRAME_S
        return np.concatenate(out), centres
