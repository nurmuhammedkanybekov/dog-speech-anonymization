"""Step 25: Whisper-encoder features, pooling and the 20 ms muting rule (no model files needed)."""

import numpy as np
import pytest

from scripts.extract_whisper_encoder import CHUNK, N_FRAMES, SR, encode_file, log_mel
from scripts.train_fine_detector import fine_segments
from scripts.train_whisper_detector import pool_frames


def test_log_mel_matches_whisper_reference():
    torch = pytest.importorskip("torch")
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(CHUNK)).astype(np.float32)
    filters = librosa.filters.mel(sr=SR, n_fft=400, n_mels=80).astype(np.float32)
    ours = log_mel(x, filters)
    # whisper.audio.log_mel_spectrogram, written out with torch
    st = torch.stft(torch.from_numpy(x), 400, 160, window=torch.hann_window(400), return_complex=True)
    mag = st[..., :-1].abs() ** 2
    lg = torch.clamp(torch.from_numpy(filters) @ mag, min=1e-10).log10()
    ref = ((torch.maximum(lg, lg.max() - 8.0) + 4.0) / 4.0).numpy()
    assert ours.shape == (80, 3000)
    assert np.abs(ours - ref).max() < 1e-3


class FakeEncoder:
    """Returns, for every 20 ms frame, the window start time (s) and the frame's absolute time."""

    def __init__(self):
        self.starts = []

    def __call__(self, x):
        self.starts.append(len(self.starts))
        start_s = 20.0 * (len(self.starts) - 1)
        f = np.zeros((N_FRAMES, 768), np.float32)
        f[:, 0] = start_s
        f[:, 1] = start_s + 0.02 * np.arange(N_FRAMES)
        return f


def test_encode_file_takes_each_frame_from_its_most_central_window():
    x = np.zeros(int(65 * SR), np.float32)          # 65 s -> windows at 0, 20, 40 s
    enc = FakeEncoder()
    feats, times = encode_file(x, enc)
    assert len(enc.starts) == 3
    assert len(times) == 3250
    # every frame's absolute time is right, whichever window it came from
    assert np.allclose(feats[:, 1].astype(float), times - 0.01, atol=0.04)   # float16 storage
    start = feats[:, 0].astype(float)
    assert start[int(10 / 0.02)] == 0.0            # t = 10 s: first window
    assert start[int(30 / 0.02)] == 20.0           # t = 30 s: centre of the second window
    assert start[int(60 / 0.02)] == 40.0           # t = 60 s: last window


def test_pool_frames_is_the_window_mean():
    t = (np.arange(500) + 0.5) * 0.02
    f = np.stack([t, np.ones_like(t)], 1)
    out = pool_frames(f, t, np.array([0.48, 5.0]))
    assert np.allclose(out[:, 1], 1.0)
    assert out[0, 0] == pytest.approx(0.48, abs=0.011)
    assert out[1, 0] == pytest.approx(5.0, abs=0.011)


def test_fine_segments_pads_and_merges_close_runs():
    t = (np.arange(200) + 0.5) * 0.02               # 4 s of 20 ms frames
    flags = np.zeros(200, bool)
    flags[50:60] = True                             # 1.00-1.20 s
    flags[65:70] = True                             # 1.30-1.40 s: gap 0.1 s < 0.2 s -> merged
    flags[150:151] = True                           # 3.00-3.02 s: on its own
    segs = fine_segments(t, flags, 4.0, pad_s=0.06, gap_s=0.2)
    assert len(segs) == 2
    assert segs[0][0] == pytest.approx(0.94) and segs[0][1] == pytest.approx(1.46)
    assert segs[1][0] == pytest.approx(2.94) and segs[1][1] == pytest.approx(3.08)
