"""Checks for the end-to-end anonymizer (Step 18)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.anonymize_recording import flagged_segments

ROOT = Path(__file__).resolve().parent.parent


def test_flagged_frames_become_padded_merged_segments():
    centres = 0.48 + 0.24 * np.arange(10)
    prob = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0], float)
    segs = flagged_segments(prob, centres, 0.5, total_s=3.0)
    # frames 1-2 -> [0.72-0.12-0.12, 0.96+0.12+0.12]; frame 6 alone
    assert np.allclose(segs[0], (0.48, 1.20)) and np.allclose(segs[1], (1.68, 2.16))
    assert flagged_segments(np.zeros(10), centres, 0.5, 3.0) == []


@pytest.mark.skipif(not (ROOT / "models" / "human_detector_eat40.npz").exists()
                    or not (ROOT / "models" / "EfficientAT" / "models").exists(), reason="trained detector not available")
def test_output_keeps_rate_channels_and_only_zeroes(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((48_000 * 12, 2)) * 0.01).astype(np.float32)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, x, 48_000, subtype="PCM_24")
    subprocess.run([sys.executable, "-m", "scripts.anonymize_recording", str(src), str(dst), "--strictness", "strict"],
                   cwd=ROOT, check=True, capture_output=True)
    y, sr = sf.read(dst, always_2d=True)
    xi, _ = sf.read(src, always_2d=True)
    assert sr == 48_000 and y.shape == xi.shape and sf.info(dst).subtype == "PCM_24"
    changed = np.any(np.abs(y - xi) > 1e-6, axis=1)
    assert np.all(y[changed] == 0)                      # nothing is altered except muted samples
    assert (dst.parent / "out.wav.segments.csv").exists()
