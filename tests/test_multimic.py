"""Tests for the three-microphone helpers (Step 15)."""

import numpy as np

from scripts.compare_micfusion_pipeline import at_privacy
from scripts.extract_spatial_features import SR, WIN_S, align, feature_names, frame_features
from scripts.train_multimic_detector import roles


def test_roles():
    assert roles("P-02-S-P_1") == ("M2", "M3")
    assert roles("P-05-G-P_2") == ("M3", "M2")


def test_align_finds_offset():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(SR * 40).astype(np.float32) * np.repeat(rng.random(400), SR // 10).astype(np.float32)
    session = ref[SR * 7 + 123 : SR * 30]
    offset, corr = align(session, ref)
    assert offset == SR * 7 + 123
    assert corr > 0.99


def test_gcc_delay_sign_and_level_difference():
    rng = np.random.default_rng(1)
    n = int(WIN_S * SR)
    src = rng.standard_normal(n + 200).astype(np.float32)
    delay = 40  # samples = 2.5 ms; M1 hears the source later than M2
    windows = {"M1": src[100 - delay : 100 - delay + n] * 0.5, "M2": src[100 : 100 + n], "M3": None}
    f = dict(zip(feature_names(), frame_features(windows, {"M1": 0.0, "M2": 0.0, "M3": 0.0})))
    assert abs(f["tdoa_ms_M1M2"] - delay / SR * 1000) < 0.07
    assert abs(f["ild_M1M2_500-1500"] - 20 * np.log10(0.5)) < 0.5
    assert np.isnan(f["tdoa_ms_M1M3"]) and f["has_M3"] == 0.0


def test_gcc_ignores_zero_lag_crosstalk():
    rng = np.random.default_rng(2)
    n = int(WIN_S * SR)
    a = rng.standard_normal(n + 100).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    # identical electrical leak in both channels + the real source 30 samples later on M2
    windows = {"M1": b + a[50 : 50 + n], "M2": b + a[20 : 20 + n], "M3": None}
    f = dict(zip(feature_names(), frame_features(windows, {"M1": 0.0, "M2": 0.0, "M3": 0.0})))
    assert abs(f["tdoa_ms_M1M2"] - (-30 / SR * 1000)) < 0.07


def test_at_privacy_interpolates():
    curve = [(0.0, 0.20), (0.10, 0.08), (0.30, 0.03)]
    assert abs(at_privacy(curve, 0.08) - 0.10) < 1e-9
    assert abs(at_privacy(curve, 0.055) - 0.20) < 1e-9
    assert at_privacy(curve, 0.01) is None
