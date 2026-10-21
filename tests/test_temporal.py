"""Checks for the temporal detector (Step 17)."""

import numpy as np
import torch

from scripts.train_temporal_detector import TemporalCNN, predict, train_fold


def test_shape_and_receptive_field():
    model = TemporalCNN(10).eval()
    x = torch.zeros(1, 10, 101)
    x[0, :, 50] = 5.0
    with torch.no_grad():
        base = model(torch.zeros(1, 10, 101))
        y = model(x)
    assert y.shape == (1, 101)
    changed = (y - base).abs()[0] > 1e-7
    idx = np.flatnonzero(changed.numpy())
    # kernel 3 with dilations 1, 2, 4, 8 -> +-15 frames
    assert idx.min() == 50 - 15 and idx.max() == 50 + 15


def test_learns_a_simple_rule():
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(6):
        x = rng.standard_normal((300, 4)).astype(np.float32)
        y = (x[:, 0] > 0.5).astype(np.float32)
        seqs.append((x, y))
    model, mu, sd = train_fold(seqs, 4, seed=0, steps=150, crop=100, batch=8)
    x = rng.standard_normal((300, 4)).astype(np.float32)
    p = predict(model, mu, sd, x)
    assert p.shape == (300,)
    assert ((p > 0.5) == (x[:, 0] > 0.5)).mean() > 0.9
