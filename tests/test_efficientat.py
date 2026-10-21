"""Checks for the EfficientAT feature extractor (Step 16); skipped if the model code is not cloned."""

from pathlib import Path

import numpy as np
import pytest
import torch

EAT_DIR = Path(__file__).resolve().parent.parent / "models" / "EfficientAT"
pytestmark = pytest.mark.skipif(not (EAT_DIR / "models" / "mn").exists(), reason="EfficientAT not cloned into models/")


@pytest.fixture(scope="module")
def eat():
    from scripts.extract_efficientat import load_model

    return load_model(EAT_DIR, "mn10_as")


def test_shapes_follow_the_step9_grid(eat):
    from scripts.extract_efficientat import embed

    model, mel, labels = eat
    audio = torch.randn(16_000 * 13) * 0.01
    n = int((13 - 0.96) // 0.24) + 1
    emb, logits = embed(audio, model, mel, n)
    assert emb.shape == (n, 960) and logits.shape == (n, 527) and len(labels) == 527
    assert np.isfinite(logits).all()


def test_head_on_pooled_map_equals_the_model_output(eat):
    """Averaging all feature-map columns and applying the head = the model's own clip-level output."""
    model, mel, _ = eat
    spec = mel(torch.randn(1, 32_000 * 10) * 0.05)
    with torch.no_grad():
        logits, _ = model(spec[:, None])
        fmap = model.features(spec[:, None])[0]              # (960, F', T')
        pooled = fmap.mean((1, 2))
        mine = model.classifier[2:](pooled[None])
    assert torch.allclose(logits.reshape(-1), mine.reshape(-1), atol=1e-4)
