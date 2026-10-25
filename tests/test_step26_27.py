"""Steps 26-27: call-type weights, the dog extractor's frame alignment and the --keep-dog helpers."""

import numpy as np
import pytest

from scripts.train_calltype_detector import TYPES, calltype_weights


def test_calltype_weights_equalize_types_and_keep_total():
    types = np.array([0] * 90 + [3] * 10 + [-1] * 50)          # 90 pant frames, 10 bark frames, 50 human frames
    neg = types >= 0
    w = calltype_weights(types, neg, alpha=1.0)
    assert w[types == 0].sum() == pytest.approx(w[types == 3].sum())   # each call type weighs the same in total
    assert w[neg].sum() == pytest.approx(neg.sum())                     # negatives keep their total weight
    assert np.all(w[~neg] == 1.0)                                       # positives untouched
    assert np.allclose(calltype_weights(types, neg, alpha=0.0), 1.0)    # alpha 0 = Step 25
    assert TYPES[3] == "Bark/yelp"


def test_whisper_frames_map_to_stft_frames():
    torch = pytest.importorskip("torch")
    from scripts.train_dog_extractor_w import to_stft_grid

    wf = torch.arange(100, dtype=torch.float32)[None, :, None]          # frame j holds the value j
    g = to_stft_grid(wf, 50)[0, :, 0].numpy()                            # STFT frames every 16 ms
    t = np.arange(50) * 0.016
    assert np.all(np.abs((g + 0.5) * 0.02 - t) <= 0.0101)                # nearest 20 ms frame centre
    g2 = to_stft_grid(wf, 10, offset_s=-1.0)[0, :, 0].numpy()            # a chunk starting 1 s into the file
    assert g2[0] == pytest.approx(49.5, abs=0.6)


def test_concat_indexes_across_arrays():
    from scripts.train_dog_extractor_w import _Concat

    c = _Concat([np.arange(6).reshape(3, 2), np.arange(6, 10).reshape(2, 2)])
    assert len(c) == 5
    assert np.array_equal(c[[0, 3, 4]], np.array([[0, 1], [6, 7], [8, 9]]))


def test_time_mask_and_linear_probability():
    from scripts.anonymize_recording import linear_probability, time_mask

    m = time_mask([(0.5, 1.0)], 200, 100)
    assert m.sum() == 50 and m[50] and not m[49] and not m[100]
    model = {"mean": np.zeros(2), "scale": np.ones(2), "coef": np.array([1.0, 0.0]), "intercept": 0.0, "smooth": 1}
    p = linear_probability(np.array([[0.0, 5.0], [100.0, -5.0]]), model)
    assert p[0] == pytest.approx(0.5) and p[1] == pytest.approx(1.0)
