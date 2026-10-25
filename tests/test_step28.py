"""Step 28: the frequency-decoder extractor, the speech-leak loss and the synthetic-voice mixtures."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_w2_extractor_shape_and_range():
    from scripts.train_dog_extractor_w import DogExtractorW2

    m = DogExtractorW2().eval()
    with torch.no_grad():
        y = m(torch.randn(2, 63, 257), torch.randn(2, 63, 768))
    assert y.shape == (2, 63, 257)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_load_w_picks_the_saved_architecture(tmp_path):
    from scripts.train_dog_extractor_w import DogExtractorW, DogExtractorW2, load_w

    for cls, arch in ((DogExtractorW, None), (DogExtractorW2, "w2")):
        ck = {"model": cls().state_dict()}
        if arch:
            ck["arch"] = arch
        torch.save(ck, tmp_path / "m.pt")
        assert type(load_w(tmp_path / "m.pt")) is cls             # old files (no "arch") load as Step 27 models


def test_leak_term_rewards_removing_speech():
    from scripts.train_dog_extractor_w import loss_fn

    class Const(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, logpow, wfeat):
            return torch.full_like(logpow, self.v)

    g = torch.Generator().manual_seed(0)
    d = 0.01 * torch.randn(1, 16000, generator=g)
    s = 0.1 * torch.randn(1, 16000, generator=g)
    w = torch.zeros(1, 50, 768)
    base = [float(loss_fn(Const(v), d, s, w)[0]) for v in (0.3, 0.1)]
    leak = [float(loss_fn(Const(v), d, s, w, leak=1.0)[0]) for v in (0.3, 0.1)]
    # with the leak term, going from mask 0.3 to 0.1 (speech -10.5 dB -> -20 dB) gains ~9.5/40 more
    assert (leak[0] - leak[1]) - (base[0] - base[1]) == pytest.approx(20 * np.log10(3) / 40, abs=0.01)
    assert float(loss_fn(Const(0.5), d, torch.zeros_like(s), w, leak=1.0)[0]) == pytest.approx(
        float(loss_fn(Const(0.5), d, torch.zeros_like(s), w)[0]))   # no speech: no leak term


def test_banks_draws_synthetic_voices_at_lab_level():
    from scripts.train_dog_extractor import Banks, SR

    b = Banks.__new__(Banks)                                         # no sessions needed for the human-sound draw
    b.human = [np.full(SR, 0.2, np.float32)]
    b.human_power = [0.04]
    b.libri = []
    b.tts = [np.sin(np.arange(2 * SR) / 5).astype(np.float32)]
    b.p_libri, b.p_tts = 0.0, 1.0
    b.rirs = [np.r_[1.0, np.zeros(99)].astype(np.float32)]          # dry room
    rng = random.Random(0)
    from scripts.finetune_gtcrn import active_power

    for _ in range(5):
        h = b.human_sound(rng)
        assert len(h) > SR // 2
        assert active_power(h) == pytest.approx(0.04, rel=1e-3)       # level of a real lab voice
    b.p_tts = 0.0
    assert np.all(b.human_sound(rng) == 0.2)                          # p_tts 0: the real events (Step 27)


def test_room_tone_matches_the_quiet_frames():
    from vad_anonymization.comfort_noise import _frames, room_tone, room_tone_spectrum

    rng = np.random.default_rng(0)
    x = np.convolve(0.01 * rng.standard_normal(16000 * 20), np.ones(6) / 6, "same").astype(np.float32)
    x[16000 * 5 : 16000 * 8] += 0.3 * np.sin(np.arange(16000 * 3) / 3)          # a loud event
    flagged = np.zeros(len(x), bool)
    flagged[16000 * 5 : 16000 * 8] = True                                        # ... inside the flagged time
    spec = room_tone_spectrum(x, flagged)
    n = room_tone(x, flagged, 6.0, seed=1)
    got = (np.abs(_frames(n)) ** 2).mean(0)
    assert 10 * np.log10(got.sum() / spec.sum()) == pytest.approx(6.0, abs=0.05)
    shape = 10 * np.log10(got[5:200] / spec[5:200]) - 6.0
    assert np.median(np.abs(shape)) < 0.5                                        # same spectral shape
    assert len(n) == len(x)


def test_adaptive_room_tone_follows_the_removed_power():
    from vad_anonymization.comfort_noise import adaptive_room_tone, room_tone

    rng = np.random.default_rng(1)
    x = (0.005 * rng.standard_normal(16000 * 20)).astype(np.float32)
    voice = slice(16000 * 4, 16000 * 8)
    x[voice] += 0.2 * rng.standard_normal(16000 * 4).astype(np.float32)          # loud "voice" ...
    y = x.copy()
    y[voice] = 0.005 * rng.standard_normal(16000 * 4)                            # ... that the extractor removed
    flagged = np.zeros(len(x), bool)
    flagged[16000 * 3 : 16000 * 14] = True
    n = adaptive_room_tone(x, y, flagged, rel_db=-12.0, seed=2)
    base = room_tone(x, flagged, 0.0, seed=2)
    p = lambda a, s: float((a[s] ** 2).mean())
    removed = p(x, voice) - p(y, voice)
    assert 10 * np.log10(p(n, slice(16000 * 5, 16000 * 7)) / removed) == pytest.approx(-12.0, abs=1.0)
    quiet = slice(16000 * 11, 16000 * 13)                                        # flagged, nothing removed
    assert 10 * np.log10(p(n, quiet) / p(base, quiet)) == pytest.approx(0.0, abs=0.5)


def test_compare_leaks_recovered_words():
    from scripts.compare_leaks import HALLUCINATION, NON_LEXICAL, recovered_words

    assert recovered_words("gyere ide gyere", "gyere gyere most") == ["gyere", "gyere"]
    assert NON_LEXICAL.match("ha") and NON_LEXICAL.match("mm-hmm") and not NON_LEXICAL.match("kiss")
    assert "you" in HALLUCINATION and "kiss" not in HALLUCINATION
