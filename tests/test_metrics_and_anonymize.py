"""Tests for vad_anonymization.metrics and vad_anonymization.anonymize (no models needed)."""

from __future__ import annotations

import math

import pytest
import torch

from vad_anonymization.anonymize import gated_subtract, subtract, vad_mute
from vad_anonymization.detection import SpeechSegment
from vad_anonymization.metrics import dog_retention_db, si_sdr_db, speech_leakage_db

SR = 16_000
torch.manual_seed(0)
SPEECH = torch.randn(SR * 2) * 0.1
DOG = torch.randn(SR * 2) * 0.1


def test_leakage_is_zero_db_when_speech_untouched():
    assert speech_leakage_db(SPEECH + DOG, SPEECH) == pytest.approx(0.0, abs=0.2)


def test_leakage_reflects_scaled_speech():
    # speech at 10% amplitude -> -20 dB
    assert speech_leakage_db(0.1 * SPEECH + DOG, SPEECH) == pytest.approx(-20.0, abs=0.5)


def test_dog_retention_and_si_sdr_perfect_for_clean_dog():
    assert dog_retention_db(DOG, DOG) == pytest.approx(0.0)
    assert si_sdr_db(DOG, DOG) > 90


def test_si_sdr_of_mixture_is_about_minus_snr():
    # equal power speech and dog -> about 0 dB SI-SDR w.r.t. the dog
    assert si_sdr_db(SPEECH + DOG, DOG) == pytest.approx(0.0, abs=0.5)


def test_oracle_subtraction_recovers_the_dog_exactly():
    y = subtract(SPEECH + DOG, SPEECH)
    assert torch.allclose(y, DOG, atol=1e-6)
    assert speech_leakage_db(y, SPEECH) < -40


def test_mute_zeroes_only_the_detected_window():
    x = SPEECH + DOG
    y = vad_mute(x, [SpeechSegment(0.5, 1.0)])
    assert torch.all(y[int(0.5 * SR) : int(1.0 * SR)] == 0)
    assert torch.equal(y[: int(0.5 * SR)], x[: int(0.5 * SR)])
    assert torch.equal(y[int(1.0 * SR) + 1 :], x[int(1.0 * SR) + 1 :])


def test_gated_subtraction_leaves_audio_outside_segments_untouched():
    x = SPEECH + DOG
    y = gated_subtract(x, SPEECH, [SpeechSegment(1.0, 1.5)])
    assert torch.allclose(y[: int(0.9 * SR)], x[: int(0.9 * SR)])
    mid = slice(int(1.1 * SR), int(1.4 * SR))
    assert torch.allclose(y[mid], DOG[mid], atol=1e-6)


from vad_anonymization.metrics import sdr_db


def test_plain_sdr_scores_silence_as_zero_and_penalises_turning_the_dog_down():
    assert sdr_db(torch.zeros_like(DOG), DOG) == pytest.approx(0.0)
    assert sdr_db(DOG, DOG) == 100.0
    assert sdr_db(0.5 * DOG, DOG) == pytest.approx(6.02, abs=0.05)


def test_hybrid_remove_mutes_human_only_and_subtracts_in_overlap():
    from vad_anonymization.anonymize import hybrid_remove

    sr = 16_000
    x = torch.ones(4 * sr)
    s_hat = torch.full((4 * sr,), 0.25)
    human = [SpeechSegment(1.0, 3.0)]
    dog = [SpeechSegment(2.0, 4.0)]
    y = hybrid_remove(x, s_hat, human, dog, fade_s=0.0)
    assert torch.allclose(y[: sr], x[: sr])                      # no human: untouched
    assert y[int(1.1 * sr) : int(1.9 * sr)].abs().max() == 0     # human only: muted
    assert torch.allclose(y[int(2.1 * sr) : int(2.9 * sr)], torch.tensor(0.75))  # overlap: subtracted
    assert torch.allclose(y[int(3.1 * sr) :], x[int(3.1 * sr) :])  # dog only: untouched
