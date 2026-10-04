"""Tests for vad_anonymization.mixing."""

from __future__ import annotations

import pytest
import torch

from vad_anonymization.mixing import build_overlap_mixture

SAMPLE_RATE = 16_000


def test_noise_is_placed_at_the_requested_offset():
    speech = torch.zeros(SAMPLE_RATE * 5)  # 5s of silence standing in for speech
    noise = torch.full((SAMPLE_RATE,), 0.5)  # 1s of constant-value "noise"

    mixture = build_overlap_mixture(speech, noise, SAMPLE_RATE, noise_offset_s=2.0)

    start_sample = int(2.0 * SAMPLE_RATE)
    end_sample = start_sample + SAMPLE_RATE
    assert torch.allclose(mixture.waveform[:start_sample], torch.zeros(start_sample))
    assert torch.allclose(mixture.waveform[start_sample:end_sample], noise)
    assert torch.allclose(mixture.waveform[end_sample:], torch.zeros(mixture.waveform.shape[-1] - end_sample))


def test_ground_truth_timing_matches_construction():
    speech = torch.zeros(SAMPLE_RATE * 5)
    noise = torch.full((SAMPLE_RATE * 2,), 0.3)  # 2s noise clip

    mixture = build_overlap_mixture(speech, noise, SAMPLE_RATE, noise_offset_s=1.5)

    assert mixture.noise_start_s == 1.5
    assert mixture.noise_end_s == 3.5  # 1.5s offset + 2s clip


def test_excerpt_is_padded_when_noise_runs_past_the_end():
    speech = torch.zeros(SAMPLE_RATE * 2)  # 2s
    noise = torch.full((SAMPLE_RATE * 2,), 0.4)  # 2s

    # Noise starting at 1.0s would end at 3.0s - 1s past the 2s excerpt.
    mixture = build_overlap_mixture(speech, noise, SAMPLE_RATE, noise_offset_s=1.0)

    assert mixture.excerpt_duration_s == 3.0
    assert mixture.waveform.shape[-1] == SAMPLE_RATE * 3


def test_result_never_clips():
    # Two full-scale signals summed would exceed [-1, 1] without normalization.
    speech = torch.full((SAMPLE_RATE * 2,), 0.9)
    noise = torch.full((SAMPLE_RATE,), 0.9)

    mixture = build_overlap_mixture(speech, noise, SAMPLE_RATE, noise_offset_s=0.0)

    assert mixture.waveform.abs().max().item() <= 1.0 + 1e-6


def test_no_normalization_when_it_would_not_clip():
    speech = torch.zeros(SAMPLE_RATE * 2)
    noise = torch.full((SAMPLE_RATE,), 0.2)

    mixture = build_overlap_mixture(speech, noise, SAMPLE_RATE, noise_offset_s=0.0)

    # Peak stays exactly 0.2 - i.e. normalization did not kick in and
    # needlessly rescale a mixture that was already within range.
    assert mixture.waveform.abs().max().item() == pytest.approx(0.2)


# --- SNR-controlled mixing -------------------------------------------------

import math

from vad_anonymization.mixing import (
    active_power,
    build_snr_mixture,
    noise_gain_for_snr,
    signal_power,
)


def _tone(seconds: float, freq: float = 440.0, amp: float = 0.5) -> torch.Tensor:
    t = torch.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return amp * torch.sin(2 * math.pi * freq * t)


@pytest.mark.parametrize("snr_db", [20.0, 5.0, 0.0, -10.0])
def test_snr_mixture_hits_the_target_snr(snr_db):
    speech = _tone(4.0, 300.0, 0.3)
    noise = _tone(2.0, 1000.0, 0.8)

    mix = build_snr_mixture(speech, noise, SAMPLE_RATE, snr_db=snr_db)

    # Undo any anti-clipping normalization the same way for both parts.
    speech_part = mix.waveform - mix.noise_only
    measured = 10 * math.log10(
        signal_power(speech_part) / active_power(mix.noise_only[int(mix.noise_start_s * SAMPLE_RATE):int(mix.noise_end_s * SAMPLE_RATE)], SAMPLE_RATE)
    )
    assert measured == pytest.approx(snr_db, abs=0.05)


def test_doubling_gain_lowers_snr_by_about_6_db():
    speech = _tone(2.0, 300.0)
    noise = _tone(1.0, 1000.0)
    x_10 = noise_gain_for_snr(speech, noise, SAMPLE_RATE, 10.0)
    x_4 = noise_gain_for_snr(speech, noise, SAMPLE_RATE, 10.0 - 20 * math.log10(2))
    assert x_4 == pytest.approx(2 * x_10, rel=1e-6)


def test_active_power_ignores_silence_in_sparse_noise():
    burst = _tone(0.2, 1000.0, 0.5)
    sparse = torch.cat([torch.zeros(SAMPLE_RATE * 2), burst, torch.zeros(SAMPLE_RATE * 2)])
    # Power of the burst itself, not diluted by the 4s of silence around it.
    assert active_power(sparse, SAMPLE_RATE) == pytest.approx(signal_power(burst), rel=0.05)
    assert signal_power(sparse) < active_power(sparse, SAMPLE_RATE) / 10


def test_snr_mixture_is_never_padded_and_noise_is_trimmed():
    speech = _tone(3.0, 300.0)
    noise = _tone(5.0, 1000.0)  # longer than the excerpt
    mix = build_snr_mixture(speech, noise, SAMPLE_RATE, snr_db=0.0)
    assert mix.waveform.shape[-1] == speech.shape[-1]
    assert mix.speech_duration_s == pytest.approx(3.0)
    assert mix.noise_start_s == 0.0
    assert mix.noise_end_s == pytest.approx(3.0)


def test_snr_mixture_centres_noise_by_default():
    speech = _tone(10.0, 300.0)
    noise = _tone(4.0, 1000.0)
    mix = build_snr_mixture(speech, noise, SAMPLE_RATE, snr_db=0.0)
    assert mix.noise_start_s == pytest.approx(3.0)
    assert mix.noise_end_s == pytest.approx(7.0)


def test_snr_mixture_never_clips_even_at_very_low_snr():
    speech = _tone(2.0, 300.0, 0.9)
    noise = _tone(1.0, 1000.0, 0.9)
    mix = build_snr_mixture(speech, noise, SAMPLE_RATE, snr_db=-20.0)
    assert mix.waveform.abs().max().item() <= 1.0 + 1e-6


from vad_anonymization.mixing import build_scene


def test_scene_layout_and_components_add_up():
    speech = _tone(5.0, 300.0, 0.3)
    dog = _tone(5.0, 1000.0, 0.3)
    sc = build_scene(speech, dog, SAMPLE_RATE, snr_db=0.0)
    assert sc.waveform.shape[-1] == SAMPLE_RATE * 10
    assert torch.allclose(sc.waveform, sc.speech + sc.dog, atol=1e-6)
    assert sc.speech_span_s == (0.0, 5.0)
    assert sc.dog_span_s == (2.5, 7.5)
    # dog-only part has no speech, speech-only part has no dog
    assert sc.speech[int(5.5 * SAMPLE_RATE):].abs().max() == 0
    assert sc.dog[: int(2.4 * SAMPLE_RATE)].abs().max() == 0


def test_scene_snr_is_measured_between_the_two_clips():
    speech = _tone(5.0, 300.0, 0.3)
    dog = _tone(5.0, 1000.0, 0.3)
    sc = build_scene(speech, dog, SAMPLE_RATE, snr_db=-6.0)
    s = sc.speech[: 5 * SAMPLE_RATE]
    d = sc.dog[int(2.5 * SAMPLE_RATE): int(7.5 * SAMPLE_RATE)]
    measured = 10 * math.log10(signal_power(s) / active_power(d, SAMPLE_RATE))
    assert measured == pytest.approx(-6.0, abs=0.05)
