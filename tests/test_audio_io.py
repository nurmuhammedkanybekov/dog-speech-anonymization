"""Tests for vad_anonymization.audio_io.

These use synthetic sine waves and temp files rather than the real
project recordings, so they run offline and don't depend on any
specific dataset being present.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from vad_anonymization.audio_io import load_audio, resample, save_audio


def _sine_wave(frequency_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * frequency_hz * t)).astype("float32")


def test_resample_no_op_when_rates_match():
    waveform = torch.from_numpy(_sine_wave(440, 1.0, 16_000))
    result = resample(waveform, orig_sample_rate=16_000, target_sample_rate=16_000)
    assert torch.equal(result, waveform)


def test_resample_changes_sample_count_proportionally():
    waveform = torch.from_numpy(_sine_wave(440, 1.0, 44_100))
    result = resample(waveform, orig_sample_rate=44_100, target_sample_rate=16_000)
    expected_length = round(waveform.shape[-1] * 16_000 / 44_100)
    # torchaudio's resampler can be off by a handful of samples depending
    # on its internal filter length; a tight tolerance still catches a
    # broken rate conversion (e.g. off by a factor of 2).
    assert abs(result.shape[-1] - expected_length) <= 4


def test_load_audio_resamples_to_target_rate(tmp_path):
    native_rate = 44_100
    target_rate = 16_000
    duration_s = 0.5
    data = _sine_wave(220, duration_s, native_rate)
    wav_path = tmp_path / "tone.wav"
    sf.write(str(wav_path), data, native_rate)

    waveform = load_audio(wav_path, target_sample_rate=target_rate)

    assert waveform.dtype == torch.float32
    expected_length = round(duration_s * target_rate)
    assert abs(waveform.shape[-1] - expected_length) <= 4


def test_load_audio_collapses_stereo_to_mono(tmp_path):
    sample_rate = 16_000
    left = _sine_wave(220, 0.2, sample_rate)
    right = _sine_wave(440, 0.2, sample_rate)
    stereo = np.stack([left, right], axis=1)
    wav_path = tmp_path / "stereo.wav"
    sf.write(str(wav_path), stereo, sample_rate)

    waveform = load_audio(wav_path, target_sample_rate=sample_rate)

    assert waveform.ndim == 1
    assert waveform.shape[-1] == len(left)


def test_load_audio_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "does_not_exist.wav", target_sample_rate=16_000)


def test_save_audio_round_trip(tmp_path):
    sample_rate = 16_000
    waveform = torch.from_numpy(_sine_wave(330, 0.3, sample_rate))
    out_path = tmp_path / "nested" / "out.wav"

    save_audio(out_path, waveform, sample_rate)

    assert out_path.exists()
    data, sr = sf.read(str(out_path))
    assert sr == sample_rate
    assert len(data) == waveform.shape[-1]
