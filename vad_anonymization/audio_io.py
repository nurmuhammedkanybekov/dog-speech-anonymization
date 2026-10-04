"""Audio loading, resampling, and saving.

Design note
-----------
Silero VAD ships a convenience loader, ``silero_vad.read_audio``, which
loads files via ``torchaudio.load``. As of torchaudio 2.9, that function
requires the optional ``torchcodec`` package to actually decode audio
files, and ``torchcodec`` does not yet have reliable prebuilt wheels for
every platform. Rather than depend on it, this module reads files with
``soundfile`` (a small, dependency-light library with no such
requirement) and does all further processing as plain ``torch`` tensors.

Resampling uses ``torchaudio.transforms.Resample``, which operates on
in-memory tensors and does not go through torchaudio's file-decoding
path - so it carries no ``torchcodec`` dependency either, while still
giving a proper windowed-sinc resampler instead of naive interpolation.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load_audio(path: str | Path, target_sample_rate: int) -> torch.Tensor:
    """Load a ``.wav`` file as a mono float32 tensor at a target rate.

    Args:
        path: Path to a ``.wav`` file.
        target_sample_rate: Desired output sample rate, in Hz. The file
            is resampled to this rate if its native rate differs.

    Returns:
        A 1-D ``float32`` tensor of audio samples.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, native_sample_rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        # Multiple channels: collapse to mono by averaging.
        data = data.mean(axis=1)

    waveform = torch.from_numpy(data)
    return resample(waveform, native_sample_rate, target_sample_rate)


def resample(waveform: torch.Tensor, orig_sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    """Resample a 1-D waveform to ``target_sample_rate``.

    Args:
        waveform: 1-D float tensor of audio samples.
        orig_sample_rate: The waveform's current sample rate, in Hz.
        target_sample_rate: The desired sample rate, in Hz.

    Returns:
        The resampled waveform. Returned unchanged (same object) if the
        rates already match.
    """
    if orig_sample_rate == target_sample_rate:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
    return resampler(waveform)


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a 1-D float32 tensor as a ``.wav`` file.

    Args:
        path: Destination path. Parent directories are created if needed.
        waveform: 1-D float tensor of audio samples.
        sample_rate: Sample rate to write into the file header, in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy(), sample_rate)
