"""Building synthetic speech/noise mixtures with known ground truth.

Why synthetic mixtures
----------------------
Evaluating a detector against a found recording that happens to contain
both a dog and a person talking requires guessing where each sound
actually starts and stops - so a mismatch between the detector's output
and that guess cannot be distinguished from the guess simply being
wrong. Constructing the mixture from two sources whose individual
timing is already known removes that ambiguity: the "correct answer"
for where the noise sits is fixed by construction, before the detector
ever runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OverlapMixture:
    """A speech excerpt with a noise clip overlaid at a known offset."""

    waveform: torch.Tensor
    sample_rate: int
    excerpt_duration_s: float
    noise_start_s: float
    noise_end_s: float


def build_overlap_mixture(
    speech_excerpt: torch.Tensor,
    noise_clip: torch.Tensor,
    sample_rate: int,
    noise_offset_s: float,
) -> OverlapMixture:
    """Overlay ``noise_clip`` onto ``speech_excerpt`` at a known offset.

    "Mixing" two audio signals that occupy the same span of time is
    simply adding their sample values together, sample by sample, over
    that span - which is what happens physically when two sounds occur
    at once and are picked up by the same microphone.

    Args:
        speech_excerpt: 1-D float tensor, the speech to overlay noise onto.
        noise_clip: 1-D float tensor, the noise (e.g. a dog bark) to overlay.
        sample_rate: Sample rate of both inputs, in Hz. Both must already
            share this rate (resample beforehand if they don't).
        noise_offset_s: Where the noise clip should start, in seconds
            from the beginning of the excerpt.

    Returns:
        An ``OverlapMixture`` with the combined waveform and the exact
        ground-truth timing of the noise within it. The excerpt is
        padded with silence if the noise clip would run past its end.
        The result is peak-normalized only if the sum would otherwise
        exceed the [-1, 1] range (clipping).
    """
    mixture = speech_excerpt.clone()

    noise_start_sample = int(round(noise_offset_s * sample_rate))
    noise_end_sample = noise_start_sample + noise_clip.shape[-1]

    if noise_end_sample > mixture.shape[-1]:
        pad_amount = noise_end_sample - mixture.shape[-1]
        mixture = torch.nn.functional.pad(mixture, (0, pad_amount))

    mixture[noise_start_sample:noise_end_sample] += noise_clip

    peak = mixture.abs().max().item()
    if peak > 1.0:
        mixture = mixture / peak

    return OverlapMixture(
        waveform=mixture,
        sample_rate=sample_rate,
        excerpt_duration_s=mixture.shape[-1] / sample_rate,
        noise_start_s=noise_offset_s,
        noise_end_s=noise_offset_s + noise_clip.shape[-1] / sample_rate,
    )


# ---------------------------------------------------------------------------
# SNR-controlled mixing
# ---------------------------------------------------------------------------
#
# Why SNR control
# ---------------
# ``build_overlap_mixture`` simply adds the two waveforms, so how loud the
# dog is relative to the speech is whatever it happened to be in the two
# source recordings - an uncontrolled variable. Signal-to-noise ratio
# (SNR) makes that loudness an explicit, chosen parameter:
#
#     SNR = P_speech / P_noise          (linear)
#     SNR_dB = 10 * log10(P_speech / P_noise)
#
# where P is signal power (mean of squared samples). To hit a target SNR
# the noise is multiplied by a gain ``x`` before mixing:
#
#     mixture = speech + x * noise,   x = sqrt(P_speech / (P_noise * 10^(SNR_dB/10)))
#
# Scaling amplitude by ``x`` scales power by ``x**2``, so doubling ``x``
# (e.g. 0.1 -> 0.2) lowers the SNR by about 6 dB, i.e. makes the noise
# louder relative to the speech.
#
# Measuring noise power: barking is intermittent - a 5s clip can contain a
# single 0.2s bark and silence elsewhere. Averaging power over the whole
# clip would count that silence and make the bark itself far louder than
# the nominal SNR suggests. Noise power is therefore measured over the
# clip's *active* frames only (frames within ``active_floor_db`` of the
# loudest frame), in the spirit of "active speech level" measurement.


@dataclass(frozen=True)
class SnrMixture:
    """A speech excerpt with a noise clip mixed in at a controlled SNR."""

    waveform: torch.Tensor  # speech + scaled noise
    noise_only: torch.Tensor  # the scaled noise alone, same length/normalization
    sample_rate: int
    snr_db: float
    noise_gain: float  # the ``x`` the noise was multiplied by
    speech_duration_s: float
    noise_start_s: float
    noise_end_s: float


def signal_power(waveform: torch.Tensor) -> float:
    """Mean power (mean of squared samples) of a 1-D waveform."""
    return float(waveform.double().pow(2).mean().item())


def active_power(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_s: float = 0.02,
    active_floor_db: float = 30.0,
) -> float:
    """Mean power over the waveform's active (non-silent) frames.

    A frame counts as active if its power is within ``active_floor_db``
    of the loudest frame in the clip.
    """
    frame_len = max(1, int(frame_s * sample_rate))
    n_frames = waveform.shape[-1] // frame_len
    if n_frames == 0:
        return signal_power(waveform)
    frames = waveform[: n_frames * frame_len].double().reshape(n_frames, frame_len)
    frame_power = frames.pow(2).mean(dim=1)
    peak = frame_power.max().item()
    if peak <= 0.0:
        return 0.0
    threshold = peak * 10 ** (-active_floor_db / 10)
    return float(frame_power[frame_power >= threshold].mean().item())


def noise_gain_for_snr(
    speech: torch.Tensor,
    noise: torch.Tensor,
    sample_rate: int,
    snr_db: float,
) -> float:
    """Gain ``x`` that puts ``x * noise`` at ``snr_db`` relative to ``speech``."""
    p_speech = signal_power(speech)
    p_noise = active_power(noise, sample_rate)
    if p_noise <= 0.0:
        raise ValueError("Noise clip is silent; SNR is undefined.")
    return (p_speech / (p_noise * 10 ** (snr_db / 10))) ** 0.5


def build_snr_mixture(
    speech_excerpt: torch.Tensor,
    noise_clip: torch.Tensor,
    sample_rate: int,
    snr_db: float,
    noise_offset_s: float | None = None,
) -> SnrMixture:
    """Mix ``noise_clip`` into ``speech_excerpt`` at a target SNR.

    Unlike ``build_overlap_mixture``, the excerpt is never padded: a noise
    clip longer than the space left after ``noise_offset_s`` is trimmed,
    so every second of the mixture is real speech and coverage can be
    measured against the true speech duration. With ``noise_offset_s``
    left as ``None`` the noise is centred in the excerpt.

    If the sum would clip, the whole mixture (and ``noise_only``) is scaled
    down by the same factor, which leaves the SNR unchanged.
    """
    n_speech = speech_excerpt.shape[-1]
    if noise_offset_s is None:
        start = max(0, (n_speech - noise_clip.shape[-1]) // 2)
    else:
        start = int(round(noise_offset_s * sample_rate))
    if start >= n_speech:
        raise ValueError("noise_offset_s is past the end of the speech excerpt.")
    noise = noise_clip[: n_speech - start]
    end = start + noise.shape[-1]

    gain = noise_gain_for_snr(speech_excerpt, noise, sample_rate, snr_db)
    noise_only = torch.zeros_like(speech_excerpt)
    noise_only[start:end] = gain * noise
    mixture = speech_excerpt + noise_only

    peak = mixture.abs().max().item()
    if peak > 1.0:
        mixture = mixture / peak
        noise_only = noise_only / peak

    return SnrMixture(
        waveform=mixture,
        noise_only=noise_only,
        sample_rate=sample_rate,
        snr_db=snr_db,
        noise_gain=gain,
        speech_duration_s=n_speech / sample_rate,
        noise_start_s=start / sample_rate,
        noise_end_s=end / sample_rate,
    )


@dataclass(frozen=True)
class Scene:
    """A longer recording with speech and a dog only partly overlapping.

    Layout (defaults): speech 0-5 s, dog 2.5-7.5 s, silence to 10 s, giving
    a speech-only part, an overlap, a dog-only part and silence - closer to
    a real session than a 100% overlap, and the only way to see whether a
    method damages dog sounds when nobody is talking.
    """

    waveform: torch.Tensor  # speech + dog
    speech: torch.Tensor  # speech component alone (same length)
    dog: torch.Tensor  # dog component alone, already scaled to the SNR
    sample_rate: int
    snr_db: float
    speech_span_s: tuple[float, float]
    dog_span_s: tuple[float, float]


def build_scene(
    speech_clip: torch.Tensor,
    dog_clip: torch.Tensor,
    sample_rate: int,
    snr_db: float,
    speech_start_s: float = 0.0,
    dog_start_s: float = 2.5,
    total_s: float = 10.0,
) -> Scene:
    """Place speech and dog on one timeline, dog scaled to ``snr_db``.

    SNR is computed between the speech clip and the dog clip themselves
    (speech power vs. active dog power, as in ``build_snr_mixture``), so it
    describes how loud they are relative to each other where they overlap,
    not diluted by the silent parts of the scene.
    """
    n = int(round(total_s * sample_rate))
    gain = noise_gain_for_snr(speech_clip, dog_clip, sample_rate, snr_db)
    speech = torch.zeros(n)
    dog = torch.zeros(n)
    s0, d0 = int(round(speech_start_s * sample_rate)), int(round(dog_start_s * sample_rate))
    s_clip = speech_clip[: n - s0]
    d_clip = dog_clip[: n - d0]
    speech[s0 : s0 + s_clip.shape[-1]] = s_clip
    dog[d0 : d0 + d_clip.shape[-1]] = gain * d_clip
    mixture = speech + dog
    peak = mixture.abs().max().item()
    if peak > 1.0:
        mixture, speech, dog = mixture / peak, speech / peak, dog / peak
    return Scene(
        waveform=mixture,
        speech=speech,
        dog=dog,
        sample_rate=sample_rate,
        snr_db=snr_db,
        speech_span_s=(s0 / sample_rate, (s0 + s_clip.shape[-1]) / sample_rate),
        dog_span_s=(d0 / sample_rate, (d0 + d_clip.shape[-1]) / sample_rate),
    )
