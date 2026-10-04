"""Ways to anonymize a recording, from "mute everything" to "subtract the speech".

* ``vad_mute`` - the original Task 1 approach: find speech with Silero VAD
  and set those samples to zero. Simple and thorough, but everything else
  in those windows - including the dog - is deleted too.
* ``subtract`` - estimate the speech with a pretrained enhancement model
  and subtract it: ``y = x - s_hat``. Keeps the dog wherever the model
  correctly leaves it out of ``s_hat``.
* ``gated_subtract`` - subtract only inside VAD-detected speech windows and
  leave the rest of the recording bit-for-bit untouched, so dog-only
  passages can never be damaged by model artifacts.
* ``hybrid_remove`` - for real recordings where dog sounds are known (or
  detected): inside human segments, mute wherever no dog is vocalizing
  (nothing to preserve there) and subtract the speech estimate only where
  the human overlaps a dog sound.
"""

from __future__ import annotations

from typing import Protocol

import torch

from vad_anonymization.detection import SpeechSegment

SAMPLE_RATE = 16_000


class Enhancer(Protocol):
    name: str

    def enhance(self, wave: torch.Tensor) -> torch.Tensor: ...


def segments_to_sample_mask(segments: list[SpeechSegment], n_samples: int, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    mask = torch.zeros(n_samples, dtype=torch.bool)
    for seg in segments:
        mask[int(seg.start_s * sample_rate) : int(seg.end_s * sample_rate)] = True
    return mask


def vad_mute(x: torch.Tensor, segments: list[SpeechSegment]) -> torch.Tensor:
    y = x.clone()
    y[segments_to_sample_mask(segments, x.shape[-1])] = 0.0
    return y


def subtract(x: torch.Tensor, speech_estimate: torch.Tensor) -> torch.Tensor:
    return x - speech_estimate


def _smooth_gate(gate: torch.Tensor, fade_s: float) -> torch.Tensor:
    fade = max(1, int(fade_s * SAMPLE_RATE))
    if fade > 1 and gate.any():
        kernel = torch.ones(1, 1, fade) / fade
        gate = torch.nn.functional.conv1d(gate[None, None], kernel, padding=fade // 2)[0, 0, : gate.shape[-1]]
    return gate


def gated_subtract(
    x: torch.Tensor, speech_estimate: torch.Tensor, segments: list[SpeechSegment], fade_s: float = 0.02
) -> torch.Tensor:
    """Subtract ``speech_estimate`` only inside ``segments`` (with short fades)."""
    gate = _smooth_gate(segments_to_sample_mask(segments, x.shape[-1]).float(), fade_s)
    return x - gate * speech_estimate


def hybrid_remove(
    x: torch.Tensor,
    speech_estimate: torch.Tensor,
    human_segments: list[SpeechSegment],
    dog_segments: list[SpeechSegment],
    fade_s: float = 0.02,
) -> torch.Tensor:
    """Mute human-only time, subtract ``speech_estimate`` where human and dog overlap."""
    human = segments_to_sample_mask(human_segments, x.shape[-1])
    dog = segments_to_sample_mask(dog_segments, x.shape[-1])
    mute = _smooth_gate((human & ~dog).float(), fade_s)
    sub = _smooth_gate((human & dog).float(), fade_s)
    return (x - sub * speech_estimate) * (1 - mute)
