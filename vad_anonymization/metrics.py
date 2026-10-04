"""Metrics for "remove the speech, keep the dog".

Every test mixture is built as ``x = s + d`` from a clean speech clip ``s``
and a clean dog clip ``d``, so for any processed output ``y`` both
ground truths are known and two questions can be answered with numbers:

Privacy - how much of the speech is still in ``y``?
    * ``speech_leakage_db`` - size of the speech component left in ``y``,
      relative to the original speech: 20*log10(|<y,s>| / <s,s>).
      0 dB = untouched, -20 dB = speech amplitude cut to 10%.
      Because ``s`` and ``d`` are independent recordings, the dog part of
      ``y`` contributes ~nothing to this projection.
    * ``stoi`` - Short-Time Objective Intelligibility of ``s`` inside ``y``
      (Taal et al. 2011; 0-1, higher = easier to understand). For
      anonymization, lower is better.
    * ``vad_speech_s`` - seconds of ``y`` that Silero VAD still calls
      speech (the privacy measure used by the 2026 "privacy firewall"
      paper). Some barks trigger this on their own, so it is reported next
      to the same number for the clean dog clip.

Utility - how much of the dog survives?
    * ``dog_retention_db`` - 20*log10(<y,d> / <d,d>); 0 dB = dog kept at
      full level, very negative = dog removed (e.g. muted).
    * ``dog_si_sdr_db`` - scale-invariant signal-to-distortion ratio of
      ``y`` against the clean dog (Le Roux et al. 2019); higher = ``y``
      sounds more like the clean dog recording. Leftover speech, missing
      dog and artifacts all lower it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

FLOOR_DB = -60.0


def _db(ratio: float) -> float:
    return max(FLOOR_DB, 20 * math.log10(abs(ratio))) if ratio != 0 else FLOOR_DB


def projection_gain(y: torch.Tensor, ref: torch.Tensor) -> float:
    """Least-squares coefficient a in y ~= a * ref."""
    ref = ref.double()
    energy = float((ref * ref).sum())
    return float((y.double() * ref).sum() / energy) if energy > 0 else 0.0


def speech_leakage_db(y: torch.Tensor, speech: torch.Tensor) -> float:
    return _db(projection_gain(y, speech))


def dog_retention_db(y: torch.Tensor, dog: torch.Tensor) -> float:
    return _db(projection_gain(y, dog))


def si_sdr_db(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """Scale-invariant SDR (dB) of ``estimate`` w.r.t. ``target``."""
    est = estimate.double() - estimate.double().mean()
    tgt = target.double() - target.double().mean()
    alpha = float((est * tgt).sum() / (tgt * tgt).sum())
    projected = alpha * tgt
    noise = est - projected
    num, den = float((projected * projected).sum()), float((noise * noise).sum())
    if den == 0:
        return 100.0
    if num == 0:
        return FLOOR_DB
    return max(FLOOR_DB, 10 * math.log10(num / den))


def stoi(speech: torch.Tensor, y: torch.Tensor, sample_rate: int) -> float:
    from pystoi import stoi as _stoi

    return float(_stoi(speech.numpy().astype(np.float64), y.numpy().astype(np.float64), sample_rate, extended=False))


def sdr_db(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """Plain (not scale-invariant) SDR: 10*log10(||target||^2 / ||estimate - target||^2).

    Unlike SI-SDR this penalises a target that was removed or turned down:
    replacing the dog with silence scores 0 dB, keeping it perfectly scores
    high, and leftover speech or artifacts push it below 0.
    """
    tgt = target.double()
    err = estimate.double() - tgt
    num, den = float((tgt * tgt).sum()), float((err * err).sum())
    if num == 0:
        return float("nan")
    if den == 0:
        return 100.0
    return max(FLOOR_DB, 10 * math.log10(num / den))
