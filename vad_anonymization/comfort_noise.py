"""Step 28: room-tone noise for the processed (flagged) time.

Muted time is digital silence; time where the dog extractor's output replaces the
original still carries a faint residue of the removed voice. Both get a quiet,
steady noise with the spectrum of the recording's own room tone, a few dB above
it: muted gaps then sound like the room instead of a dropout, and the residue of
the voice sits under a noise floor that a speech recognizer has to dig through,
while the much louder dog calls stay clearly above it.

The room tone is estimated without annotations: the quietest 20% of the
recording's STFT frames outside the flagged time.
"""

from __future__ import annotations

import numpy as np

N_FFT, HOP = 512, 256


def _frames(x: np.ndarray) -> np.ndarray:
    win = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    n = 1 + max(0, len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1)


def room_tone_spectrum(x: np.ndarray, flagged: np.ndarray, quiet_share: float = 0.2) -> np.ndarray:
    """Power spectrum (N_FFT // 2 + 1) of the quietest frames outside the flagged time."""
    X = np.abs(_frames(x)) ** 2
    centres = np.arange(len(X)) * HOP + N_FFT // 2
    ok = ~flagged[np.clip(centres, 0, len(flagged) - 1)]
    if ok.sum() < 10:
        ok = np.ones(len(X), bool)
    e = X[ok].sum(1)
    quiet = X[ok][e <= np.quantile(e, quiet_share)]
    return quiet.mean(0)


def room_tone(x: np.ndarray, flagged: np.ndarray, gain_db: float, seed: int = 0) -> np.ndarray:
    """Steady noise as long as ``x`` with the room tone's spectrum, ``gain_db`` above it."""
    spec = room_tone_spectrum(x, flagged)
    rng = np.random.default_rng(seed)
    n_frames = len(x) // HOP + 2
    # white Gaussian noise shaped in the STFT domain, overlap-added with a periodic Hann window
    win = np.hanning(N_FFT + 1)[:-1]
    out = np.zeros((n_frames + 1) * HOP + N_FFT)
    amp = np.sqrt(spec) * 10 ** (gain_db / 20)
    for k in range(n_frames):
        z = (rng.standard_normal(len(amp)) + 1j * rng.standard_normal(len(amp))) / np.sqrt(2)
        out[k * HOP : k * HOP + N_FFT] += np.fft.irfft(z * amp, N_FFT) * win
    y = out[N_FFT // 2 : N_FFT // 2 + len(x)]
    # calibrate: measured the same way as the room tone, the noise has its power + gain_db
    measured = (np.abs(_frames(y)) ** 2).mean(0).sum()
    return (y * np.sqrt(spec.sum() * 10 ** (gain_db / 10) / max(measured, 1e-20))).astype(np.float32)


def removed_power(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per STFT frame: power the extractor took out (x = input, y = its output), >= 0."""
    return np.maximum((np.abs(_frames(x)) ** 2).sum(1) - (np.abs(_frames(y)) ** 2).sum(1), 0.0)


def adaptive_room_tone(x: np.ndarray, y: np.ndarray, flagged: np.ndarray, rel_db: float = -6.0,
                       base_db: float = 0.0, smooth_s: float = 1.0, seed: int = 0) -> np.ndarray:
    """Room tone whose level follows how much the extractor removed (Step 28).

    Level per frame = max(room tone + ``base_db``, ``rel_db`` x the power removed, averaged over
    ``smooth_s``). Where the extractor removed a voice, the noise rises to cover what is left of it;
    where it removed almost nothing (a dog call the detector flagged without any person), the
    noise stays at the room's own level and the call stays clean. The one-second averaging keeps
    the level from following syllables, so the noise carries no trace of what was said."""
    tone = room_tone_spectrum(x, flagged).sum()                 # frame power, same scale as removed_power
    r = removed_power(x, y)
    k = max(1, int(round(smooth_s * 16_000 / HOP)))
    r = np.convolve(r, np.ones(k) / k, mode="same")
    level = np.maximum(tone * 10 ** (base_db / 10), r * 10 ** (rel_db / 10))
    gain = np.sqrt(level / tone)                                   # >= 1 for base_db >= 0
    t = np.arange(len(gain)) * HOP + N_FFT // 2
    g = np.interp(np.arange(len(x)), t, gain).astype(np.float32)
    return room_tone(x, flagged, 0.0, seed) * g
