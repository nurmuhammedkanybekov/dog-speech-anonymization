"""Independent privacy check: can a strong speech recognizer still read the words?

A VAD saying "no speech left" is not proof of privacy - the same kind of
detector was used to decide what to remove, so the argument is circular.
Here a separate, much stronger model (OpenAI Whisper "small", run through
sherpa-onnx) plays the attacker: it transcribes the anonymized audio, and
the transcript is compared with what it reads from the clean speech.

Models (int8 ONNX, from github.com/k2-fsa/sherpa-onnx releases) are
fetched by ``python -m scripts.download_models --asr``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from vad_anonymization.enhancers import DEFAULT_MODEL_DIR

SAMPLE_RATE = 16_000
WHISPER_MODELS = {
    "small.en": "sherpa-onnx-whisper-small.en",  # English (LibriSpeech tests)
    "small": "sherpa-onnx-whisper-small",        # multilingual (real Hungarian recordings)
}


class WhisperASR:
    def __init__(self, model: str = "small.en", language: str = "en", num_threads: int = 1,
                 model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        import sherpa_onnx

        d = Path(model_dir) / WHISPER_MODELS[model]
        self._rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(d / f"{model}-encoder.int8.onnx"),
            decoder=str(d / f"{model}-decoder.int8.onnx"),
            tokens=str(d / f"{model}-tokens.txt"),
            language=language,
            task="transcribe",
            num_threads=num_threads,
        )

    def transcribe(self, waveform: torch.Tensor | np.ndarray) -> str:
        x = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        x = np.ascontiguousarray(x, dtype=np.float32)
        stream = self._rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, x)
        self._rec.decode_stream(stream)
        return stream.result.text.strip()


def normalize_text(text: str) -> str:
    """Lower case, letters/digits/apostrophes only, single spaces.

    Whisper's sound tags such as "(dog barks)" or "[BLANK_AUDIO]" are dropped:
    they describe the audio, they are not words someone said.
    """
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]|\*[^*]*\*", " ", text)
    text = re.sub(r"[^\w' ]+", " ", text.lower().replace("-", " "))
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words / number of reference words."""
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    if not ref:
        return float("nan")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def words_recovered(reference: str, hypothesis: str) -> float:
    """Share of reference words that also appear in the hypothesis (bag of words).

    Easier to read than WER: 0.0 = not a single word of the original
    can be read back, 1.0 = every word can.
    """
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    if not ref:
        return float("nan")
    pool: dict[str, int] = {}
    for w in hyp:
        pool[w] = pool.get(w, 0) + 1
    hit = 0
    for w in ref:
        if pool.get(w, 0) > 0:
            pool[w] -= 1
            hit += 1
    return hit / len(ref)
