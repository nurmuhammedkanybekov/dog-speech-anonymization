"""Pretrained speech-enhancement models, used here to *estimate the speech*.

Speech-enhancement models are trained to pull clean speech out of noisy
recordings (traffic, fans, babble... and, hopefully, dogs). For this
project the useful output is the opposite: if a model can estimate the
speech component ``s_hat`` of a mixture ``x = s + d``, then

    x - s_hat  ~=  d

is the recording with the speech removed and the dog kept. All three
models below run on CPU, need no training, and have their weights hosted
on GitHub (see ``scripts/download_models.py``):

* DTLN (Westhausen & Meyer, Interspeech 2020) - 16 kHz, two-stage LSTM, ONNX.
* GTCRN (Rong et al., ICASSP 2024) - 16 kHz, ultra-light (~48k params), ONNX.
* DeepFilterNet3 (Schroeter et al., Interspeech 2023) - 48 kHz, PyTorch.

Each wrapper takes and returns a 1-D float tensor at 16 kHz, and its output
is time-aligned with its input (checked in tests), because subtraction
only works sample-for-sample.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16_000
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class DTLNEnhancer:
    """DTLN, run block by block exactly as in the authors' ONNX example."""

    name = "dtln"
    block_len = 512
    block_shift = 128

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        self._m1 = onnxruntime.InferenceSession(str(model_dir / "dtln_model_1.onnx"), opts)
        self._m2 = onnxruntime.InferenceSession(str(model_dir / "dtln_model_2.onnx"), opts)

    def enhance(self, wave: torch.Tensor) -> torch.Tensor:
        n = wave.shape[-1]
        delay = self.block_len - self.block_shift
        audio = np.concatenate([wave.numpy().astype(np.float32), np.zeros(delay + self.block_shift, np.float32)])
        names1 = [i.name for i in self._m1.get_inputs()]
        names2 = [i.name for i in self._m2.get_inputs()]
        state1 = np.zeros([1, 2, 128, 2], np.float32)
        state2 = np.zeros([1, 2, 128, 2], np.float32)
        in_buf = np.zeros(self.block_len, np.float32)
        out_buf = np.zeros(self.block_len, np.float32)
        out = np.zeros(len(audio), np.float32)
        n_blocks = (len(audio) - (self.block_len - self.block_shift)) // self.block_shift
        for idx in range(n_blocks):
            in_buf[: -self.block_shift] = in_buf[self.block_shift :]
            in_buf[-self.block_shift :] = audio[idx * self.block_shift : (idx + 1) * self.block_shift]
            spec = np.fft.rfft(in_buf)
            mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
            mask, state1 = self._m1.run(None, {names1[0]: mag, names1[1]: state1})
            block = np.fft.irfft(mag * mask * np.exp(1j * np.angle(spec))).reshape(1, 1, -1).astype(np.float32)
            out_block, state2 = self._m2.run(None, {names2[0]: block, names2[1]: state2})
            out_buf[: -self.block_shift] = out_buf[self.block_shift :]
            out_buf[-self.block_shift :] = 0.0
            out_buf += np.squeeze(out_block)
            out[idx * self.block_shift : (idx + 1) * self.block_shift] = out_buf[: self.block_shift]
        # The streaming loop delays its output by (block_len - block_shift) samples.
        return torch.from_numpy(out[delay : delay + n].copy())


class GTCRNEnhancer:
    """GTCRN streaming ONNX model, fed one STFT frame at a time."""

    name = "gtcrn"
    n_fft = 512
    hop = 256

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        self._sess = onnxruntime.InferenceSession(str(model_dir / "gtcrn_simple.onnx"), opts)
        self._window = torch.hann_window(self.n_fft).pow(0.5)

    def enhance(self, wave: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wave, self.n_fft, self.hop, self.n_fft, self._window, return_complex=False)  # (F,T,2)
        conv_cache = np.zeros([2, 1, 16, 16, 33], np.float32)
        tra_cache = np.zeros([2, 3, 1, 1, 16], np.float32)
        inter_cache = np.zeros([2, 1, 33, 16], np.float32)
        frames = []
        for t in range(spec.shape[1]):
            frame = spec[:, t : t + 1, :].numpy()[None].astype(np.float32)  # (1,F,1,2)
            enh, conv_cache, tra_cache, inter_cache = self._sess.run(
                None, {"mix": frame, "conv_cache": conv_cache, "tra_cache": tra_cache, "inter_cache": inter_cache}
            )
            frames.append(enh[0])
        enh_spec = torch.from_numpy(np.concatenate(frames, axis=1))  # (F,T,2)
        out = torch.istft(
            torch.view_as_complex(enh_spec.contiguous()), self.n_fft, self.hop, self.n_fft, self._window, length=wave.shape[-1]
        )
        return out


def _shim_old_torchaudio() -> None:
    """DeepFilterNet 0.5.6 imports a torchaudio module removed in torchaudio 2.x.

    It only uses it for a type annotation in its file-loading helpers, which
    are not used here, so a placeholder module is enough.
    """
    if "torchaudio.backend.common" in sys.modules:
        return
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # noqa: D401 - placeholder
        pass

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


class DeepFilterNetEnhancer:
    """DeepFilterNet3 (48 kHz); input/output resampled from/to 16 kHz."""

    name = "deepfilternet3"

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        _shim_old_torchaudio()
        from df.enhance import init_df

        self._model, self._state, _ = init_df(model_base_dir=str(model_dir / "DeepFilterNet3"), log_level="ERROR")
        self._sr = self._state.sr()

    def enhance(self, wave: torch.Tensor) -> torch.Tensor:
        from df.enhance import enhance

        up = torchaudio.functional.resample(wave, SAMPLE_RATE, self._sr)[None]
        out = enhance(self._model, self._state, up)[0]
        down = torchaudio.functional.resample(out, self._sr, SAMPLE_RATE)
        return _fit_length(down, wave.shape[-1])


def _fit_length(x: torch.Tensor, n: int) -> torch.Tensor:
    return x[:n] if x.shape[-1] >= n else torch.nn.functional.pad(x, (0, n - x.shape[-1]))


ENHANCERS = {cls.name: cls for cls in (DTLNEnhancer, GTCRNEnhancer, DeepFilterNetEnhancer)}
