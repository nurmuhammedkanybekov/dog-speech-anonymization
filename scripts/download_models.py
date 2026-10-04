"""Download the pretrained speech-enhancement models into ``models/``.

All weights come straight from the authors' GitHub repositories:

* DTLN           - github.com/breizhn/DTLN            (MIT)
* GTCRN          - github.com/Xiaobin-Rong/gtcrn      (MIT)
* DeepFilterNet3 - github.com/Rikorose/DeepFilterNet  (MIT / Apache-2.0)

With ``--asr`` it also fetches the Whisper "small" speech recognizers
(OpenAI, MIT; int8 ONNX exports from github.com/k2-fsa/sherpa-onnx, ~0.5 GB
each) used for the independent privacy check. Needs ``pip install sherpa-onnx``.

With ``--detector`` it fetches the VGGish weights (Google, Apache-2.0;
PyTorch port from github.com/harritaylor/torchvggish releases, ~290 MB)
used by the trained human-sound detector (``train_human_detector``).

DeepFilterNet3 also needs the ``deepfilternet`` pip package
(``pip install -r requirements-separation.txt``).

Usage (from inside speech_anonymization/):
    python -m scripts.download_models
    python -m scripts.download_models --asr --detector
"""

from __future__ import annotations

import argparse
import io
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FILES = {
    "dtln_model_1.onnx": "https://raw.githubusercontent.com/breizhn/DTLN/master/pretrained_model/model_1.onnx",
    "dtln_model_2.onnx": "https://raw.githubusercontent.com/breizhn/DTLN/master/pretrained_model/model_2.onnx",
    "gtcrn_simple.onnx": "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/stream/onnx_models/gtcrn_simple.onnx",
}
SHERPA_ASR = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"
VGGISH = {
    "vggish/vggish-10086976.pth": "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth",
}
WHISPER = {"sherpa-onnx-whisper-small.en": "small.en", "sherpa-onnx-whisper-small": "small"}
DFN3_ZIP = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3.zip"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def fetch_whisper(folder: str, prefix: str) -> None:
    """Download + unpack one sherpa-onnx Whisper export, keep only the int8 files."""
    target = MODEL_DIR / folder
    keep = {f"{prefix}-encoder.int8.onnx", f"{prefix}-decoder.int8.onnx", f"{prefix}-tokens.txt"}
    if all((target / k).exists() for k in keep):
        print(f"  {folder}/: ok")
        return
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen(SHERPA_ASR.format(name=folder), timeout=600) as resp:
            while chunk := resp.read(1 << 20):
                tmp.write(chunk)
        tmp.seek(0)
        with tarfile.open(fileobj=tmp, mode="r:bz2") as tar:
            for member in tar.getmembers():
                if Path(member.name).name in keep:
                    member.name = f"{folder}/{Path(member.name).name}"
                    tar.extract(member, MODEL_DIR)
    print(f"  {folder}/: ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asr", action="store_true", help="also fetch the Whisper models (~1 GB)")
    p.add_argument("--detector", action="store_true", help="also fetch the VGGish weights (~290 MB)")
    args = p.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = MODEL_DIR / name
        if not target.exists():
            target.write_bytes(fetch(url))
        print(f"  {name}: {target.stat().st_size / 1e6:.1f} MB")
    if not (MODEL_DIR / "DeepFilterNet3" / "config.ini").exists():
        zipfile.ZipFile(io.BytesIO(fetch(DFN3_ZIP))).extractall(MODEL_DIR)
    print("  DeepFilterNet3/: ok")
    if args.detector:
        for name, url in VGGISH.items():
            target = MODEL_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with urllib.request.urlopen(url, timeout=600) as resp, open(target, "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
            print(f"  {name}: {target.stat().st_size / 1e6:.0f} MB")
    if args.asr:
        for folder, prefix in WHISPER.items():
            fetch_whisper(folder, prefix)
    print(f"Models in {MODEL_DIR}")


if __name__ == "__main__":
    main()
