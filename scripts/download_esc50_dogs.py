"""Download every "dog" clip from the ESC-50 dataset.

ESC-50 (Piczak, ACM Multimedia 2015; CC BY-NC 3.0) has 40 clips per
category, 5 seconds each. This script reads the dataset's own metadata
file, picks the 40 clips labelled "dog", and downloads them into
``../data/esc50_dog/`` so the SNR sweep can be reproduced from scratch
without committing audio files to the repository.

Usage (from inside speech_anonymization/):
    python -m scripts.download_esc50_dogs
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "esc50_dog"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{BASE_URL}/meta/esc50.csv", timeout=30) as resp:
        meta = resp.read().decode("utf-8")
    (OUT_DIR / "esc50_meta.csv").write_text(meta)

    dog_files = [row["filename"] for row in csv.DictReader(io.StringIO(meta)) if row["category"] == "dog"]
    print(f"{len(dog_files)} dog clips listed in ESC-50 metadata")

    for name in dog_files:
        target = OUT_DIR / name
        if target.exists() and target.stat().st_size > 0:
            continue
        with urllib.request.urlopen(f"{BASE_URL}/audio/{name}", timeout=30) as resp:
            target.write_bytes(resp.read())
        print(f"  downloaded {name}")
    print(f"Done: {len(list(OUT_DIR.glob('*.wav')))} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
