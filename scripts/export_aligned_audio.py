"""Step 17b: every microphone's part of every session at a chosen sample rate (default 32 kHz).

Step 15 exported the aligned microphones at 16 kHz, the rate of the whole
pipeline. EfficientAT (Step 16) was trained on 32 kHz audio and uses the band
up to 16 kHz, where part of a bark or a hiss lives. This cuts the same
aligned excerpts straight from the 48 kHz raw recordings, using the offsets
found by ``extract_spatial_features.py`` (``outputs/spatial_features/alignment.json``).

Needs only ffmpeg/ffprobe (no Python packages), so it runs on the laptop.

Usage (from inside speech_anonymization/):
    python -m scripts.export_aligned_audio                       # -> ../data/elte_barks/aligned_mics_32k/
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "outputs"


def duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--sr", type=int, default=32_000)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    out_dir = args.output_dir or args.data_root / f"aligned_mics_{args.sr // 1000}k"
    out_dir.mkdir(parents=True, exist_ok=True)
    alignment = json.loads((OUT / "spatial_features" / "alignment.json").read_text())
    for name, info in sorted(alignment.items()):
        participant = name[:4]
        total = duration_s(args.data_root / "play_sessions" / f"{name}.wav")
        for mic in info["mics"]:
            src = next((f for ext in ("wav", "WAV") if (f := args.data_root / "raw_audio" / participant / f"{participant}-{mic}.{ext}").exists()), None)
            dst = out_dir / f"{name}_{mic}.flac"
            if src is None or dst.exists():
                continue
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{info['offset_s']:.6f}", "-t", f"{total:.6f}",
                            "-i", str(src), "-ac", "1", "-ar", str(args.sr), "-sample_fmt", "s16", str(dst)], check=True)
        print(f"{name}: {len(info['mics'])} mics, {total:.1f} s", flush=True)


if __name__ == "__main__":
    main()
