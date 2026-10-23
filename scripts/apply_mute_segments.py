"""Step 24b: write the anonymized copies of the original play-session files.

Reads the mute segments made by ``make_release_segments.py`` (one CSV per session
and level) and sets those intervals to zero in the ORIGINAL 48 kHz files,
keeping their sample rate, channels and sample format. Every sample outside
the muted intervals stays bit-identical: the audio is decoded to its own
integer or float format with ffmpeg, changed with numpy, and encoded back
with the same codec. Needs only numpy and ffmpeg (no torch), so it runs on a
laptop.

Also writes ``MANIFEST.csv``: per file, the share muted and the SHA-256 of
the original and of the anonymized file. The script then checks that every
sample outside the muted intervals matches the original exactly.

Usage (from inside speech_anonymization/):
    python -m scripts.apply_mute_segments --level standard
        # -> ../data/elte_barks/anonymized_v2/standard/<session>.wav + MANIFEST.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FORMATS = {"pcm_s16le": ("s16le", np.int16), "pcm_s24le": ("s32le", np.int32), "pcm_s32le": ("s32le", np.int32),
           "pcm_f32le": ("f32le", np.float32), "pcm_f64le": ("f64le", np.float64)}


def probe(path: Path) -> tuple[str, int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=codec_name,sample_rate,channels", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    return st["codec_name"], int(st["sample_rate"]), int(st["channels"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode(path: Path, raw_fmt: str, dtype, channels: int) -> np.ndarray:
    data = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", raw_fmt, "-"],
                          check=True, capture_output=True).stdout
    return np.frombuffer(data, dtype=dtype).reshape(-1, channels).copy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="standard", choices=["light", "standard", "strict"])
    p.add_argument("--release", default="release_v2", help="release_v2 (Step 25, default) or release_v1 (Step 24)")
    p.add_argument("--segments-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    args.segments_dir = args.segments_dir or Path(__file__).resolve().parent.parent / "outputs" / args.release
    out_dir = args.output_dir or args.data_root / f"anonymized_{args.release.split('_')[-1]}" / args.level
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest").mkdir(exist_ok=True)
    for seg_file in sorted((args.segments_dir / args.level).glob("*.csv")):
        name = seg_file.stem
        if (out_dir / "manifest" / f"{name}.json").exists():
            continue                     # done in an earlier run (the script can be stopped and restarted)
        src = args.data_root / "play_sessions" / f"{name}.wav"
        dst = out_dir / f"{name}.wav"
        codec, sr, ch = probe(src)
        if codec not in FORMATS:
            raise SystemExit(f"{name}: unsupported codec {codec}")
        raw_fmt, dtype = FORMATS[codec]
        x = decode(src, raw_fmt, dtype, ch)
        mask = np.zeros(len(x), bool)
        for row in csv.DictReader(open(seg_file)):
            a, b = int(float(row["start_s"]) * sr), int(np.ceil(float(row["end_s"]) * sr))
            mask[max(0, a) : min(len(x), b)] = True
        y = x.copy()
        y[mask] = 0
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", raw_fmt, "-ar", str(sr), "-ac", str(ch),
                        "-i", "-", "-c:a", codec, str(dst)], input=y.tobytes(), check=True)
        back = decode(dst, raw_fmt, dtype, ch)
        ok = back.shape == x.shape and np.array_equal(back[~mask], x[~mask]) and not np.any(back[mask])
        entry = {"session": name, "codec": codec, "sample_rate": sr, "channels": ch,
                 "duration_s": round(len(x) / sr, 2), "muted_share": round(float(mask.mean()), 4),
                 "kept_samples_bit_identical": bool(ok), "sha256_original": sha256(src), "sha256_anonymized": sha256(dst)}
        (out_dir / "manifest" / f"{name}.json").write_text(json.dumps(entry))
        print(f"{name}: {codec} {sr} Hz x{ch}, muted {mask.mean():.1%}, check {'ok' if ok else 'FAILED'}", flush=True)
    manifest = [json.loads(f.read_text()) for f in sorted((out_dir / "manifest").glob("*.json"))]
    with open(out_dir / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    bad = [m["session"] for m in manifest if not m["kept_samples_bit_identical"]]
    print(f"{len(manifest)} files written to {out_dir}; check failed for: {bad or 'none'}")


if __name__ == "__main__":
    main()
