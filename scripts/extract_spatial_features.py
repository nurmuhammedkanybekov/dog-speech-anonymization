"""Step 15a: spatial cues from the three synchronized microphones.

The raw recordings (Step 13, ``download_raw_audio.py``) have up to three
microphones per participant on one Zoom H4 recorder: M1 clip-on, M2 shotgun,
M3 cardioid. The person and the dog are usually at different places in the
room, so *where* a sound comes from is a cue that does not depend on what it
sounds like. This script computes, on the same 0.24 s grid as the VGGish
detector (Step 9), for every analysis window of 0.96 s:

* inter-microphone level differences (ILD) in 4 frequency bands, for every
  microphone pair, and each microphone's level relative to its session median;
* GCC-PHAT time differences of arrival (TDoA) for every pair: the lag of the
  peak (ms) and its height (how clearly one direction dominates).

Each play-session file is first located inside the raw recording of its own
microphone (S = M2 shotgun, G = M3 cardioid) by cross-correlation, so the
raw microphones line up sample-exactly with the annotations.

Needs only numpy and ffmpeg (no torch), so it runs on a plain laptop.

Usage (from inside speech_anonymization/):
    python -m scripts.extract_spatial_features            # -> outputs/spatial_features/<session>.npz
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

SR = 16_000
HOP_S, WIN_S, FIRST_CENTRE_S = 0.24, 0.96, 0.48
BANDS_HZ = [(100, 500), (500, 1500), (1500, 4000), (4000, 8000)]
MAX_LAG_MS = 20.0
MICS = ["M1", "M2", "M3"]
PAIRS = list(itertools.combinations(MICS, 2))
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def read_audio(path: Path) -> np.ndarray:
    """Any audio file -> 16 kHz mono float32, via ffmpeg (no Python audio libraries needed)."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def feature_names() -> list[str]:
    names = [f"level_{m}" for m in MICS]
    names += [f"ild_{a}{b}_{lo}-{hi}" for a, b in PAIRS for lo, hi in BANDS_HZ]
    names += [f"tdoa_ms_{a}{b}" for a, b in PAIRS] + [f"gcc_peak_{a}{b}" for a, b in PAIRS]
    names += [f"has_{m}" for m in MICS]
    return names


def envelope(x: np.ndarray, hop: int = 160) -> np.ndarray:
    n = len(x) // hop
    return 10 * np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)


def xcorr_lag(big: np.ndarray, small: np.ndarray) -> tuple[int, float]:
    """Lag (in samples of ``big``) where ``small`` fits best, and the normalized correlation there."""
    n = len(big) + len(small)
    nfft = 1 << (n - 1).bit_length()
    s0 = small - small.mean()
    c = np.fft.irfft(np.fft.rfft(big, nfft) * np.conj(np.fft.rfft(s0, nfft)), nfft)[: len(big) - len(small) + 1]
    # local energy of big for normalization
    cs = np.concatenate([[0.0], np.cumsum(big.astype(np.float64) ** 2)])
    cm = np.concatenate([[0.0], np.cumsum(big.astype(np.float64))])
    m = len(small)
    var = (cs[m:] - cs[:-m]) - (cm[m:] - cm[:-m]) ** 2 / m
    r = c / (np.sqrt(np.maximum(var, 1e-20)) * np.linalg.norm(s0) + 1e-20)
    k = int(np.argmax(r))
    return k, float(r[k])


def align(session: np.ndarray, ref: np.ndarray) -> tuple[int, float]:
    """Sample offset of the session inside the raw reference microphone recording."""
    es, er = np.clip(envelope(session), -100, None), np.clip(envelope(ref), -100, None)
    coarse, _ = xcorr_lag(er, es)                     # in 10 ms frames
    a = max(0, coarse * 160 - SR)                     # refine on samples, +-1 s around the coarse lag
    seg = session[: min(len(session), 30 * SR)]
    window = ref[a : a + len(seg) + 2 * SR]
    fine, r = xcorr_lag(window, seg)
    return a + fine, r


def frame_features(windows: dict[str, np.ndarray | None], level_ref: dict[str, float]) -> np.ndarray:
    nwin = int(WIN_S * SR)
    freqs = np.fft.rfftfreq(nwin, 1 / SR)
    hann = np.hanning(nwin).astype(np.float32)
    spec, band_db, level = {}, {}, {}
    for m, x in windows.items():
        if x is None or len(x) < nwin:
            continue
        X = np.fft.rfft(x[:nwin] * hann)
        spec[m] = X
        p = np.abs(X) ** 2
        band_db[m] = [10 * np.log10(p[(freqs >= lo) & (freqs < hi)].sum() + 1e-12) for lo, hi in BANDS_HZ]
        level[m] = 10 * np.log10(float((x ** 2).mean()) + 1e-12) - level_ref[m]
    f = [level.get(m, np.nan) for m in MICS]
    for a, b in PAIRS:
        f += [band_db[a][k] - band_db[b][k] if a in band_db and b in band_db else np.nan for k in range(len(BANDS_HZ))]
    lags, peaks = [], []
    max_lag = int(MAX_LAG_MS / 1000 * SR)
    for a, b in PAIRS:
        if a in spec and b in spec:
            cross = spec[a] * np.conj(spec[b])
            band = (freqs >= 100) & (freqs <= 7000)
            cross = np.where(band, cross / (np.abs(cross) + 1e-12), 0)
            g = np.fft.irfft(cross, nwin)
            g = np.concatenate([g[-max_lag:], g[: max_lag + 1]])
            # the recorder's inputs leak into each other electrically (a sharp peak at exactly lag 0 in
            # every pair, even with the wireless clip-on, whose own delay is several ms): ignore +-1 sample
            g[max_lag - 1 : max_lag + 2] = -np.inf
            k = int(np.argmax(g))
            lags.append((k - max_lag) / SR * 1000)
            peaks.append(float(g[k]))
        else:
            lags.append(np.nan)
            peaks.append(np.nan)
    f += lags + peaks + [float(m in spec) for m in MICS]
    return np.array(f, dtype=np.float32)


def write_flac(path: Path, x: np.ndarray) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-sample_fmt", "s16", str(path)], input=np.clip(x, -1, 1).astype(np.float32).tobytes(), check=True)


def find_raw(raw_root: Path, participant: str, mic: str) -> Path | None:
    for ext in ("wav", "WAV"):
        p = raw_root / participant / f"{participant}-{mic}.{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "spatial_features")
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--force", action="store_true", help="recompute sessions that already have features")
    p.add_argument("--export-audio", type=Path, default=None,
                   help="also write each microphone's part of the session, aligned, as <dir>/<session>_<mic>.flac")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = args.data_root / "raw_audio"
    sessions = sorted(p.stem for p in (args.data_root / "play_sessions").glob("*.wav"))
    if args.sessions:
        sessions = [s for s in sessions if s in args.sessions]
    t0 = time.time()
    info = {}
    for name in sessions:
        out = args.output_dir / f"{name}.npz"
        if out.exists() and not args.force:
            continue
        participant, kind = name[:4], name.split("-")[2]          # P-02-S-P_1 -> P-02, S
        ref_mic = "M2" if kind == "S" else "M3"
        raw = {m: (read_audio(f) if (f := find_raw(raw_root, participant, m)) else None) for m in MICS}
        session = read_audio(args.data_root / "play_sessions" / f"{name}.wav")
        # the session was cut from its own microphone; if that one is missing (or fits badly), use the best other one
        order = [ref_mic] + [m for m in MICS if m != ref_mic]
        offset, corr = None, -1.0
        for m in order:
            if raw[m] is None:
                continue
            o, r = align(session, raw[m])
            if r > corr:
                offset, corr, used = o, r, m
            if corr > 0.9:
                break
        total_s = len(session) / SR
        n_frames = int((total_s - WIN_S) // HOP_S) + 1
        centres = FIRST_CENTRE_S + HOP_S * np.arange(max(0, n_frames))
        # each microphone's level reference: median 0.96 s level over the session span
        level_ref = {}
        for m, x in raw.items():
            if x is None:
                continue
            span = x[offset : offset + len(session)]
            e = envelope(span, hop=int(WIN_S * SR))
            level_ref[m] = float(np.median(e)) if len(e) else 0.0
        if args.export_audio:
            args.export_audio.mkdir(parents=True, exist_ok=True)
            for m, x in raw.items():
                if x is not None:
                    seg = x[offset : offset + len(session)]
                    write_flac(args.export_audio / f"{name}_{m}.flac", np.pad(seg, (0, len(session) - len(seg))))
        feats = []
        half = int(WIN_S * SR) // 2
        for c in centres:
            a = offset + int(c * SR) - half
            windows = {m: (x[a : a + 2 * half] if x is not None and a >= 0 else None) for m, x in raw.items()}
            feats.append(frame_features(windows, level_ref))
        np.savez_compressed(out, X=np.stack(feats) if feats else np.zeros((0, len(feature_names())), np.float32),
                            centres=centres, names=np.array(feature_names()), offset_s=offset / SR, align_corr=corr)
        info[name] = {"offset_s": round(offset / SR, 4), "align_corr": round(corr, 4), "frames": len(centres),
                      "mics": [m for m in MICS if raw[m] is not None], "aligned_on": used}
        print(f"{name}: offset {offset / SR:8.3f} s on {used}, corr {corr:.3f}, mics {info[name]['mics']}, {len(centres)} frames "
              f"({time.time() - t0:.0f}s)", flush=True)
    prev = args.output_dir / "alignment.json"
    if prev.exists():
        info = {**json.loads(prev.read_text()), **info}
    prev.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
