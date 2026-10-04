"""Cut 5 s "dog only" windows out of the real ELTE play sessions.

A window qualifies if at least ``--min-dog`` of it is annotated dog sound
and no human vocalization is annotated within ``--human-margin`` seconds.
Up to ``--per-session`` non-overlapping windows (most dog sound first) are
saved per session as 16 kHz mono WAV, named after the session, start time
and the dominant dog sound type. The windows keep the real room and
microphone character of the lab recordings, so mixing speech into them is
a much closer stand-in for the real task than ESC-50 clips.

Usage (from inside speech_anonymization/):
    python -m scripts.extract_elte_dog_windows
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from vad_anonymization.audio_io import load_audio, save_audio
from vad_anonymization.elte_barks import events_to_mask, load_sessions

SAMPLE_RATE = 16_000
FRAME_S = 0.01
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--window-s", type=float, default=5.0)
    p.add_argument("--min-dog", type=float, default=0.3)
    p.add_argument("--human-margin", type=float, default=1.0)
    p.add_argument("--per-session", type=int, default=3)
    args = p.parse_args()

    out_dir = args.data_root / "dog_windows"
    out_dir.mkdir(exist_ok=True)
    win = int(args.window_s / FRAME_S)
    counts: Counter = Counter()
    for sess in load_sessions(args.data_root):
        audio = load_audio(sess.audio_path, SAMPLE_RATE)
        n = int(audio.shape[-1] / SAMPLE_RATE / FRAME_S)
        dog = events_to_mask(sess.dog, n, FRAME_S)
        near_human = events_to_mask(sess.human, n, FRAME_S, pad_s=args.human_margin)
        candidates = []
        for start in range(0, n - win, win // 5):
            if near_human[start : start + win].any():
                continue
            cover = float(dog[start : start + win].float().mean())
            if cover >= args.min_dog:
                candidates.append((cover, start))
        taken: list[int] = []
        for cover, start in sorted(candidates, reverse=True):
            if len(taken) >= args.per_session:
                break
            if any(abs(start - t) < win for t in taken):
                continue
            taken.append(start)
            t0, t1 = start * FRAME_S, (start + win) * FRAME_S
            types = Counter()
            for e in sess.dog:
                overlap = min(e.end_s, t1) - max(e.start_s, t0)
                if overlap > 0:
                    types[e.label.strip().replace("/", "-")] += overlap
            dominant = types.most_common(1)[0][0]
            counts[dominant] += 1
            clip = audio[int(t0 * SAMPLE_RATE) : int(t0 * SAMPLE_RATE) + int(args.window_s * SAMPLE_RATE)]
            save_audio(out_dir / f"{sess.name}_{t0:06.1f}s_{dominant}.wav", clip, SAMPLE_RATE)
    print(f"Saved {sum(counts.values())} windows to {out_dir}")
    print("Dominant dog sound per window:", dict(counts))


if __name__ == "__main__":
    main()
