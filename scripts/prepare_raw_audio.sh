#!/usr/bin/env bash
# Convert the downloaded raw multi-microphone recordings (download_raw_audio.py)
# to small 16 kHz mono FLAC files, one per microphone, for analysis.
#
# The M1 and M3 files are stereo but both channels carry the same signal, so the
# left channel is kept. Nothing is resampled in time order or trimmed, so the
# files stay sample-aligned with each other (one recorder, one clock).
#
# Usage (from inside speech_anonymization/):  bash scripts/prepare_raw_audio.sh
set -euo pipefail
SRC="${1:-../data/raw/elte_barks_raw_audio}"
DST="${2:-../data/elte_barks/raw16k}"
command -v ffmpeg >/dev/null || { echo "ffmpeg is needed (brew install ffmpeg)"; exit 1; }
find "$SRC" -type f \( -iname '*.wav' \) | sort | while read -r f; do
  rel="${f#"$SRC"/}"; out="$DST/${rel%.*}.flac"
  [ -s "$out" ] && { echo "skip $rel"; continue; }
  mkdir -p "$(dirname "$out")"
  ffmpeg -hide_banner -loglevel error -y -i "$f" -af "pan=mono|c0=c0" -ar 16000 -sample_fmt s16 "$out"
  echo "ok   $rel -> $(du -h "$out" | cut -f1)"
done
