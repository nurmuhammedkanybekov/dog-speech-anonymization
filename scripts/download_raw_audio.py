"""Download ONLY the raw multi-microphone audio from the ELTE BARKS dataset.

"02 Raw recordings.zip" on Zenodo is 43 GB, but almost all of it is video.
The 48 audio files inside (per participant: M1 clip-on wireless, M2 Sennheiser
ME66 shotgun, M3 Sennheiser ME64 cardioid, recorded together on one Zoom H4)
are only ~2 GB compressed. Zenodo supports HTTP range requests, so this script
reads the zip's table of contents from the end of the file and then downloads
and unpacks just the audio entries - no 43 GB download, no extra packages
(Python standard library only).

Each file is checked against the CRC-32 stored in the zip. Files that are
already complete are skipped, so the script can be stopped and restarted.

Usage (from inside speech_anonymization/):
    python -m scripts.download_raw_audio --list                 # show what would be downloaded
    python -m scripts.download_raw_audio                        # all 17 participants (~5.2 GB on disk)
    python -m scripts.download_raw_audio --participants P-02    # just one
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import time
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

URL = "https://zenodo.org/api/records/18972388/files/02%20Raw%20recordings.zip/content"
DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "data" / "elte_barks" / "raw_audio"
AUDIO_RE = re.compile(r"^02 Raw recordings/(P-\d\d)/Audios/([^/]+\.wav)$", re.IGNORECASE)
CHUNK = 8 << 20

RangeReader = Callable[[int, int], bytes]  # (start, end inclusive) -> bytes


@dataclass
class Entry:
    name: str
    participant: str
    filename: str
    method: int
    crc: int
    csize: int
    usize: int
    header_offset: int


def http_range_reader(url: str, retries: int = 5) -> tuple[RangeReader, int]:
    """A function reading byte ranges of ``url``, and the total size."""

    def read(a: int, b: int) -> bytes:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "dog-speech-anonymization"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    if resp.status != 206:
                        raise RuntimeError(f"server ignored the range request (HTTP {resp.status})")
                    data = resp.read()
                if len(data) != b - a + 1:
                    raise IOError(f"short read: {len(data)} of {b - a + 1} bytes")
                return data
            except Exception as exc:  # network hiccup: wait and retry
                if attempt == retries - 1:
                    raise
                print(f"    retry {attempt + 1} after error: {exc}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")

    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "dog-speech-anonymization"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers["Content-Range"].split("/")[-1])
    return read, total


def list_entries(read: RangeReader, total: int) -> list[Entry]:
    """Parse the (ZIP64) central directory at the end of the archive."""
    tail_len = min(total, 1 << 16)
    tail = read(total - tail_len, total - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("end of central directory not found")
    n_entries, cd_size, cd_offset = struct.unpack_from("<HII", tail, eocd + 10)
    loc = eocd - 20
    if loc >= 0 and tail[loc : loc + 4] == b"PK\x06\x07":  # ZIP64
        (eocd64,) = struct.unpack_from("<Q", tail, loc + 8)
        rec = read(eocd64, eocd64 + 55)
        n_entries, cd_size, cd_offset = struct.unpack_from("<QQQ", rec, 32)
    cd = read(cd_offset, cd_offset + cd_size - 1)
    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p : p + 4] == b"PK\x01\x02":
        method, = struct.unpack_from("<H", cd, p + 10)
        crc, csize, usize, nlen, xlen, clen = struct.unpack_from("<IIIHHH", cd, p + 16)
        (offset,) = struct.unpack_from("<I", cd, p + 42)
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
        x, xend = p + 46 + nlen, p + 46 + nlen + xlen
        while x + 4 <= xend:  # ZIP64 extra field holds the real 64-bit values
            hid, hsz = struct.unpack_from("<HH", cd, x)
            if hid == 1:
                q = x + 4
                if usize == 0xFFFFFFFF:
                    (usize,) = struct.unpack_from("<Q", cd, q)
                    q += 8
                if csize == 0xFFFFFFFF:
                    (csize,) = struct.unpack_from("<Q", cd, q)
                    q += 8
                if offset == 0xFFFFFFFF:
                    (offset,) = struct.unpack_from("<Q", cd, q)
            x += 4 + hsz
        m = AUDIO_RE.match(name)
        if m:
            entries.append(Entry(name, m.group(1), m.group(2), method, crc, csize, usize, offset))
        p = xend + clen
    return entries


def extract(read: RangeReader, e: Entry, target: Path) -> None:
    """Download one entry's compressed bytes in chunks and unpack them to ``target``."""
    header = read(e.header_offset, e.header_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {e.name}")
    nlen, xlen = struct.unpack_from("<HH", header, 26)
    start = e.header_offset + 30 + nlen + xlen
    if e.method not in (0, 8):
        raise ValueError(f"unsupported compression method {e.method} for {e.name}")
    inflater = zlib.decompressobj(-15) if e.method == 8 else None
    tmp = target.with_suffix(target.suffix + ".part")
    crc, written, t0 = 0, 0, time.time()
    with open(tmp, "wb") as f:
        for a in range(start, start + e.csize, CHUNK):
            b = min(a + CHUNK, start + e.csize) - 1
            data = read(a, b)
            out = inflater.decompress(data) if inflater else data
            crc = zlib.crc32(out, crc)
            written += len(out)
            f.write(out)
            done = (b - start + 1) / e.csize
            print(f"\r    {e.filename}: {100 * done:5.1f}%  ({(b - start + 1) / 1e6 / max(1e-6, time.time() - t0):.1f} MB/s)", end="", flush=True)
        if inflater:
            rest = inflater.flush()
            crc = zlib.crc32(rest, crc)
            written += len(rest)
            f.write(rest)
    print()
    if written != e.usize or (crc & 0xFFFFFFFF) != e.crc:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{e.filename}: size/CRC mismatch ({written} vs {e.usize} bytes) - please rerun")
    tmp.replace(target)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participants", nargs="+", default=None, help="e.g. P-02 P-05 (default: all)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--list", action="store_true", help="only list the audio files")
    p.add_argument("--url", default=URL)
    args = p.parse_args()

    read, total = http_range_reader(args.url)
    entries = list_entries(read, total)
    if args.participants:
        entries = [e for e in entries if e.participant in args.participants]
    print(f"{len(entries)} audio files, {sum(e.csize for e in entries) / 1e9:.2f} GB to download, "
          f"{sum(e.usize for e in entries) / 1e9:.2f} GB on disk")
    if args.list:
        for e in entries:
            print(f"  {e.participant}/{e.filename}  {e.usize / 1e6:7.1f} MB")
        return
    for e in entries:
        target = args.output_dir / e.participant / e.filename
        if target.exists() and target.stat().st_size == e.usize:
            print(f"  {e.participant}/{e.filename}: already there")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {e.participant}/{e.filename} ({e.csize / 1e6:.0f} MB compressed)")
        extract(read, e, target)
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
