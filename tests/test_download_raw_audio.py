"""Tests for scripts.download_raw_audio against a local ZIP64 archive (no network)."""

from __future__ import annotations

import os
import zipfile

from scripts.download_raw_audio import extract, list_entries


def _make_zip(path):
    payload = {
        "02 Raw recordings/P-01/Audios/P-01-M1.wav": os.urandom(300_000) + b"\0" * 300_000,
        "02 Raw recordings/P-01/Audios/P-01-M2.WAV": b"RIFF" + os.urandom(1000),
        "02 Raw recordings/P-01/Videos/P-01_cam-1.mp4": os.urandom(5000),
        "__MACOSX/02 Raw recordings/P-01/Audios/._P-01-M1.wav": b"x" * 212,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in payload.items():
            with z.open(name, "w", force_zip64=True) as f:
                f.write(data)
    return payload


def test_lists_and_extracts_only_audio(tmp_path):
    zpath = tmp_path / "raw.zip"
    payload = _make_zip(zpath)
    blob = zpath.read_bytes()

    def read(a, b):
        return blob[a : b + 1]

    entries = list_entries(read, len(blob))
    assert sorted(e.filename for e in entries) == ["P-01-M1.wav", "P-01-M2.WAV"]
    for e in entries:
        target = tmp_path / e.participant / e.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        extract(read, e, target)
        assert target.read_bytes() == payload[e.name]
