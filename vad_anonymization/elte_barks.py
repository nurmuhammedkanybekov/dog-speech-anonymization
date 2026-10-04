"""Loader for the ELTE BARKS Lab "Dog play pant" dataset (Zenodo 18972388).

30 dog-human social play sessions recorded at the Department of Ethology,
ELTE (Budapest), 48 kHz WAV, with expert ("second layer") annotations
exported from Praat: one row per sound with ``tmin``, ``tier``, ``text``,
``tmax``. Tier ``calls`` = dog sounds (Pant, Growl, Grunt, Bark/yelp, ...);
tier ``noise`` with text ``Human`` = human vocalizations.

Cite: Cuaya, L. V. et al. (2026). Dog play pant: An annotated dataset of dog
vocalizations during dog-human social play. Zenodo.
https://doi.org/10.5281/zenodo.18972388 (CC BY-NC 4.0 per the readme).

Expected layout (see the project README):
    data/elte_barks/play_sessions/P-01-S-P_1.wav ...
    data/elte_barks/annotations_expert/P-01-S-P_1.Table.txt ...
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(frozen=True)
class Event:
    start_s: float
    end_s: float
    label: str  # "Human" or a dog call type such as "Pant"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Session:
    name: str
    audio_path: Path
    human: list[Event] = field(default_factory=list)
    dog: list[Event] = field(default_factory=list)


def read_expert_annotations(path: Path) -> tuple[list[Event], list[Event]]:
    """Return (human events, dog call events) from a 2nd-layer annotation file."""
    human, dog = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = (row.get("text") or "").strip()
            tier = (row.get("tier") or "").strip()
            event = Event(float(row["tmin"]), float(row["tmax"]), text)
            if tier == "noise" and text.lower() == "human":
                human.append(event)
            elif tier == "calls":
                dog.append(event)
    return human, dog


def load_sessions(root: Path) -> list[Session]:
    sessions = []
    for wav in sorted((root / "play_sessions").glob("*.wav")):
        ann = root / "annotations_expert" / f"{wav.stem}.Table.txt"
        if not ann.exists():
            continue
        human, dog = read_expert_annotations(ann)
        sessions.append(Session(wav.stem, wav, human, dog))
    return sessions


def events_to_mask(events: list[Event], n_frames: int, frame_s: float, pad_s: float = 0.0) -> torch.Tensor:
    """Boolean frame mask that is True inside any event (optionally widened by ``pad_s``)."""
    mask = torch.zeros(n_frames, dtype=torch.bool)
    for e in events:
        a = max(0, int((e.start_s - pad_s) / frame_s))
        b = min(n_frames, int(round((e.end_s + pad_s) / frame_s)))
        mask[a:b] = True
    return mask
