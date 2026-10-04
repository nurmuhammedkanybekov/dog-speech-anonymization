"""Tests for vad_anonymization.elte_barks (annotation parsing and masks)."""

from __future__ import annotations

from vad_anonymization.elte_barks import Event, events_to_mask, read_expert_annotations

SAMPLE = (
    "tmin\ttier\ttext\ttmax\r\n"
    "0.070177\tnoise\tHuman\t3.167055\r\n"
    "3.537210\tcalls\tPant\t3.802295\r\n"
    "4.000000\tcalls\tBark/yelp\t4.500000\r\n"
    "5.000000\tnoise\tDoor\t5.500000\r\n"
)


def test_parses_human_and_dog_tiers(tmp_path):
    f = tmp_path / "P-01-S-P_1.Table.txt"
    f.write_text(SAMPLE, encoding="utf-8")
    human, dog = read_expert_annotations(f)
    assert human == [Event(0.070177, 3.167055, "Human")]
    assert [e.label for e in dog] == ["Pant", "Bark/yelp"]  # non-human noise rows are ignored


def test_events_to_mask_with_padding():
    events = [Event(1.0, 1.5, "Human")]
    mask = events_to_mask(events, n_frames=300, frame_s=0.01)
    assert mask[100:150].all() and not mask[:100].any() and not mask[150:].any()
    padded = events_to_mask(events, n_frames=300, frame_s=0.01, pad_s=0.2)
    assert padded[80:170].all() and not padded[:80].any()
