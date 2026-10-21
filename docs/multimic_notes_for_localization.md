# Notes on the three-microphone recordings (for the localization / TDoA work)

These came up while using the extra microphones for the speech detector
(README Steps 13 and 15). They may save time for anyone estimating time
differences of arrival on the same data.

## The data

- The ELTE BARKS "Dog play pant" dataset has raw recordings for all 17
  participants, with up to three microphones on ONE recorder (Zoom H4
  Essential, 48 kHz). They are therefore synchronized:
  - M1: Rode Wireless Go 2 clip-on (wireless);
  - M2: Sennheiser ME66 shotgun;
  - M3: Sennheiser ME64 cardioid.
- Missing microphones: P-01 has no M3; P-05 and P-06 have no M2.
- The annotated play-session files are exact excerpts of one raw microphone:
  M2 for the `-S-` sessions, M3 for the `-G-` sessions (P-05, P-06).
- The preprint gives the room size, 5.46 x 4.41 m, and mentions 7 cameras.
  The microphone positions are **not** given; that would be a question for
  the lab.

## Getting and aligning it

```bash
python -m scripts.download_raw_audio            # ~5 GB, into ../data/elte_barks/raw_audio/
python -m scripts.extract_spatial_features      # aligns every session + level differences and GCC-PHAT per 0.96 s window
python -m scripts.export_aligned_audio          # each microphone's part of every session, 32 kHz, aligned
```

- Alignment: every session file was located inside its raw microphone file by
  envelope cross-correlation, then refined at sample level. All 30 sessions
  match at correlation 1.000, at whole-second offsets (e.g. P-02: 141 s and
  317 s). The offsets are in `outputs/spatial_features/alignment.json`.
- After alignment, the expert annotations apply to all
  microphones, sample-exactly.

## Two things to watch for

1. **Crosstalk at lag 0.** GCC-PHAT shows a sharp peak at exactly lag 0 in
   every microphone pair, even between the wireless clip-on and the wired
   microphones, which cannot be acoustically simultaneous. It is electrical
   crosstalk between the recorder's inputs. PHAT weighting makes even a tiny
   leak dominate. Masking ±1 sample around lag 0 before taking the peak
   fixes it; `extract_spatial_features.py` does this.
2. **The wireless clip-on has its own delay.** Its transmission delay is
   added to every M1 time difference, so M1 delays need a constant offset
   correction before they can be read geometrically.

## What did not work for detection (may still matter for localization)

- Level differences and GCC-PHAT delays between microphones separate human
  from dog sounds within a session (shotgun vs cardioid level difference:
  AUC 0.77). Across participants they do not generalize (AUC 0.66), because
  people and dogs stand in different places in every session.
- For localization this is the expected situation, but it means a model
  trained on some sessions will not transfer positions to others without the
  microphone geometry.
