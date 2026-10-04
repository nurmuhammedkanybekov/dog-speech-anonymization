# What the anonymization has to remove — working definition

*Decided 4 Oct 2026 as our working assumption; to be confirmed with Dr. Kovács
at the next consultation. Everything below is built so it can be changed
later without redoing the pipeline.*

## Goal
Remove human vocalizations that carry **linguistic content or could reveal
what was said**, while keeping dog vocalizations as intact as possible.

## Three tiers

| tier | examples | requirement |
|---|---|---|
| **A — must remove** | words, names, commands ("come here", "good boy", "no", "hey"), sentences, numbers | high recall (target ≥ 95% of tier-A time); a miss leaks content |
| **B — should remove** | voiced non-speech that is clearly a person's voice: voiced laughter, "ah!", "oh!", "wow" | best effort; reported separately |
| **C — not a target** | kissing / clicking noises, whistles, coughs, claps, footsteps, handling noise | left untouched; not counted as misses |

## Why this scope
- Anonymization is about content and identity. Removing *every* human-made
  sound turns the project into general human-sound detection, which is not
  well defined and not what privacy requires.
- The ELTE "Human" annotation covers all three tiers in one label, so a
  single recall number mixes them. Each human event is being tagged with its
  tier by listening (`outputs/labeling/`), and detection is then reported per
  tier.

## Trade-off rule
For tier A, prefer recall over precision: a missed word is a privacy leak; a
false alarm only costs a little dog audio — and with gated subtraction
instead of muting, that cost is small.

## How privacy is measured (not with the detector itself)
The detector that decides where speech is must not also be the judge of
whether speech is gone. Privacy is measured independently: speech energy
left (ground truth available in synthetic mixtures), intelligibility (STOI),
automatic speech recognition word error rate (PocketSphinx), and listening.
