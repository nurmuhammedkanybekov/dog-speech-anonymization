# Speech Anonymization for Multi-Microphone Dog Recordings

**Task 1 of 3** in the dog-sound-recording analysis project (ELTE Modeling
Laboratory, supervised by Dr. Péter Kovács): detect and suppress human
speech in the lab's recordings so they can be anonymized before further
use. 
The other two tasks (Sound-source localization and Echo cancellation) both of them are out of scope for this sub-repository.

## Approach

The goal is to remove human speech from the recordings while keeping the
dog vocalizations intact.

1. **Detect human sounds.** Silero VAD (pretrained) works on clean
   mixtures but finds only ~5% of the human sounds in real lab recordings, so
   a detector trained on the lab's own annotations (frozen VGGish embeddings
   + logistic regression) is used instead (Step 9).
2. **Remove them, keep the dog: hybrid removal.** Muting deletes the dog
   wherever it overlaps speech. Inside detected human segments the hybrid
   therefore *mutes* where no dog vocalizes (71% of human time in the real
   data) and *subtracts* a pretrained speech estimate (DTLN) only where a dog
   overlaps (Steps 4 and 6).
3. **Check privacy independently.** An ASR model (Whisper) acts as the
   attacker and tries to read words back from the anonymized audio
   (Steps 7–8). A speech detector reporting "no speech left" is not used as
   proof, since a similar detector decided what to remove.

Every step was first tested on synthetic mixtures with known ground truth
(SNR-controlled, Steps 2–4) and then on real recordings (ELTE BARKS Lab
"Dog play pant", Steps 5–11). The real-data results are the reference. See
`docs/target_definition.md` for what counts as "speech" to remove.

## Project layout

```
speech_anonymization/
├── vad_anonymization/          # reusable library code
│   ├── audio_io.py             # load / resample / save audio as tensors
│   ├── detection.py            # SpeechDetector: wraps Silero VAD
│   ├── mixing.py               # overlap tests, SNR-controlled mixing, 10 s scenes
│   ├── enhancers.py            # pretrained speech estimators: DTLN, GTCRN, DeepFilterNet3
│   ├── anonymize.py            # mute, (gated) subtraction, hybrid removal
│   ├── metrics.py              # speech leakage, STOI, dog retention, SDR
│   ├── elte_barks.py           # ELTE BARKS Lab sessions + expert annotations
│   ├── asr.py                  # Whisper (sherpa-onnx) as an independent privacy check
│   └── vggish_features.py      # VGGish embeddings for the trained detector
├── scripts/                    # one script per step (see Usage), plus plots and downloads
├── tests/                      # pytest unit tests (offline, synthetic signals)
├── docs/target_definition.md   # what counts as "human speech" to remove (tiers A/B/C)
├── reports/Progress_Report.docx
├── models/                     # downloaded weights (git-ignored)
└── outputs/                    # generated results (git-ignored; real voices stay local)
```

Raw audio inputs live one directory up, in `../data/`, shared between
this codebase and the exploratory scripts that came before it. The
first working versions of this logic were written by hand as two
linear scripts to prove the approach end-to-end before this codebase
existed; their full content (including the errors hit and how they
were resolved) is preserved verbatim in the author's own working
notes (kept separately, outside this repository) rather than kept as
separate, now-duplicate files.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt   # or requirements.txt for runtime-only
```

## Usage

Run all commands from inside this directory (`speech_anonymization/`),
so that `vad_anonymization` is importable as a local package.

**Step 1 — detect and mute speech in a clip:**

```bash
python -m scripts.run_speech_detection ../data/sample_speech.wav muted.wav
```

**Step 2 — test detection under a dog-bark overlap:**

```bash
python -m scripts.run_overlap_test
```

Defaults use the Session 1/2 sample files (`../data/sample_speech.wav`
and `../data/dog_bark_esc50_1-100032-A-0.wav`); pass `--speech`,
`--noise`, `--excerpt-start`, `--excerpt-end`, or `--noise-offset` to
test other clips or timings. Run `python -m scripts.run_overlap_test
--help` for the full option list. Output audio and a `results.json`
summary are written to `outputs/`.

**Step 3 — SNR sweep (how loud does the dog have to be to break detection?):**

```bash
pip install matplotlib                  # plotting only
python -m scripts.download_esc50_dogs   # all 40 ESC-50 dog clips -> ../data/esc50_dog/
python -m scripts.run_snr_sweep         # 42 clips x 9 SNR levels -> outputs/snr_sweep/
python -m scripts.plot_snr_sweep        # -> outputs/snr_sweep/snr_sweep.png
```

The noise is scaled to a target signal-to-noise ratio
(`mixture = speech + x * noise`, with `x` chosen so that
`10*log10(P_speech / P_noise)` equals the target). Noise power is measured
over the clip's active frames only, since barks are mostly silence with
short bursts. Result so far: speech detection holds down to about −10 dB
and degrades from −20 dB (mean recall 97% at −20, 78% at −40; dense
AudioSet barking degrades much faster), while 31% of dog clips trigger a
false "speech" detection on their own at some level — which, with
detect-and-mute, would delete dog vocalizations from the data.

**Step 4 — remove the speech, keep the dog (method comparison):**

```bash
pip install -r requirements-separation.txt   # pystoi + deepfilternet
python -m scripts.download_models            # DTLN, GTCRN, DeepFilterNet3 weights from GitHub -> models/
python -m scripts.run_method_comparison      # 42 dog clips x 3 SNRs, 10 s scenes -> outputs/method_comparison/
python -m scripts.plot_method_comparison     # -> outputs/method_comparison/privacy_vs_dog.png
```

Muting deletes the dog wherever it overlaps speech. The alternative tested
here estimates the speech with a pretrained speech-enhancement model and
subtracts it (`y = x - s_hat`), only inside VAD-detected speech ("gated").
Each 10 s test scene has speech (0–5 s), dog (2.5–7.5 s) and silence, and is
scored for privacy (speech left, STOI intelligibility, VAD) and for the dog
(SDR against the clean dog). At 0 dB: muting removes speech fully but keeps
the dog at ~0 dB SDR where it overlaps speech; gated DTLN subtraction leaves
speech at −31 dB (STOI 0.05, VAD finds none) while keeping the dog at +7 dB
SDR. Listen to `outputs/method_comparison/examples/`.

**Step 5 — real recordings (ELTE BARKS Lab "Dog play pant" dataset):**

Data: `../data/elte_barks/` (30 dog–human play sessions + expert annotations,
Zenodo 18972388, CC BY-NC 4.0) and `../data/librispeech_subset/` (80 clips,
40 speakers, LibriSpeech test-clean, CC BY 4.0).

```bash
python -m scripts.eval_real_detection          # Silero VAD vs expert "Human" annotations
python -m scripts.analyze_real_human_events    # per-event loudness / length / speech probability
python -m scripts.plot_real_detection
python -m scripts.extract_elte_dog_windows     # 88 real 5 s dog-only windows (mostly panting)
python -m scripts.run_method_comparison --speech-dir ../data/librispeech_subset \
       --dog-dirs ../data/elte_barks/dog_windows --output-dir outputs/method_comparison_elte
```

Findings: on the real sessions Silero VAD flags only ~5% of annotated
human-vocalization time (threshold 0.5; 96% of what it flags is correct, and
it almost never fires on dog sounds). Most annotated human sounds are short
(<1 s) and very quiet (below −60 dBFS) — the microphone points at the dog.
Speech subtraction still works on real lab dog sounds: with 40 speakers mixed
into real panting at 0 dB, gated DTLN cuts speech by ~27 dB while keeping the
dog at +8 dB SDR where they overlap (muting: ~0 dB).

**Step 6 — oracle test: if detection were perfect, does removal work on real audio?**

```bash
python -m scripts.run_oracle_real      # -> outputs/oracle_real/ (summary.csv, per_event.csv, examples/)
```

The expert "Human" annotations stand in for a perfect detector; the speech
estimate is subtracted only inside them. Real recordings have no clean
reference, so only measurable things are reported: how much the energy
inside each event drops, and how far it stays above the local background
noise floor. Result: subtraction takes the real human sounds down by only
2–4 dB (median), and only 16% of audible human-only events reach the noise
floor (DTLN; GTCRN 8%). On LibriSpeech mixtures the same models removed
26–35 dB. The pretrained enhancers do not recognize these faint,
distant, mostly non-speech sounds as "speech", so removal on real audio is a
second bottleneck, next to detection. Loudness-normalizing the input first
did not help (it made it slightly worse).

But 71% of annotated human time has no dog call under it, and there muting
costs nothing (there is no dog to keep). `anonymize.hybrid_remove` therefore
mutes human-only time and subtracts only where a human overlaps a dog call.

**Step 7 — independent privacy check: can Whisper still read the words?**

```bash
pip install -r requirements-research.txt
python -m scripts.download_models --asr        # Whisper small / small.en (sherpa-onnx int8, from GitHub)
python -m scripts.run_asr_privacy              # LibriSpeech scenes -> outputs/asr_privacy/
python -m scripts.plot_asr_privacy
```

"VAD finds no speech left" is a circular argument (the same kind of model
decided what to remove). Here OpenAI Whisper (small.en), a much stronger
model, plays the attacker and transcribes every method's output; the
reference is its transcript of the clean speech. Same 42 scenes × 3 SNRs as
Step 4.

| method | words read back (+10 / 0 / −10 dB) | scenes with 0 words |
|---|---|---|
| no processing | 98% / 97% / 92% | 0% |
| VAD + mute | 1% / 1% / 1% | 86–88% |
| DTLN subtraction | 0% / 0% / 0% | 100% |
| gated DTLN subtraction | 0% / 0% / 0% | 95% |
| gated GTCRN subtraction | 1% / 2% / 0% | 81–95% |
| gated DeepFilterNet3 subtraction | 87% / 76% / 54% | 0–14% |

The few "recovered" words are a first word slipping in before the VAD
onset ("How", "He") or Whisper hallucinations on noise ("This is a video
about the…"). DTLN subtraction leaves nothing readable; DeepFilterNet3
leaves most sentences readable and is ruled out.

**Step 8 — Whisper on the real recordings**

```bash
python -m scripts.run_asr_real --method dtln   # -> outputs/asr_real/ (transcripts: keep local)
```

Multilingual Whisper (language auto-detected) transcribes each annotated
human event before and after oracle removal, plus a same-length background
window with no human or dog (to measure how often Whisper invents words from
noise). All windows are normalized to the same loudness first, as an
attacker would turn the volume up.

Share of the 581 events where Whisper reads 2 or more words (oracle
detection, DTLN):

| | original | subtraction | hybrid (mute + subtract) | background only (noise floor) |
|---|---|---|---|---|
| all events | 23% | 20% | 7% | 4% |
| events > 1 s | 36% | 36% | 11% | 2% |
| human only (no dog overlap) | 21% | 17% | 0.6% | 5% |
| human + dog overlap | 26% | 24% | 17% | 3% |

Of the words Whisper read in the original (events with 2+ words),
subtraction leaves 35% readable and hybrid 9%. So on real audio:
subtraction alone barely changes what can be understood, while hybrid
removal brings human-only events to zero. The remaining risk is longer
speech that overlaps a dog call, where subtraction is still weak. Some
real sentences survive there. On short, faint events Whisper's
language detection jumps around (Japanese, Russian, Portuguese on the same
sessions), so many of those "words" are guesses; the background windows
show that floor.

**Step 9 — a detector trained on the lab's own annotations**

```bash
python -m scripts.download_models --detector   # VGGish weights from GitHub
python -m scripts.train_human_detector         # -> outputs/human_detector/
python -m scripts.plot_human_detector
```

Frozen VGGish embeddings (pretrained on AudioSet) every 0.24 s, with the
neighbouring frames as context, and a logistic regression on top. It is
evaluated leave-one-participant-out (17 folds), so it is always tested on a
person and dog it never saw. Target for now: all annotated human sounds.

| operating point | human time caught | events touched | dog-call time flagged |
|---|---|---|---|
| Silero VAD (0.5) | 4.5% | 1.4% | 0.0% |
| detector, strict | 50% | 56% | 6% |
| detector, balanced | 70% | 82% | 25% |
| detector, high recall | 90% | 95% | 66% |

ROC AUC 0.80. That is a big step up from Silero, but at high recall it
also flags much of the dog time. A flag on dog time means "subtract here",
not "mute here", and in Step 4 subtraction cost the dog about 2 dB. Once the
human events are labeled by type (Step 10), the detector is retrained on
real speech only (tier A), which should be an easier target than every
kiss, click and laugh.

**Step 10 — label the human sounds by type (manual)**

```bash
python -m scripts.make_labeling_page --asr-results outputs/asr_real/per_event.csv
# -> outputs/labeling/label_human_events.html + clips/ + audit_clips/
```

With the Whisper results the page is ordered so ~30 min covers what matters:
206 priority events (Whisper read 2+ words, or longer than 1.5 s), a random
spot-check of 30 short events Whisper found nothing in, and a leak audit of the
79 hybrid outputs where Whisper still read something (real words vs. Whisper
hallucination). The remaining short events are optional.

Open `outputs/labeling/label_human_events.html` in a browser, listen to each
of the 581 human events (made louder so they can be heard), and press 1–5:
speech / voiced non-speech / non-vocal human noise / no human audible /
unsure. Export the CSV to `outputs/labeling/human_event_labels.csv`, then:

```bash
python -m scripts.train_human_detector --labels outputs/labeling/human_event_labels.csv \
       --tiers A_speech --output-dir outputs/human_detector_tierA
```

See `docs/target_definition.md` for the tiers.

**Step 11 — can the hybrid work without the expert annotations?**

```bash
python -m scripts.eval_hybrid_routing   # dog detector (same VGGish method) + routing check -> outputs/hybrid_routing/
```

The hybrid needs to know where a dog vocalizes. A dog detector trained the
same way (leave-one-participant-out) reaches ROC AUC 0.77; most dog sounds here
are quiet panting. Routing inside the annotated human time:

| dog threshold | human-only time → mute | human + dog time → subtract |
|---|---|---|
| 0.3 | 50% | 79% |
| 0.5 | 71% | 65% |
| 0.7 (privacy-first) | 89% | 43% |

So the oracle hybrid is an upper bound; a deployable hybrid needs a better dog
detector or a privacy-first rule ("when unsure, mute").

**Step 12: the full pipeline on real recordings, with no expert labels**

```bash
python -m scripts.run_pipeline_real --worker 0 --n-workers 2   # and --worker 1 in parallel (~2 h on 2 CPUs)
python -m scripts.run_pipeline_real --summarize-only
python -m scripts.plot_pipeline_real   # -> outputs/pipeline_real/privacy_vs_dog_loss.png
```

Until now the real-data privacy numbers assumed perfect detection (the
"oracle"). Here the trained human detector (Step 9) and the trained dog
detector (Step 11) do everything, using their leave-one-participant-out
predictions, so each session is processed by models that never saw that
participant. The expert annotations are used only for scoring:

- privacy: Whisper, as in Step 8, on all 581 human events;
- dog: the share of dog-vocalization time without a human that gets muted
  (lost) or subtracted (slightly altered).

| configuration | events with 2+ readable words | long events (> 1 s) | dog time muted | dog time subtracted |
|---|---|---|---|---|
| no processing | 22.9% | 36.3% | 0% | 0% |
| Silero VAD + mute (original approach) | 21.5% | 32.1% | 0% | 0% |
| automatic hybrid, strict | 10.3% | 12.1% | 3.6% | 5.4% |
| automatic hybrid, balanced | 11.0% | 14.2% | 11.3% | 21.3% |
| automatic hybrid, balanced, privacy-first dog threshold | 8.1% | 8.9% | 19.3% | 13.4% |
| automatic hybrid, high recall | 9.3% | 13.2% | 24.2% | 50.8% |
| **trained detector + mute, strict** | **7.6%** | **7.4%** | **9.0%** | 0% |
| **trained detector + mute, balanced** | **3.3%** | **2.6%** | 32.7% | 0% |
| oracle hybrid (expert labels, Step 8) | 7.2% | 11.1% | n/a | n/a |
| background only (Whisper hallucination level) | 4.1% | 1.6% | | |

Findings:

1. The original approach gives no privacy on real recordings (21.5% vs. 22.9%).
2. With the trained detector, simply muting what it finds works well:
   - at the strict setting it matches the oracle hybrid (7.6%) while losing
     9% of dog time;
   - at the balanced setting it reaches Whisper's own hallucination level
     (3.3% vs. 4.1%), but a third of the dog time is lost.
3. The automatic hybrid does *worse* than plain muting at the same dog cost.
   The dog detector sends much of the human speech to subtraction, and
   subtraction barely works on these recordings (Step 6). The hybrid only pays
   off once removal in the overlap works on real far-field speech.

So the practical recommendation for now is trained detector + mute, with
the strict/balanced setting chosen by how much dog sound may be lost. The
research problem that remains is removing speech where it overlaps a dog.

Caveats:

- The thresholds were picked on the pooled out-of-fold scores (slightly
  optimistic).
- Whisper is a proxy for a human listener.
- Both detectors still need the lab's annotations for training (from other
  participants).

**Step 13: the raw recordings have three synchronized microphones**

```bash
python -m scripts.download_raw_audio --list   # 48 audio files, 2.04 GB, out of the 43 GB raw zip
python -m scripts.download_raw_audio          # downloads only the audio, using HTTP range requests
bash scripts/prepare_raw_audio.sh             # -> ../data/elte_barks/raw16k/P-xx/*.flac (16 kHz mono)
```

The Zenodo "02 Raw recordings.zip" is 43 GB, almost all video. The audio
inside is small. Per participant it has:

- M1: Rode Wireless Go 2 clip-on, all 17 participants;
- M2: Sennheiser ME66 shotgun, 15 participants (not P-05/P-06);
- M3: Sennheiser ME64 cardioid, 16 participants (not P-01).

All three were recorded on one Zoom H4 Essential. Checked on P-02:

- The three files have identical length, which fits one recorder and one
  clock, i.e. synchronized.
- The session files we use are exact excerpts of M2: envelope correlation
  0.999, at 141.0 s and 317.0 s.
- M1 and M3 are stereo files with the same signal in both channels.
- The clip-on is not worn by the person. Human sounds are not louder on M1
  than on the other microphones, so it gives no clean speech reference.
- The level ratio between microphones differs between human and dog sounds
  (the dog is closer to M3). On its own, this level difference separates
  human from dog events with AUC 0.67–0.69 on P-02, a cue that is
  independent of the sound itself.

Per the dataset preprint (bioRxiv 10.64898/2026.04.20.719471), the room is
the 5.46 m x 4.41 m BARKS audio lab, with 7 synchronized cameras. Exact
microphone positions are not given.

Next uses:

- inter-microphone level and time-difference features for the human/dog
  detectors;
- spatial filtering;
- TDoA localization (Ivannova's task).

## Testing

```bash
pytest
```

Tests cover the pure logic in `vad_anonymization` (resampling
correctness, mono conversion, mixture timing and clipping behavior)
using synthetic tones and temporary files, so they run offline and
don't depend on the project's specific audio samples being present.
`SpeechDetector.detect()` itself — the part that actually calls Silero
VAD — is exercised as a manual, documented run in the author's own
working notes instead of here, since it
requires downloading the real model and its output depends on real
audio content, making it an integration check rather than a
deterministic unit test.

## Data and licensing

- `../data/sample_speech.wav` — a public speech sample used to validate
  detection on plain speech (Session 1).
- `../data/dog_bark_esc50_1-100032-A-0.wav` — a real dog bark, one clip
  from the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50)
  (Environmental Sound Classification, category "dog"). Licensed
  CC BY-NC 3.0, which covers this non-commercial research use.
  Citation: Karol J. Piczak, *"ESC: Dataset for Environmental Sound
  Classification,"* Proceedings of the 23rd ACM International
  Conference on Multimedia, 2015.

  **Known limitation:** this specific clip contains a single ~0.2s
  bark burst inside its fixed 5-second length (every ESC-50 clip is
  exactly 5s by dataset design), not sustained or repeated barking.
  Addressed below (Round 2) with a genuinely sustained-bark ESC-50
  clip.
- `../data/dog_bark_esc50_1-30344-A-0.wav` — the sustained-bark ESC-50
  clip used in Round 2 below (7 bark bursts spread across its full 5s).
- `../data/audioset_candidates/audioset_dog_bark_*.flac` — two real
  dog-bark clips from [AudioSet](https://research.google.com/audioset/)
  (10s each, 48kHz stereo), Google's large-scale audio-event dataset.
  Citation: Gemmeke et al., *"Audio Set: An Ontology and Human-Labeled
  Dataset for Audio Events,"* ICASSP 2017. Used in Rounds 3–4 below to
  test against real-world recordings rather than curated sound-effect
  clips.

None of these files are the lab's own recordings; all were used to
validate the pipeline mechanics before any lab data is involved.

## Open assumptions (pending confirmation from Dr. Kovács)

- **Dataset**: the professor's verbal mention of a public dataset for
  this task is assumed to refer to **AudioSet** (its ontology has both
  a "Human voice"/"Speech" branch and a "Dog" branch), but this has not
  yet been confirmed.
- **Success metric**: "anonymized" is currently operationalized as
  "flagged and muted by the detector"; whether that (vs. a stricter
  quantitative metric, e.g. measured speech-energy reduction or a
  classifier confirming no detectable speech) is the right bar for
  success has not yet been confirmed.

## Results — Step 2 overlap test

Five noise conditions have now been tested against the same 12.3s
speech excerpt, from simplest to most realistic:

| Condition | Coverage | Gap | False positive (Check B) |
|---|---|---|---|
| ESC-50, single short burst (~0.2s) | 99% | None | None |
| ESC-50, sustained barking (7 bursts across 5s) | 99% | None | None |
| Real-world AudioSet clip A | 98% | 0.10s, inside overlap window | **Yes** — 0.3s of the bark alone flagged as speech |
| Real-world AudioSet clip B | 99% | None | None |
| Both AudioSet clips combined (overlaid on each other) | 99% | None | None |

**Round 1 (ESC-50, short burst):** one continuous detected segment,
0.10s–12.30s. Only the first 0.10s (unrelated to the dog) went
undetected.

**Round 2 (ESC-50, sustained, 7 bursts across the full 5s):** identical
result to Round 1 — detection held up cleanly under repeated
interference from this clip.

**Round 3 (real-world AudioSet clips) — the most informative result so
far:** one of the two AudioSet clips broke both checks. Detected speech
split into two segments (0.10s–8.30s and 8.40s–12.30s) — a genuine
0.10s coverage gap inside the dog-overlap window — and separately, 0.3s
of that same bark played alone (zero speech) was flagged as speech: a
real false positive. The second AudioSet clip passed cleanly under
identical conditions. This is the first genuine detection failure found
in this project, and it only appeared on real-world audio — neither
ESC-50 clip produced it, including the sustained one. That's a concrete
data point for whether ESC-50 is an adequate stand-in for this task's
eventual real recordings.

**Round 4 (both AudioSet clips overlaid together):** passed cleanly,
including the clip that had failed alone in Round 3. Caveat: the
combined signal was peak-normalized to avoid clipping, which lowered
its overall loudness relative to either clip alone — this likely
diluted whatever specific feature caused Round 3's failure, so this is
not read as evidence that combining sounds makes detection more
robust; it's flagged as a probable normalization artifact rather than
a finding.
