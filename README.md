# Speech Anonymization for Multi-Microphone Dog Recordings

**Task 1 of 3** in the dog-sound-recording analysis project (ELTE Modeling
Laboratory, supervised by Dr. Péter Kovács): detect and suppress human
speech in the lab's recordings so they can be anonymized before further
use. 
The other two tasks (Sound-source localization and Echo cancellation) both of them are out of scope for this sub-repository.

## Current approach and results (25 October 2026)

The goal is to remove human speech from the recordings while keeping the
dog vocalizations intact. The recommended pipeline needs no annotations at
run time:

1. **Detect human sounds** with two frozen networks: Whisper small's encoder
   (trained on 680,000 hours of speech) and EfficientAT `mn40_as` (527
   AudioSet sound classes). A logistic regression, trained on the lab's
   expert annotations to tell human sounds from dog vocalizations, with every
   dog call type weighted equally, combines them; its output is smoothed over
   5 frames of 0.24 s (Steps 25-26).
2. **Mute** the flagged time (±0.12 s) in the original file; everything else
   stays bit-identical.
3. **Optionally keep the dog under speech** (`--keep-dog`, Steps 27-28): inside
   the muted time, where a dog router hears a dog call, a dog extractor's
   output (human sound removed) replaces the silence (`safe`, `more`); with
   `all`, the extractor's output fills all the muted time and steady room
   tone covers what is left of the voice where the router hears no dog.
4. **Check privacy independently** with three speech recognizers as
   attackers: Whisper small, Whisper large-v3-turbo and NVIDIA Parakeet (a
   different model family), on every annotated human event.

Results on the ELTE BARKS play sessions (581 human events, 17 dogs, every
model tested on dogs it never saw; Whisper small as the attacker):

| setting | human events with 2+ readable words | original words still readable | all dog sound muted | barks / play pants untouched |
|---|---|---|---|---|
| no processing | 19.8% | 100% | 0% | 100% / 100% |
| previous default (Step 17, `standard`) | 1.4% | 5.6% | 16.4% | 75% / 71% |
| **`standard` (default, Step 26)** | **1.0%** | 5.2% | 12.1% | **100% / 82%** |
| `standard --keep-dog safe` (Step 27) | 1.0%* | 5.2%* | **9.8%** | 100% / 82% + part of the calls under speech |
| `standard --keep-dog more` | 1.0%* | 5.2%* | 8.2% | |
| `standard --keep-dog all` (Step 28) | 2.6%† | 5.2%* | **0%**† | every call in the muted time comes back, a third of them clearly above the added room tone |
| `light` / `strict` | 1.7% / 1.0% | 6.2% / 4.3% | – | |

\* the same words read back as with muting for Whisper small; Parakeet and
turbo read a few more (Steps 27-28: for `all`, laughter, "mm-hmm" and "thank
you", no new words for Whisper small, Parakeet or a DeepFilterNet-cleaning
attacker; turbo +1.4 points). `--keep-dog` stays off by default until a person
has listened to the 12 clips in `outputs/listening_overlap/listen.html`.

† with `all` nothing is muted, but the pants in the muted time (the quietest
call type) come back at about the level of the added room tone (median within
1 dB); louder calls stand out more. The 2.6% is from the Step 28 run, where
muting gives 0.9% and room tone alone 1.4%: Whisper writes 2+ words for room
tone more often than for silence, while the words it reads back from the
original stay at muting's level.

- Whisper "reads" 2+ words in 1.0% of background-only windows: `standard`
  reaches that floor.
- Compared with Step 17: a third less dog sound lost at the same privacy (95%
  CI of the reduction 21–51%), all barks kept, and with `--keep-dog` part of
  the dog sound under speech comes back.
- On synthetic recordings with unseen speakers, unseen dogs and simulated
  rooms (Steps 20–21, 25), `standard` leaves 0 of 120 utterances readable.
- Re-checks (Step 22): no unannotated speech outside the annotated events; a
  bigger attacker does not change the conclusions.

**Anonymize a recording** (after the setup below, `git clone https://github.com/fschmid56/EfficientAT models/EfficientAT`
and `python -m scripts.download_models --asr`):

```bash
python -m scripts.anonymize_recording input.wav anonymized.wav                      # standard
python -m scripts.anonymize_recording input.wav anonymized.wav --keep-dog safe      # + dog under speech
python -m scripts.anonymize_recording input.wav anonymized.wav --keep-dog all       # nothing muted (Step 28)
```

The command runs at about 8x real time on 2 CPU cores and keeps the sample
rate, channels and bit depth of the input. It needs `models/human_detector_v3.npz`
(`python -m scripts.train_final_detector`) and, for `--keep-dog`,
`models/dog_router_v1.npz` and `models/dog_extractor/dog_extractor_w_all.pt`
(Step 27). The models and the anonymized copies of the dataset
(`data/elte_barks/anonymized_v3/`) are kept out of git until the dataset
question is settled.

A paper draft with these results is in `paper/` (`main.pdf`).

Every step, with its commands and full results, is below (Steps 1–28). The
early steps describe the approaches that were tried and replaced: Silero VAD,
VGGish, hybrid removal with DTLN, and EfficientAT alone.

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
│   ├── vggish_features.py      # VGGish embeddings (Steps 9-15)
│   └── gtcrn_model.py          # GTCRN speech remover, trainable (Step 14)
├── scripts/                    # one script per step (see Usage), plus plots and downloads
├── tests/                      # pytest unit tests (offline, synthetic signals)
├── docs/target_definition.md   # what counts as "human speech" to remove (tiers A/B/C)
├── docs/multimic_notes_for_localization.md  # the three synchronized microphones (for the TDoA work)
├── reports/                    # progress reports (5 Oct, 21 Oct, 22 Oct 2026)
├── paper/                      # paper draft (LaTeX source, figure, PDF)
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
python -m scripts.download_raw_audio          # downloads only the audio (HTTP range requests) -> ../data/elte_barks/raw_audio/
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

**Step 14: a better speech remover for far-field speech (GTCRN, fine-tuned for the lab)**

```bash
python -m scripts.finetune_gtcrn --train-group odd    # ~40 min on 1 CPU core; and --train-group even in parallel
python -m scripts.eval_finetuned_gtcrn --worker 0 --n-workers 2   # and --worker 1 (~1.5 h)
python -m scripts.eval_finetuned_gtcrn --summarize-only           # -> outputs/finetuned_gtcrn/
```

Removal was the second bottleneck: the pretrained models take only 2–4 dB off
the real human sounds (Step 6). GTCRN (Rong et al. 2024; 48k parameters; MIT
license; the authors' DNS3 weights as the starting point) was fine-tuned on
mixtures that imitate the lab:

- speech: LibriSpeech convolved with simulated room impulse responses of a
  5.46 m x 4.41 m room (pyroomacoustics; RT60 0.15–0.6 s, speaker 0.8–4 m from
  the microphone);
- background: real audio from the ELTE sessions, away from human
  annotations, at its real (very quiet) level;
- speech-to-background ratio from −10 to +15 dB, and 15% of examples with no
  speech (then the target is silence, so the dog is left alone);
- frozen BatchNorm statistics, 1,500 steps.

The simulation reproduces the real problem: the pretrained model removes only
~3–4 dB of the simulated far-field speech, just as on the real recordings.
Two models were trained on separate participant groups (odd- vs.
even-numbered participants). Every test below uses only the group the model
never saw.

| remover | synthetic: speech left | synthetic: background kept | synthetic: output SDR | real, oracle hybrid: events with 2+ readable words | ...of which overlap a dog | original words still readable |
|---|---|---|---|---|---|---|
| DTLN (used in Steps 4–12) | −2.7 dB | −3.2 dB | 0.8 dB | 7.1% | 16.7% | 8.5% |
| GTCRN, pretrained | −12.6 dB | −2.6 dB | 4.3 dB | 3.6% | 8.1% | 4.5% |
| **GTCRN, fine-tuned** | **−15.8 dB** | **−1.7 dB** | **6.7 dB** | **3.6%** | **8.1%** | **3.7%** |
| GTCRN, fine-tuned + 2x spectral over-subtraction | −17.5 dB | −2.9 dB | 6.1 dB | 3.6% | 8.1% | 4.4% |

(Whisper's own hallucination level on background-only audio is 4.1%.)

Findings:

1. DTLN, which looked best on close-talk synthetic speech (Steps 4 and 7), is
   the worst in far-field conditions. Replacing it with GTCRN halves the
   readable speech left where a human overlaps a dog (16.7% → 8.1% of
   overlap events). This is statistically significant: on the same events,
   GTCRN fixes 31 of DTLN's leaks and adds 11 new ones (McNemar p = 0.003).
2. With GTCRN, the oracle hybrid reaches Whisper's hallucination level (3.6%
   vs. 4.1%). About half of the remaining overlap "leaks" are events where
   Whisper could not read anything in the original either, i.e. mostly
   hallucinations on dog sounds.
3. Fine-tuning gives clear gains on the held-out synthetic test: 3 dB more
   speech removed, 1 dB more of the dog kept, +2.4 dB SDR. On the real events
   it ties with the pretrained GTCRN on readable events (11 vs. 11 discordant
   events) and leaves slightly fewer original words (3.7% vs. 4.5%).

Energy-based numbers on the real events (median drop −0.4 dB for the
fine-tuned model vs. −2.9 dB for DTLN) are misleading on their own. Many
"human" events are clicks, claps or breathing, and the fine-tuned model is
trained to leave everything except speech alone.

**The full pipeline with the better remover (no expert labels):**

```bash
python -m scripts.run_pipeline_real --remover gtcrn_ft --output-dir outputs/pipeline_real_gtcrn_ft \
       --configs hybrid_strict hybrid_balanced hybrid_balanced_dog07 hybrid_high_recall --worker 0 --n-workers 2
python -m scripts.plot_pipeline_real --improved-dir outputs/pipeline_real_gtcrn_ft --oracle-override 0.036
```

| automatic hybrid | readable events, DTLN remover | readable events, fine-tuned GTCRN | dog time muted |
|---|---|---|---|
| strict | 10.3% | 9.8% | 3.6% |
| balanced | 11.0% | 8.4% | 11.3% |
| balanced, privacy-first dog threshold | 8.1% | 5.9% | 19.3% |
| high recall | 9.3% | 7.2% | 24.2% |

For comparison, trained detector + mute gives 7.6% at 9.0% dog loss
(strict) and 3.3% at 32.7% (balanced).

- The better remover moves the automatic hybrid from clearly worse than
  muting to roughly on the same privacy/dog trade-off curve.
- It still does not beat muting. With detector + mute, about 90% of the
  remaining leaks with readable original speech are events the detector
  mostly missed (less than half of the event flagged): 34 of 37 at the strict
  setting, 14 of 16 at balanced.
- So for privacy, detection is now the bottleneck again. Next: the
  intelligible-speech labels (Step 10) and the multi-microphone cues
  (Step 13).

**Step 15: a human detector that listens with all three microphones**

```bash
# on the laptop (needs the raw audio from Step 13; numpy + ffmpeg only, ~3 min)
python -m scripts.extract_spatial_features --export-audio ../data/elte_barks/aligned_mics
# VGGish of every microphone, then the detectors (~20 min on 2 cores)
python -m scripts.extract_multimic_vggish --worker 0 --n-workers 2      # and --worker 1
python -m scripts.train_multimic_detector                               # 15c: what does NOT work
python -m scripts.train_micfusion_detector                              # 15d: what works
python -m scripts.plot_micfusion_detector
# the no-label pipeline (Step 12) with the new detector, at matched dog cost (~1.5 h on 2 cores)
python -m scripts.run_pipeline_real --human-oof outputs/human_detector_micfusion/oof_predictions.npz \
       --extra-configs outputs/configs_micfusion.json --output-dir outputs/pipeline_real_micfusion --worker 0 --n-workers 2
python -m scripts.run_pipeline_real --extra-configs outputs/configs_baseline_curve.json \
       --output-dir outputs/pipeline_real_baseline_curve --worker 0 --n-workers 2
python -m scripts.compare_micfusion_pipeline        # -> outputs/micfusion_pipeline/
```

(`configs_*.json` hold the thresholds from `outputs/human_detector_micfusion/thresholds.json`:
the point where each detector flags 3 / 6 / 10 / 15 / 25% of dog-only frames.)

Step 12 showed that about 90% of the readable speech left after detect-and-mute
comes from human events the detector mostly missed. The raw recordings
(Step 13) have three synchronized microphones per participant, so this step
asks whether the other two microphones help find those events.

*Alignment.* Every play-session file was located inside the raw recording of
its own microphone by cross-correlation: all 30 sessions align at correlation
1.000, at whole-second offsets (e.g. P-02: 141 s and 317 s). VGGish of the
aligned own-microphone audio matches the Step 9 features (median cosine
0.994–0.998), so the other microphones are on exactly the same frames as the
annotations.

*15c: what does not help.*

| detector input (leave-one-participant-out) | ROC AUC |
|---|---|
| session microphone only (Step 9) | 0.797 |
| spatial cues only (band level differences, GCC-PHAT delays) | 0.658 |
| session microphone + spatial cues | 0.796 |
| all three microphones' VGGish in one input vector | 0.727 |

- The spatial cues carry some information (level difference shotgun vs.
  cardioid: AUC 0.77 for human vs. dog), but it does not transfer between
  participants (median per-participant AUC 0.57): the people and dogs stand
  in different places each time. A gradient-boosted model does not help
  either (0.659).
- GCC-PHAT finds a sharp peak at lag 0 in every microphone pair, including
  the wireless clip-on, which has its own transmission delay. That is
  electrical crosstalk inside the recorder, so lags of ±1 sample are now
  ignored.
- Putting all microphones into one long input hurts: the model learns each
  training participant's setup (which microphone is missing, where it stood).

*15d: what helps: one detector, applied to every microphone.* Train ONE
single-microphone detector on the frames of all microphones of the training
participants (3x the data, same labels, because the microphones are
synchronized). Run it on every microphone of the unseen participant, take the
maximum probability, and smooth over 5 frames (±0.48 s).

| human detector | ROC AUC | human time found, at 6% dog-only frames flagged (strict) | ... at 25% (balanced) |
|---|---|---|---|
| Step 9 baseline | 0.797 | 50.0% | 70.0% |
| baseline + smoothing only | 0.812 | 50.9% | 72.0% |
| trained on all mics, session mic only | 0.807 | 52.9% | 72.6% |
| trained on all mics, max over mics | 0.820 | 55.2% | 75.2% |
| **trained on all mics, max over mics + smoothing** | **0.837** | **56.7%** | **78.2%** |

The same idea does **not** improve the dog detector that the hybrid uses to
route sounds (AUC 0.776 vs. 0.770; smoothing alone gives 0.784). So the
microphones help with the quiet, distant human sounds specifically.

*The full no-label pipeline (detect → mute), Whisper privacy check.* Both
detectors were run at the thresholds where they flag the same share of
dog-only frames:

| dog-only frames flagged | one mic: readable events | one mic: dog time muted | three mics: readable events | three mics: dog time muted | events readable only with one mic / only with three | McNemar p |
|---|---|---|---|---|---|---|
| 3% | 9.5% | 4.8% | **7.9%** | 4.2% | 17 / 8 | 0.11 |
| 6% (strict) | 7.6% | 9.0% | **5.9%** | 8.0% | 16 / 6 | 0.053 |
| 10% | 5.3% | 14.3% | **4.0%** | 12.6% | 16 / 8 | 0.15 |
| 15% | 4.0% | 20.7% | 4.1% | 18.4% | 8 / 9 | 1.0 |
| 25% (balanced) | 3.3% | 32.7% | 4.0% | 29.3% | 5 / 9 | 0.42 |

(No processing: 22.9%. Whisper's level on background-only audio: 4.1%.)

Dog time each detector has to mute to reach a given privacy level
(interpolated along each curve; 95% interval from resampling the 17
participants, 2,000 times):

| target: readable events | one microphone | three microphones | dog time saved (95% CI) |
|---|---|---|---|
| 8% | 8.1% | 4.2% | 3 points (0 to 7) |
| **6%** | **12.7%** | **7.8%** | **5 points (0.5 to 12)** |
| 5% | 15.8% | 10.2% | 6 points (−4 to 17) |

Findings:

1. Across the useful range (3–10% of dog-only frames flagged), the
   three-microphone detector is better on both axes at every point: fewer
   readable events AND less dog time muted. For the same privacy it mutes about
   40% less of the dog (6% readable: 7.8% vs. 12.7% of dog time). The
   participant bootstrap puts this gain above zero (98.6% of resamples). Each
   single point on its own is borderline (McNemar p = 0.05–0.15), because only
   about 25 events differ between the two.
2. At 10% of dog-only frames flagged, the pipeline reaches Whisper's
   background level (4.0%) while muting 12.6% of the dog time. The one-mic
   detector needed about 20% for that.
3. Above about 15% the two curves meet at about 4% readable events. The
   remaining ~22 events are real speech that neither detector finds, even at
   high thresholds (e.g. 13 of 18 at 25% are less than half covered). Reaching
   them needs a different signal; the speech labels from Step 10 are the
   obvious next one.
4. Caveat: the thresholds were chosen on the same out-of-fold predictions
   they are evaluated on, exactly as in Steps 9–12, for both detectors alike.
   The models themselves never saw the test participant.

Output: `outputs/human_detector_micfusion/` (comparison, thresholds,
predictions, `recall_vs_dog_cost.png`), `outputs/multimic_detector/` (15c),
`outputs/micfusion_pipeline/` (`comparison.csv`, `summary.json`,
`privacy_vs_dog_loss_micfusion.png`). `run_pipeline_real` now takes
`--human-oof`, `--dog-oof` and `--extra-configs`, and skips the speech
remover for mute-only runs.

**Step 16: a stronger frozen audio model (EfficientAT) for the detectors**

```bash
git clone https://github.com/fschmid56/EfficientAT models/EfficientAT     # MIT licence; weights download themselves
python -m scripts.extract_efficientat --worker 0 --n-workers 2           # and --worker 1 (~3 min in total)
python -m scripts.train_micfusion_detector --features eat_logits          # human detector
python -m scripts.train_micfusion_detector --features eat_logits --target dog
python -m scripts.train_micfusion_detector --features eat_emb             # the 960-d embedding instead
# all three human detectors in ONE pipeline run (same Whisper, same events), 5 thresholds each (~1.5 h on 2 cores)
python -m scripts.run_pipeline_real --extra-configs outputs/configs_v16.json --output-dir outputs/pipeline_real_v16 \
       --worker 0 --n-workers 2                                           # and --worker 1
python -m scripts.compare_detectors_pipeline                              # -> outputs/detectors_pipeline/
```

VGGish (2017) was the weak link. EfficientAT's `mn10_as` (Schmid et al.,
ICASSP 2023) is a 4.9M-parameter MobileNetV3 trained on AudioSet (mAP 0.471),
small enough for a laptop CPU. It also outputs the 527 AudioSet class scores
directly (Speech, Conversation, Laughter, Dog, Bark, ...).

*One technical trap.* The model was trained on 10 s clips. Fed 0.96 s
windows, its zero padding dominates and every output saturates (on our audio
"Door", "Chainsaw" at probability 1.0). So each session runs through the
convolutional part in 10 s chunks (±2.56 s of context, dropped afterwards).
For every 0.96 s window, the feature-map columns it covers are averaged and
passed through the model's own classification head. A test checks that
averaging all columns reproduces the model's clip-level output. With this,
human frames score Speech 0.63 and dog frames score Dog 0.53, as they should.

*Detector level (leave-one-participant-out, same frames and labels as
Steps 9 and 15; human time found at a fixed share of dog-only frames flagged):*

| human detector | ROC AUC | at 6% (strict) | at 25% (balanced) |
|---|---|---|---|
| VGGish, session mic (Step 9) | 0.797 | 50.0% | 70.0% |
| VGGish, three mics + smoothing (Step 15) | 0.837 | 56.7% | 78.2% |
| EfficientAT embedding, three mics + smoothing | 0.888 | 66.8% | 87.5% |
| EfficientAT class scores, session mic + smoothing | 0.896 | 68.4% | 89.0% |
| **EfficientAT class scores, three mics + smoothing** | **0.900** | **66.7%** | **90.6%** |

- With the stronger model, the three-microphone gain mostly disappears
  (0.896 vs 0.900): EfficientAT on one microphone already finds most of what
  VGGish needed three microphones for.
- The 527 class scores work better than the 960-d embedding as input.
- The dog detector (the hybrid's router) also improves: AUC 0.770 → 0.852
  (session mic + smoothing; three mics do not help there).
- At the event level: of the 133 human events where Whisper reads 2+ words
  in the original, the strict setting flags at least half of 81 with VGGish,
  108–112 with EfficientAT.

*Full pipeline (detect → mute, no expert labels), all three detectors in one
run:*

| dog-only frames flagged | VGGish, 1 mic: readable / words / dog lost | EfficientAT, 1 mic | EfficientAT, 3 mics |
|---|---|---|---|
| 3% | 9.5% / 33.0% / 4.8% | **5.2%** / 18.1% / 4.5% | 6.2% / 21.3% / 4.2% |
| 6% (strict) | 7.9% / 22.9% / 9.0% | 4.1% / 12.7% / 8.0% | **3.8%** / 14.0% / 8.0% |
| 10% | 5.0% / 15.0% / 14.3% | 2.9% / 8.6% / 12.6% | **2.6%** / 6.1% / 12.5% |
| 15% | 4.1% / 11.8% / 20.7% | 1.5% / 5.3% / 18.2% | **1.4%** / **2.6%** / 18.1% |
| 25% (balanced) | 3.4% / 9.6% / 32.6% | 1.5% / 5.3% / 28.9% | **1.4%** / **1.9%** / 28.7% |

("readable" = human events where Whisper reads 2+ words, unprocessed 22.9%;
"words" = share of the original words still readable, in events that had 2+
words; "dog lost" = share of dog-only vocalization time muted.)

Every EfficientAT point is significantly better than VGGish at the same
threshold rule (exact McNemar on the same events, p = 0.0002–0.036).

Findings:

1. **Same privacy for a third of the dog loss.** VGGish needs 12.5% of the dog
   time muted to get to 6% readable events. EfficientAT gets there at ≤4.5%
   (its first point is already below 6%). Participant bootstrap of the saving
   at 6%: 7.5 points of dog time (95% CI 3.5–15), better in 100% of 1,000
   resamples.
2. **Below Whisper's background level.** From 6% flagged on, readable events
   drop below 4.1%, the share Whisper "reads" on unprocessed background-only
   audio (muted audio gives Whisper nothing to hallucinate on). At 15%
   flagged (18% of dog time lost), the three-microphone detector leaves 8
   readable events out of 581, and only 4 of them had readable speech
   originally. Only 2.6% of the original words remain readable.
3. **One or three microphones?** With EfficientAT, one microphone is as good
   on "readable events". Three microphones remove more of the words at the
   private end (2.6% vs 5.3% at 15%; 1.9% vs 5.3% at 25%). For a
   single-microphone dataset, the one-microphone version is the one to use.
4. The remaining misses are again events the detector did not cover (3 of
   the 4 real ones at 15% flagged are less than half covered).
5. Caveat (as in Steps 9–15): thresholds are set on the same out-of-fold
   predictions, by the same rule for every detector; the models never saw the
   test participant. This run used a rebuilt environment; the VGGish numbers
   match the earlier runs within ±1–2 events per point (Whisper is sensitive
   to tiny resampling differences).

Output: `outputs/efficientat/` (features, not in git), `outputs/human_detector_micfusion_eat_*`,
`outputs/hybrid_routing_micfusion_eat_logits/`, `outputs/pipeline_real_v16/`,
`outputs/detectors_pipeline/` (`comparison.csv`, `summary.json`,
`privacy_vs_dog_loss_detectors.png`). `run_pipeline_real` now accepts a 4th
config entry (the detector's prediction file), so several detectors can be
compared in one run. It also caches Whisper transcripts of identical windows
across configs.

*Hybrid again, with the better detectors.* The hybrid used EfficientAT
detectors for both human and dog (3 mics for human, 1 mic for dog) and the
fine-tuned GTCRN remover. Where a dog is detected, it subtracts the speech
estimate instead of muting:

```bash
python -m scripts.run_pipeline_real --remover gtcrn_ft --extra-configs outputs/configs_v16_hybrid.json \
       --dog-oof outputs/hybrid_routing_micfusion_eat_logits/dog_oof_single_mic_smooth.npz \
       --output-dir outputs/pipeline_real_v16_hybrid --worker 0 --n-workers 2      # and --worker 1
```

| human threshold (dog-only frames flagged) | dog threshold | readable events | words still readable | dog muted | dog subtracted | same detector, mute only: readable / dog muted |
|---|---|---|---|---|---|---|
| 6% | 0.5 | 8.4% | 18.7% | 3.1% | 4.8% | 3.8% / 8.0% |
| 6% | 0.7 | 5.9% | 15.9% | 4.8% | 3.2% | |
| 10% | 0.5 | 7.6% | 10.5% | 4.7% | 7.8% | 2.6% / 12.5% |
| 10% | 0.7 | 5.3% | 8.3% | 7.6% | 4.9% | |
| 25% | 0.5 | 6.7% | 5.4% | 9.8% | 18.9% | 1.4% / 28.7% |
| 25% | 0.7 | 4.8% | 3.5% | 16.4% | 12.3% | |

The hybrid keeps more dog time than mute at the same human threshold, but
it does not beat mute. Even counting subtracted time as fully kept, 4 of the
6 hybrid points are worse on both axes than some mute point. For example,
mute with the 1-mic EfficientAT detector at 3% gives 5.2% readable events
with 4.5% of the dog muted and nothing subtracted; the hybrid at 6% / 0.7
gives 5.9% with 4.8% muted plus 3.2% subtracted. The other two points are a
trade-off, not a win: one loses less dog (3.1% muted) but leaves 8.4%
readable. Subtracting speech on real far-field audio still leaks words where
human and dog overlap.

**Recommendation from Steps 15–16: EfficientAT detector + mute.** At the
strict setting it leaves 3.8–4.1% readable events (below Whisper's
background level) for 8% of the dog time. The hybrid is only worth it if the
dog sound inside overlaps matters more than privacy there.

**Step 17: what else helps the detector? (larger model, temporal model, true 32 kHz)**

```bash
python -m scripts.extract_efficientat --model mn40_as --output-dir outputs/efficientat_mn40 --worker 0 --n-workers 2   # and 1
python -m scripts.train_micfusion_detector --features eat_logits --eat-dir efficientat_mn40 \
       --output-dir outputs/human_detector_micfusion_eat40_logits
python -m scripts.train_temporal_detector                                  # mn10 scores; add --features efficientat_mn40
python -m scripts.export_aligned_audio                                     # laptop: 32 kHz cuts from the 48 kHz raw files
python -m scripts.extract_efficientat --model mn40_as --audio-32k --output-dir outputs/efficientat_mn40_32k
python -m scripts.train_micfusion_detector --features eat_logits --eat-dir efficientat_mn40_32k \
       --output-dir outputs/human_detector_micfusion_eat40_32k_logits
# pipeline: the new detectors are ADDED to the Step 16 run (same events, same Whisper -> paired comparison)
python -m scripts.run_pipeline_real --extra-configs outputs/configs_v17.json --output-dir outputs/pipeline_real_v16 \
       --worker 0 --n-workers 2                                            # and --worker 1
python -m scripts.compare_detectors_pipeline
```

Three ideas, each tested the same way (leave-one-participant-out, human time
found at a fixed share of dog-only frames flagged):

| human detector | ROC AUC | at 3% | at 6% | at 25% |
|---|---|---|---|---|
| EfficientAT mn10 scores, 1 mic + smoothing (Step 16) | 0.896 | 58.5% | 68.4% | 89.0% |
| EfficientAT mn10 scores, 3 mics + smoothing (Step 16) | 0.900 | 55.0% | 66.7% | 90.6% |
| temporal CNN on mn10 scores, 1 mic | 0.901 | 59.8% | 69.1% | 87.9% |
| temporal CNN on mn10 scores, 3 mics | 0.901 | 57.8% | 69.4% | 87.9% |
| **EfficientAT mn40 scores, 1 mic + smoothing** | **0.912** | **62.5%** | **72.1%** | 90.8% |
| EfficientAT mn40 scores, 3 mics + smoothing | 0.918 | 60.1% | 72.1% | 91.0% |
| temporal CNN on mn40 scores, 3 mics | 0.907 | 61.9% | 70.7% | 88.5% |
| mn40, real 32 kHz audio, 1 mic + smoothing | 0.918 | 63.2% | 72.9% | 91.5% |
| mn40, real 32 kHz audio, 3 mics + smoothing | 0.913 | 59.7% | 70.8% | 90.7% |

1. **The larger EfficientAT (mn40, 4x wider, AudioSet mAP 0.484) helps a
   little:** AUC 0.896 → 0.912 with one microphone. Feature extraction for
   all 86 microphone files still takes only about 10 minutes on 2 CPU cores.
2. **A temporal model does not help.** A 100k-parameter dilated CNN that sees
   ±3.6 s around each frame (fixed design, 3 seeds averaged) does no better
   than logistic regression + 5-frame smoothing (0.901 vs 0.896–0.900 on
   mn10; 0.907 vs 0.918 on mn40). The frame model already sees ~1 s, and
   the smoothing covers the rest.
3. **The real 32 kHz band does not matter.** Cutting 32 kHz excerpts straight
   from the 48 kHz raw recordings (all 86 align at correlation 1.000) gives
   the same detector as upsampled 16 kHz audio (0.918 vs 0.912 one mic,
   0.913 vs 0.918 three mics). The whole pipeline can stay at 16 kHz.

*In the pipeline (detect → mute, same 581 events, all in one run):*

| dog-only frames flagged | VGGish, 1 mic | mn10, 1 mic | **mn40, 1 mic** | mn40, 3 mics |
|---|---|---|---|---|
| 3% | 9.5% / 33.0% / 4.8% | 5.2% / 18.1% / 4.5% | **4.8% / 16.3% / 4.3%** | 6.0% / 21.1% / 4.2% |
| 6% | 7.9% / 22.9% / 9.0% | 4.1% / 12.7% / 8.0% | **3.1% / 10.5% / 7.9%** | 3.6% / 12.5% / 7.9% |
| 10% | 5.0% / 15.0% / 14.3% | 2.9% / 8.6% / 12.6% | **2.1% / 6.3% / 12.5%** | 2.4% / 8.4% / 12.5% |
| 15% | 4.1% / 11.8% / 20.7% | 1.5% / 5.3% / 18.2% | **1.5% / 4.5% / 17.9%** | 2.1% / 6.8% / 18.0% |
| 25% | 3.4% / 9.6% / 32.6% | 1.5% / 5.3% / 28.9% | **1.0% / 3.0% / 28.5%** | 1.5% / 4.4% / 28.6% |

(readable events / original words still readable / dog-only time muted)

- mn40 with one microphone has the fewest readable events at 3, 6, 10 and 25%
  flagged. At 15% it is within one event of the best (mn10 with three mics:
  1.4%, which also leaves the fewest words at 15–25%).
- It is significantly better than VGGish everywhere (McNemar p ≤ 0.006).
  Against mn10 the gain is not significant (p = 0.30–1.0): a consistent small
  improvement, not a new level.
- With the stronger model, the extra microphones no longer help in the
  pipeline either. **The recommended detector therefore needs only the one
  recording microphone**, which also makes it usable on other datasets.
- Where the lost dog time is: at 10% flagged, 36% of the muted dog-only
  frames lie within 1 s of a human sound (VGGish: 20%). So EfficientAT's
  remaining mistakes cluster around the human sounds, mostly the edges of
  the muted segments.

**Step 18: a command that anonymizes a new recording**

```bash
python -m scripts.train_final_detector          # once: mn40 + logistic regression on all 17 participants -> models/human_detector_eat40.npz
python -m scripts.anonymize_recording input.wav anonymized.wav                 # --strictness light | standard | strict
```

It runs the recommended pipeline on any WAV file, with no annotations:
EfficientAT mn40 → the saved logistic regression → 5-frame smoothing →
threshold → mute (±0.12 s). The muting is applied to the original file (its
sample rate, bit depth and all channels), so the dog sound that is kept is
bit-identical to the input. A `.segments.csv` lists what was muted.

| strictness | threshold set on unseen participants at | readable events | words left | dog time lost |
|---|---|---|---|---|
| `light` | 6% of dog-only frames | 3.1% | 10.5% | 7.9% |
| `standard` (default) | 10% | 2.1% | 6.3% | 12.5% |
| `strict` | 15% | 1.5% | 4.5% | 17.9% |

(Numbers from Step 17, measured with models that never saw the participant;
unprocessed: 22.9% readable.)

Checks:

- A model saved with P-02 left out, run through the command on P-02's
  sessions, reproduces the leave-one-participant-out probabilities to within
  0.008. The command and the evaluation compute the same thing.
- A test confirms the output keeps the sample rate, channels and bit depth,
  and that the only samples changed are the muted ones.

The saved model (20 kB) is derived from the annotations, so it stays out of
git (`models/` is ignored) until the dataset question is settled.

**Step 19: how solid are the numbers? (metric fix, honest thresholds, finer cuts)**

```bash
python -m scripts.nested_thresholds --worker 0 --n-workers 2      # and --worker 1 (272 inner fits, ~20 min)
python -m scripts.nested_thresholds --summarize-only               # -> outputs/nested_thresholds/, configs_v19.json
python -m scripts.run_pipeline_real --extra-configs outputs/configs_v19.json --output-dir outputs/pipeline_real_v16 \
       --worker 0 --n-workers 2                                     # and --worker 1
python -m scripts.compare_detectors_pipeline                        # now recounts all words with the fixed rule
python -m scripts.extract_efficientat --model mn40_as --columns-only --output-dir outputs/efficientat_mn40_cols
python -m scripts.eval_column_detector                              # -> outputs/column_detector/
```

*1. A counting bug in the privacy measure, fixed.* When Whisper's output
stops in the middle of a sound caption, for example "(wind blowing" or
"(sounds of a" without the closing bracket, the caption was counted as
words. Looking at every event still readable after the best pipeline showed
this. `normalize_text` now drops such cut-off captions as well (with a
test), and `compare_detectors_pipeline` recounts every stored transcript
with the fixed rule:

| | before the fix | after the fix |
|---|---|---|
| unprocessed: human events with 2+ readable words | 22.9% | 19.8% |
| Whisper on background-only audio ("floor") | 4.1% | **1.0%** |
| VGGish, 1 mic, 6% flagged | 7.9% | 6.0% |
| mn40, 1 mic, 6% / 10% / 15% / 25% flagged | 3.1 / 2.1 / 1.5 / 1.0% | **2.2 / 1.4 / 0.9 / 0.5%** |

So the "floor" from Step 8 was mostly cut-off captions. The ranking of the
methods does not change. With the fixed counting, mn40 + mute is
significantly better than VGGish at every point (McNemar p ≤ 0.012). The dog
time it must mute to reach 3% readable events drops from 21.8% (VGGish) to
5.3%. Participant bootstrap of the saving: 14 points (95% CI 6–26), better in
100% of resamples. Tables in earlier steps keep the counting of their time.

*What is left at the standard setting (mn40, 10% flagged; 12 events
before the fix, 8 after).* I read all of them by hand. Most are Whisper's
typical sub-second hallucinations ("Thank you.", "Mm-hmm.", fragments in
Japanese or Korean on Hungarian audio) in events that were 0.2–0.6 s long and
never flagged. Only about two contain plausible speech, and one of those is
mostly muted (2 of 17 words left, none of them the original words).

*2. Honest thresholds.* Every threshold so far was chosen on the pooled
out-of-fold predictions, which include the test participant's.
`nested_thresholds` repeats the whole selection inside each fold: an inner
leave-one-out over the 16 other participants sets the threshold, and only
then is it applied to the 17th. Result:

- the nested thresholds vary by only ±0.013 across folds and flag the same
  frames as the pooled ones (0.2–0.6% of frames differ);
- the dog cost hits its target exactly overall (3.0 / 6.1 / 10.1 / 15.2 /
  25.1%);
- in the pipeline the nested version gives the same numbers as the pooled
  one, e.g. 6%: 2.2% readable, 9.6% words, 8% dog in both; 10%: 1.4% /
  5.6% / 12.6%.

**The selection caveat of Steps 9–18 is negligible.** What does vary is the
cost per recording: at a 3% target the per-participant dog cost ranges from
0.4% to 10.5% (`per_participant.csv`). The average is reliable, a single
recording less so.

*3. Finer cuts do not help.* A third of the lost dog time lies near human
sounds, so I classified EfficientAT's own 0.32 s feature-map columns instead
of 0.96 s windows and muted exactly the flagged 0.32 s. Scored on a 10 ms
grid against the annotations, the column detector is slightly worse at every
dog cost (human time muted at 6% dog: 70.0% vs 70.3%; at 10%: 76.9% vs
78.7%). Changing the ±0.12 s padding only moves along the same curve. The
limit is the detection itself, not the time resolution.

**Step 20: out of domain - other speakers, other dogs, other rooms**

```bash
python -m scripts.eval_out_of_domain --scene-type dog                     # 40 scenes, dogs only (no Whisper needed)
python -m scripts.eval_out_of_domain --scene-type mixed --worker 0 --n-workers 2   # and --worker 1 (Whisper)
python -m scripts.eval_feature_subsets                                    # 20b: fewer AudioSet classes as input?
python -m scripts.train_augmented_detector --save-final                   # 20c: + other dogs as negatives -> models/human_detector_eat40_aug.npz
python -m scripts.eval_out_of_domain --scene-type mixed --heldout-dogs --worker 0 --n-workers 2   # and --worker 1
```

Everything up to Step 19 is ELTE: same lab, microphones, language and kind
of play, even though every test person was unseen. Here the deployable
command (Step 18), unchanged and ELTE-trained, runs on synthetic 30 s scenes
that differ in all of these. The ground truth comes from the dry sources:

- English read speech (LibriSpeech, 40 speakers);
- dogs from ESC-50 (40 clips, other dogs and microphones, mostly outdoors);
- simulated rooms from 3 x 3 to 9 x 7 m (RT60 0.2–0.8 s, source 0.5–4 m
  away), pink noise, and an overall level from −30 to 0 dB.

Privacy is now scored against the true text: an utterance counts as leaked
if Whisper gets 2+ of its words right.

*20a. What transfers and what does not* (thresholds as set on ELTE):

| scenes | light | standard | strict | Silero VAD |
|---|---|---|---|---|
| speech only: speech time muted | 98% | 99% | 99.7% | — |
| dogs only: dog time muted | **21%** | **34%** | **45%** | 0% |
| (ELTE for comparison: dog-only time muted) | 8% | 12.5% | 18% | — |

The detector finds foreign speech almost completely. But it mutes far too
much of unfamiliar dogs: it learned what the ELTE dogs sound like, and other
dogs partly count as "human".

*20b. A smaller input does not fix it.* With only the 81 human and dog
classes as input, unfamiliar-dog muting at standard goes from 34% to 28%.
With only the 72 human classes it gets worse (41%). Even EfficientAT's own
voice scores, with no training at all, mute 16% (and find much less ELTE
speech: AUC 0.847). The ESC-50 dogs themselves overlap with voice classes
for the model.

*20c. Other dogs as negative examples fix it.* Synthetic dog-only scenes
made from half of the ESC-50 clips (20) are added as "not human" frames to
every training fold. The other 20 clips are used only for testing:

| | ELTE: human time found at 6 / 10 / 15% | unseen ESC-50 dogs: dog time muted (light / standard / strict) | foreign speech time muted |
|---|---|---|---|
| ELTE only | 72.1 / 80.0 / 84.9% | 22 / 33 / 44% | 98 / 99 / 99.7% |
| **ELTE + 20 ESC-50 dogs** | 71.8 / 80.1 / 85.2% | **1.5 / 2.9 / 4.8%** | 95 / 98 / 99% |

On ELTE nothing changes, including the full pipeline with Whisper:
2.4 / 1.4 / 1.0% readable events at 7.9 / 12.4 / 18.0% dog lost, vs.
2.2 / 1.4 / 0.9% at 7.9 / 12.5 / 17.9%, and still p ≤ 0.008 vs. VGGish.

*But where speech and barks overlap, the fix trades privacy for dog sound.*
In mixed scenes (held-out dogs only; most dog-only time falls into pauses
inside utterances, so even muting the exact utterances loses 46% of it):

| method | utterances with 2+ correct words | dog-only time muted |
|---|---|---|
| no processing | 100% | 0% |
| ELTE only, light / standard / strict | 13.3 / 9.2 / 4.2% | 42 / 53 / 59% |
| ELTE + ESC-50 dogs, light / standard / strict | 29.2 / 24.2 / 14.2% | 17 / 23 / 29% |
| Silero VAD (generic speech detector) | 12.5% | 31% |
| mute the exact utterances (oracle) | 0% | 46% |

The augmented detector stops muting speech that is covered by a bark. On
these scenes neither version beats generic Silero by much. On ELTE, Silero
finds almost nothing (21.5% readable, Step 12), because the ELTE speech is
quiet and distant.

What this means in practice:

1. On ELTE-like recordings use the default model; that is where it was
   measured.
2. On a new dataset, check a few annotated minutes first. If too much dog
   sound is lost, use `--model models/human_detector_eat40_aug.npz` together
   with a stricter level, or recalibrate the thresholds on that data. The
   per-recording variation (Step 19) already says a single recording can
   differ from the average.
3. Training on more varied dogs costs nothing on ELTE, so a future version
   should include them (AudioSet / FSD50K dog clips, ideally real
   recordings from other labs).

Caveat: the scenes are synthetic. The held-out dogs are unseen clips but
come from the same ESC-50 source and the same scene generator as the
training dogs.

**Step 21: speech under a bark is still speech - a privacy-first general model**

```bash
python -m scripts.train_overlap_augmented --save-final     # -> models/human_detector_eat40_aug2.npz (~15 min)
python -m scripts.eval_out_of_domain --scene-type mixed --heldout-dogs --worker 0 --n-workers 2   # and --worker 1 (~75 min)
python -m scripts.run_pipeline_real --extra-configs outputs/configs_v21.json --output-dir outputs/pipeline_real_v16 \
       --worker 0 --n-workers 2                             # and --worker 1
python -m scripts.compare_detectors_pipeline
```

Step 20c's detector stopped muting speech that a bark covers: it had seen
other dogs only as "not human", never a dog on top of a person. This step
adds that case. Synthetic scenes with speech AND dogs go into training, and
every frame is labelled by the true speech activity: speech for at least
half of the frame means "human", even under a bark. Everything that is
tested is held out:

- speakers are split by person: 20 LibriSpeech speakers for training, 20
  others for testing;
- dog clips are split 20 / 20, as before;
- the test scenes use other random seeds.

*Detector level* (10 ms grid, thresholds set on ELTE; light / standard / strict):

| | ELTE human time found | unseen dogs alone: dog time muted | mixed: speech muted | mixed: speech under a bark muted | mixed: dog-only time muted |
|---|---|---|---|---|---|
| ELTE only | 72 / 80 / 85% | 22 / 33 / 44% | 88 / 91 / 93% | 69 / 75 / 80% | 33 / 44 / 53% |
| + other dogs (20c) | 72 / 80 / 85% | 1.5 / 2.9 / 4.8% | 73 / 79 / 83% | 40 / 51 / 58% | 13 / 19 / 26% |
| **+ other dogs + speech under barks** | 70 / 77 / 83% | 9 / 15 / 21% | **99 / 99 / 99%** | **96 / 97 / 98%** | 58 / 68 / 75% |

*With Whisper on the held-out mixed scenes* (40 scenes x 3 utterances, unseen speakers and dogs):

| method | utterances with 2+ correct words | original words recovered | dog-only time muted |
|---|---|---|---|
| no processing | 99.2% | 66.2% | 0% |
| ELTE only, light / standard / strict | 11.7 / 7.5 / 6.7% | 2.2 / 1.2 / 1.2% | 39 / 47 / 56% |
| + other dogs (20c) | 30.8 / 20.0 / 15.0% | 8.2 / 4.3 / 3.0% | 14 / 19 / 26% |
| **+ other dogs + speech under barks (21a)** | **0 / 0 / 0%** | **0 / 0 / 0%** | 63 / 72 / 77% |
| Silero VAD | 11.7% | 2.9% | 33% |
| mute the exact utterances (oracle) | 0% | 0% | 47% |

*On ELTE* (full pipeline, same events, corrected counting): readable
events / words left / dog lost:

| level | ELTE-only model | 21a model |
|---|---|---|
| light | 2.2% / 9.6% / 7.9% | 2.2% / 9.6% / 7.8% |
| standard | 1.4% / 5.6% / 12.5% | 2.1% / 9.1% / 12.4% |
| strict | 0.9% / 4.4% / 17.9% | 1.4% / 6.1% / 17.8% |

Findings:

1. **On unseen speakers, unseen dogs and simulated rooms, the 21a model
   leaves not a single utterance readable** (0 of 120 at every level),
   like muting the exact utterances. It mutes 16–30 points more dog-only
   time than that oracle (63–77% vs 47%): in these scenes most dog-only time
   falls in the short pauses inside utterances, and the 5-frame smoothing
   mutes through them.
2. On unfamiliar dogs alone it mutes 9–21% of their time: better than ELTE
   only (22–44%), worse than 20c (1.5–4.8%).
3. On ELTE it costs a little privacy at the standard and strict levels
   (2.1% vs 1.4% readable at standard; light is identical). The model
   spends some of its capacity on the synthetic data.
4. So the three models are three points on the same trade-off, and none
   wins everywhere:

| model file | use it for | trade-off |
|---|---|---|
| `models/human_detector_eat40.npz` (default) | ELTE-like recordings | best measured privacy on ELTE |
| `models/human_detector_eat40_aug2.npz` | unknown recordings, privacy first | no readable speech in the out-of-domain test, more dog lost |
| `models/human_detector_eat40_aug.npz` | unknown recordings, dog sound first | keeps unfamiliar dogs, leaks speech under barks |

*Speed:* `anonymize_recording` processes a 10.2 min session in 32 s on 2 CPU
cores, including model loading (about 19x real time). An hour of audio takes
about 3 minutes.

Caveat (as in Step 20): the out-of-domain scenes are synthetic. They are
held out by speaker, dog clip and seed, but made by the same generator as
the training scenes. Real recordings from another lab would be the real test.

**Step 22: re-checking the evaluation from the lab's side**

```bash
python -m scripts.eval_dog_events                                   # 22a: dog calls by type, as events (seconds, no Whisper)
python -m scripts.audit_full_sessions --worker 0 --n-workers 2     # 22b: Whisper over whole recordings (~40 min), and --worker 1
python -m scripts.audit_full_sessions --summarize-only
python -m scripts.download_models --asr-turbo                       # Whisper large-v3-turbo (~1 GB) for 22c
python -m scripts.attack_stronger_asr --worker 0 --n-workers 2     # 22c: Whisper large-v3-turbo as the attacker, and --worker 1
python -m scripts.attack_stronger_asr --summarize-only
```

Steps 9–21 measured two things: readable words in the annotated human
events, and dog-only TIME muted. This step asks whether these measure what
the lab cares about.

*22a. Dog calls by type, counted as events.* 89% of the annotated dog time
is panting (3,521 of 3,966 s), so "dog time lost" is mostly a panting
number. The lab analyses calls as events, and a growl with its middle muted
is a damaged growl. For each call type, counting events with no human sound
on them: intact (not one sample muted) / lost (half or more muted):

| call type (events) | VGGish, 6% | mn40, light (6%) | mn40, standard (10%) | mn40, strict (15%) |
|---|---|---|---|---|
| Pant (1,116) | 77 / 6% | 79 / 10% | 71 / 15% | 62 / 22% |
| Other (243) | 86 / 9% | 83 / 13% | 75 / 21% | 67 / 24% |
| Growl (188) | 79 / 12% | **92 / 5%** | 90 / 7% | 84 / 11% |
| Grunt (159) | 77 / 18% | 79 / 19% | 66 / 28% | 60 / 36% |
| Bark/yelp (48) | 83 / 6% | 88 / 6% | 75 / 19% | 62 / 31% |
| Moan (32) | 56 / 34% | 72 / 25% | 62 / 28% | 53 / 38% |
| Whine (7) | 43 / 57% | 14 / 86% | 14 / 86% | 14 / 86% |

- The rest of each row is "damaged" (partly muted). Calls that overlap a human
  sound (178 pants, 10 growls, ...) are mostly lost at every setting. That is
  the price of muting, not a detector error.
- Growls are well protected. Whines are the one call type the detector
  takes for a human voice: 6 of 7 are lost (only 7 events, but a clear
  pattern).
- Which calls matter most decides which setting is right. That is a question
  for the lab (see `outputs/dog_events/by_call_type.csv` for every setting).

*22b. Whole recordings, not only the annotated events.* Every privacy number
so far read only the windows around the 581 annotated human events. Speech
the annotators did not mark would never have been noticed. Whisper therefore
read every 8 s window that has no annotated human sound within 0.5 s:
519 windows, 68 minutes, about half of all audio.

- 10 windows contain 2+ "words" in the ORIGINAL. Eight are typical Whisper
  hallucinations over dog sounds: "2.5kg" or "2.5L" (five times), "ТРЕВОЖНАЯ
  МУЗЫКА" ("anxious music") twice, and "2. 2. 2.".
- Two are short Korean phrases on Hungarian audio ("hey, careful"; a repeated
  "I..."). They could be misheard real speech. The standard setting mutes the
  first, and the second keeps two syllables. Both are worth one listen in the
  listening audit.
- **No substantial unannotated speech was found.** The expert annotations
  cover the speech in these recordings, so the event-based privacy numbers
  of Steps 8–21 are not missing a significant source of leaks.
- After anonymization, 6 (light, standard) or 4 (strict) of these windows
  still produce text, mostly the same hallucinations.

*22c. A stronger attacker.* All privacy numbers so far used Whisper small
(244M parameters) to try to read words back. Here Whisper large-v3-turbo
(809M parameters, int8 sherpa-onnx export, `models/sherpa-onnx-whisper-turbo`)
read the same 581 event windows, before and after muting:

| version | turbo: 2+ words | turbo: words of the original read back | small: 2+ words |
|---|---|---|---|
| original | 41.0% | 100% | 19.8% |
| background, no human (floor) | 24.5% | – | 1.0% |
| VGGish, 6% | 20.7% | 35.0% | 6.0% |
| mn40, light (6%) | 10.3% | 20.8% | 2.2% |
| mn40, standard (10%) | 7.1% | 13.1% | 1.4% |
| mn40 human-vs-dog (Step 23), standard | 6.9% | 11.7% | 1.2% |

- Turbo writes text on a quarter of the windows with NO human in them
  ("*crickets*" aside, it invents phrases). So "2+ words" from turbo is
  mostly invented text, and its rates are well above the floor of Whisper
  small for every version, including the original.
- The ranking does not change: mn40 at 6% leaves half as many readable
  events as VGGish at 6% (McNemar 80 vs 20 discordant events, p < 0.0001).
- A stricter test counts only **confirmed speech**: events where both
  recognizers read 2+ words in the original and agree on at least half of
  them. Only 34 of the 581 events pass. After muting, an event still leaks
  if either recognizer reads back 2+ of the words it read in the original:

| version | confirmed speech events still leaking (of 34) |
|---|---|
| VGGish, 6% | 6 (18%) |
| mn40, light (6%) | 2 (6%) |
| mn40, standard (10%) | 1 (3%): "she's like…" |
| mn40 human-vs-dog, standard | 0 |

- The difference VGGish vs mn40 at 6% on these 34 events is 6 vs 2
  (p = 0.29): too few events for a test, but the same direction as above.
- **A bigger recognizer does not change the conclusion.** It reads more
  "words" everywhere, but most of them are invented. On speech both
  recognizers agree on, at most one short phrase survives the standard
  setting. (`outputs/attack_turbo/summary.csv`, `confirmed_speech.csv`)

**Step 23: train for the trade-off we actually make (and what did not help)**

```bash
python -m scripts.train_objective_detector         # -> outputs/human_detector_objective/, configs_v23.json
python -m scripts.run_pipeline_real --extra-configs outputs/configs_v23.json \
       --configs obj40_mute_c06 obj40_mute_c10 obj40_mute_c15 --output-dir outputs/pipeline_real_v16 --worker 0 --n-workers 2
python -m scripts.compare_detectors_pipeline
python -m scripts.eval_dog_events
```

Muting background (no human, no dog) costs the lab nothing. Yet the
detector was trained to separate human frames from every other frame,
including room noise. Training it on human vs dog-only frames instead (same
inputs, same model, same threshold rule, no tuning) aims it at the actual
trade-off:

| | human time found at 6 / 10 / 15% of dog-only frames flagged | readable events (Whisper small) at 6 / 10 / 15% | background flagged at 10% |
|---|---|---|---|
| human vs everything (Steps 17–22) | 72.0 / 79.8 / 85.0% | 2.2 / 1.4 / 0.9% | 17% |
| **human vs dog only** | **74.3 / 82.5 / 86.7%** | **1.9 / 1.2 / 0.7%** | 29% |

- The detector finds 2–3 points more human time at every dog cost. Readable
  events drop slightly at every level, but the event-level difference is
  not significant (McNemar p ≥ 0.6, 1–4 discordant events).
- It mutes more background. That costs no dog sound but makes the files
  quieter.
- By call type, it protects the vocal calls better: at standard, grunts
  intact 66% → 77%, barks 75% → 81%, moans 62% → 69%. Pants do slightly worse
  (71% → 68%), because pants are almost all of the dog TIME that the threshold
  holds fixed.

So the choice between the two is about which dog sounds matter. That is a
question for the lab.

Three other ideas were tested at the frame level and did **not** help:

| idea | result (human time found at 6% dog cost) |
|---|---|
| train only on events with readable speech (Whisper labels) instead of all human sounds | worse: 65.0% (speech frames found: 87.3% vs 91.1%) |
| normalize the class scores per recording (remove each session's own average) | worse: AUC 0.866–0.898 vs 0.912 |
| subtract a separate dog detector's score | +3 points, but it needs a tuned weight; the objective change above gets the same gain without one |

Training on all human sounds works better than training on speech alone,
even for finding speech: the non-speech human sounds share the acoustics
and add training examples.

**Step 24: an anonymized copy of the dataset**

```bash
python -m scripts.make_release_segments                 # mute segments for every session and level (+ report.csv)
python -m scripts.apply_mute_segments --level standard  # numpy + ffmpeg only -> ../data/elte_barks/anonymized_v1/standard/
```

The point of the project is recordings the lab can use. The deployable
detector (trained on all 17 participants, Step 18) marked what to mute in
all 30 sessions at each level. The muting was then applied to the ORIGINAL
files: 48 kHz, their own sample format (16-bit, or 32-bit for P-01) and
channels (stereo for P-05 and P-06).

- Every file is checked: every sample outside the muted intervals is
  bit-identical to the original, and every muted sample is zero (30 of 30
  pass). `MANIFEST.csv` lists the SHA-256 of each original and anonymized file.

| level | audio muted (of 125 min) | annotated human events mostly muted | dog calls intact (no human on them): pant / growl / grunt / bark / moan / whine |
|---|---|---|---|
| light | 18.5% | 450 / 581 | 918/1116 · 179/188 · 135/159 · 46/48 · 27/32 · 2/7 |
| standard | 24.1% | 495 / 581 | 811/1116 · 176/188 · 118/159 · 45/48 · 24/32 · 1/7 |
| strict | 29.9% | 520 / 581 | 729/1116 · 167/188 · 107/159 · 42/48 · 21/32 · 1/7 |

- The share muted per session ranges from 3% to 75% (median 21%) at
  standard. Some sessions are mostly talk.
- The model has seen these participants, so these counts are better than
  for a new recording. The leave-one-participant-out numbers (Steps 17–22)
  are the estimate for new data.
- The anonymized files stay on the lab's computers (`data/`, not in git).
  Like the trained model, they can only be shared once the dataset
  question is settled.

**Step 25: listen like the attacker - Whisper's encoder as the detector's ears (new default)**

```bash
python -m scripts.download_models --asr --asr-turbo --asr-parakeet   # Whisper small (+ turbo, Parakeet for the attacks)
python -m scripts.extract_whisper_encoder                    # 25a: Whisper small encoder, every 20 ms, all sessions (~7 min)
python -m scripts.train_whisper_detector                     # 25b: inputs x objectives, leave one participant out (~25 min)
python -m scripts.train_fine_detector --objective all        # 25c: 20 ms decisions (negative result, ~30 min)
python -m scripts.run_pipeline_real --extra-configs outputs/configs_v25.json \
       --configs wenc_eat_dog_mute_c03 wenc_eat_dog_mute_c06 wenc_eat_dog_mute_c10 wenc_eat_dog_mute_c15 \
       --output-dir outputs/pipeline_real_v16                # Whisper small as attacker (~40 min)
python -m scripts.compare_detectors_pipeline                 # recounted table
python -m scripts.bootstrap_step25                           # participant bootstrap: old vs new settings
python -m scripts.attack_stronger_asr --model parakeet --configs wenc_eat_dog_mute_c03 wenc_eat_dog_mute_c06 wenc_eat_dog_mute_c10 \
       --worker 0 --n-workers 2                              # and --worker 1; then --summarize-only (~10 min)
python -m scripts.attack_stronger_asr --model turbo --configs wenc_eat_dog_mute_c03 wenc_eat_dog_mute_c06 wenc_eat_dog_mute_c10 \
       --worker 0 --n-workers 2                              # and --worker 1 (~1 h)
python -m scripts.eval_dog_events                            # dog calls by type
python -m scripts.plot_step25                                # outputs/detectors_pipeline/step25_three_attackers.png
python -m scripts.train_final_detector                       # 25d: models/human_detector_v2.npz (the new default)
python -m scripts.eval_out_of_domain --heldout-dogs          # adds the v2 model to the synthetic scenes of Steps 20-21
python -m scripts.make_release_segments                      # 25e: outputs/release_v2/
python -m scripts.apply_mute_segments --level standard       # ../data/elte_barks/anonymized_v2/standard/
```

The privacy attacker is a speech recognizer, but the detector listened through
AudioSet models, trained to name 527 kinds of sound, not to understand speech.
Whisper's encoder was trained on 680,000 hours of speech and describes every
20 ms of audio with 768 numbers. Step 25 gives the detector those numbers.

*25a-b. What the detector hears.* Same recipe as Steps 17 and 23 (logistic
regression, 5-frame smoothing, leave one participant out); only the input
changes. Share of human frames found at 3 / 6 / 10 / 15% of the dog-only frames
flagged:

| input | trained vs | 3% | 6% | 10% | 15% | AUC (human vs dog) |
|---|---|---|---|---|---|---|
| EfficientAT mn40 (Steps 17-24) | everything | 62.3 | 72.1 | 80.0 | 84.9 | 0.927 |
| EfficientAT mn40 (Step 23) | dog only | 65.6 | 74.4 | 82.5 | 86.7 | 0.938 |
| Whisper encoder | everything | 70.2 | 78.6 | 85.4 | 90.9 | 0.948 |
| Whisper encoder | dog only | 72.9 | 80.7 | 87.5 | 92.2 | 0.954 |
| Whisper + mn40 | everything | 71.2 | 81.2 | 86.6 | 90.8 | 0.951 |
| **Whisper + mn40** | **dog only** | **75.9** | **84.0** | **89.4** | **93.2** | **0.963** |

- Whisper alone beats AudioSet at every dog cost. The two together are best.
- Training against dog sounds only (Step 23) adds 1-5 points for every input.
- At 3% dog cost, the new detector finds more human sound than the old one at 6%.

*End to end* (Whisper small reads the 581 event windows, counted as in Step 19):

| setting | readable events | words still readable | dog-only time lost |
|---|---|---|---|
| no processing | 19.8% | 100% | 0% |
| background only (floor) | 1.0% | 0% | - |
| old light (mn40, 6%) | 2.2% | 9.6% | 7.9% |
| old standard (mn40, 10%) | 1.4% | 5.6% | 12.5% |
| old strict (mn40, 15%) | 0.9% | 4.4% | 17.9% |
| **new light (3%)** | 1.7% | 7.8% | **4.3%** |
| **new standard (6%)** | **1.0%** | 5.2% | **8.0%** |
| **new strict (10%)** | 0.7% | 3.5% | 12.3% |
| new, 15% | 0.5% | 2.6% | 17.8% |

- The new standard setting reaches Whisper's background floor (1.0%) and
  loses 8.0% of the dog-only sound, against 12.5% for the old standard.
- Participant bootstrap (`outputs/detectors_pipeline/bootstrap_step25.json`):
  the dog loss drops by 36% at standard (95% CI 21-51%), 46% at light
  (28-61%) and 31% at strict (16-45%). The privacy differences are not
  significant (the CIs include zero: readable events within +-1.6 points,
  words still readable within +-6 points).
- "6% of dog-only frames" is an average over all dogs: per dog it ranges
  from 0% to 19%.

*Is that only because the detector and the attacker are both Whisper?* Two more
attackers read the same windows: Whisper large-v3-turbo (809M) and NVIDIA
**Parakeet TDT 0.6B v3**, a different model family (FastConformer transducer,
trained on other data, 25 European languages including Hungarian). Since
"2+ words" counts invented text (turbo writes text on 24.5% of background
windows, Parakeet on 4.9%), the fair measure is the share of each attacker's
own original words it can still read back:

| setting | dog lost | Whisper small | turbo | Parakeet |
|---|---|---|---|---|
| old standard | 12.5% | 5.6% | 13.1% | 11.3% |
| new standard | 8.0% | 5.2% | 14.9% | 9.2% |
| new strict | 12.3% | 3.5% | 9.9% | 6.7% |

- For all three attackers the new curve lies below the old one
  (`outputs/detectors_pipeline/step25_three_attackers.png`): at the same dog
  loss, fewer words are left. At the new standard, turbo reads back slightly
  more than at the old standard (14.9 vs 13.1%) while 36% less dog is lost;
  "new strict" matches the old dog loss and leaves fewer words for everyone.
- On the 29 events where Parakeet and Whisper small agree on the original
  words (clear speech), 1 is still readable after the new standard, 3 after
  the old one.

*Dog calls by type* (no human sound on them; intact = not one sample muted):

| call (events) | old standard | new standard |
|---|---|---|
| Pant (1,116) | 71% | **82%** |
| Other (243) | 75% | **83%** |
| Growl (188) | 90% | **96%** |
| Grunt (159) | 66% | **84%** |
| Bark/yelp (48) | **75%** | 67% |
| Moan (32) | 62% | **75%** |
| Whine (7) | 14% | **29%** |

- More of most call types survives. Barks are the main exception: the speech
  encoder seems to hear the harmonic bark as voice-like. Sneezes also lose
  (83% vs 92% intact, 24 events). For bark-focused studies the old
  model is still there (`--model models/human_detector_eat40.npz`).
- The dataset is about play panting ("Dog play pant"), and 82% of the pants
  now stay untouched.

*25c. What did not help.* About 40% of the dog sound lost lies within 1 s of
a human sound, so finer boundaries looked promising. A classifier on every
20 ms Whisper frame (context +-0.06 / 0.5 / 1.5 s, trained against all other
frames) muted 68% of the human time at 3.6% dog loss, against 73% at 4.5% for
the 0.24 s detector with the same input and objective, and
interpolating the 0.24 s probabilities to 20 ms with tighter padding changed
nothing. Adding neighbouring frames or the spread of the Whisper frames to the
input did not help either (83.6% and 79.0% at 6%, vs 84.0%).

*25d. The new default model.* `train_final_detector` (now `--recipe v2` by
default) trains on all 17 participants and saves `models/human_detector_v2.npz`
(19 kB); `anonymize_recording` uses it by default. It needs the Whisper small
encoder (`download_models --asr`) next to EfficientAT. Check of the deployed
code path: a model trained without P-03 and P-12, run through
`anonymize_recording`'s code on their four sessions, gives probabilities that
correlate 0.99 with the leave-one-participant-out ones, and the same recall
and dog cost at each level.

*25e. Anonymized copy v2* (in sample; the model saw these participants):

| level | audio muted | human events mostly muted | pants intact | growls intact | grunts intact |
|---|---|---|---|---|---|
| light | 16.7% | 458 / 581 | 1029 / 1116 | 186 / 188 | 151 / 159 |
| standard | 22.2% | 518 / 581 | 952 / 1116 | 182 / 188 | 141 / 159 |
| strict | 28.4% | 548 / 581 | 855 / 1116 | 178 / 188 | 133 / 159 |

Compared with release v1 (Step 24) at standard: less audio muted (22.2 vs
24.1%), more human events covered (518 vs 495), more pants intact (952 vs 811).

*Out of domain* (the synthetic scenes of Steps 20-21 with unseen speakers and
unseen dogs; the v2 model unchanged, thresholds set on ELTE; Whisper small
scored against the LibriSpeech reference text):

| method | utterances with 2+ correct words | dog-only time muted |
|---|---|---|
| no processing | 99.2% | 0% |
| mute the exact utterances (oracle) | 0% | 47.3% |
| Silero VAD | 11.7% | 32.8% |
| old model (mn40), light / standard | 11.7% / 7.5% | 38.9% / 47.3% |
| Step 21 model (+ other dogs and speech under barks), light | 0% | 63.3% |
| **v2, light / standard** | **1.7% / 0%** | 41.1% / 56.6% |

- Without any extra training data, v2 reaches the privacy of the oracle on
  unseen speakers, unseen dogs and simulated rooms. The Step 21 model needed
  synthetic training data and still loses more dog sound (63% vs 57%).
- Foreign barks cost more dog sound here than ELTE's dogs do (as above, the
  speech encoder treats barks as voice-like).
- So one model now covers both cases. The `_aug` / `_aug2` models of Step 21
  are no longer needed for new recordings.

*Speed:* `anonymize_recording` with v2 runs at about 8x real time on 2 CPU
cores (235 s of audio in 28 s, model loading included), against about 19x for
the EfficientAT-only model: an hour of audio takes about 8 minutes.

**Step 26: every dog call type counts the same (barks fixed; new default `human_detector_v3.npz`)**

```bash
python -m scripts.train_calltype_detector --alpha 0.5 1.0      # leave one participant out, ~10 min each
python -m scripts.run_pipeline_real --extra-configs outputs/configs_v26.json \
       --configs barkw10_mute_c03 barkw10_mute_c06 barkw10_mute_c10 barkw10_mute_c15 --output-dir outputs/pipeline_real_v16
python -m scripts.compare_detectors_pipeline
python -m scripts.eval_dog_events
python -m scripts.attack_stronger_asr --model parakeet --configs barkw10_mute_c03 barkw10_mute_c06 barkw10_mute_c10
python -m scripts.train_final_detector                          # now --recipe v3: models/human_detector_v3.npz
python -m scripts.make_release_segments                         # outputs/release_v3/
python -m scripts.apply_mute_segments --level standard          # ../data/elte_barks/anonymized_v3/standard/
```

The Step 25 detector left fewer barks untouched than the old one (67% vs 75%).
All 16 barks it muted came from one dog (P-03, 21 barks); the old model had kept 9 of them. Only three
dogs bark in this dataset (P-03: 21, P-01: 18, P-13: 7), so when P-03 is left
out the model has seen 25 barks (27 s) against 3,521 s of panting. It learns
what a pant is and treats an unfamiliar bark as "not a pant", i.e. maybe a voice.

The fix changes only the training weights: each dog-only frame is weighted
by (all dog-only frames / frames of its call type)^alpha, and the dog-only
frames keep their total weight. With alpha = 1 every call type weighs the same
in total. Input, objective, thresholds rule and evaluation are unchanged.

Dog calls left untouched (no human sound on them), at the standard setting:

| call (events) | old (mn40, 10%) | Step 25 (6%) | **Step 26, alpha = 1 (6%)** |
|---|---|---|---|
| Pant (1,116) | 71% | 82% | **82%** |
| Other (243) | 75% | 83% | **87%** |
| Growl (188) | 90% | 96% | **98%** |
| Grunt (159) | 66% | 84% | **92%** |
| Bark/yelp (48) | 75% | 67% | **100%** |
| Moan (32) | 62% | 75% | **100%** |
| Sneeze (24) | 92% | 83% | 88% |
| Whine (7) | 14% | 29% | **71%** |

Privacy does not change (Whisper small; words = original words still readable):

| setting | readable events | words | dog-only time lost | Parakeet: words read back |
|---|---|---|---|---|
| Step 25, light / standard / strict | 1.7 / 1.0 / 0.7% | 7.8 / 5.2 / 3.5% | 4.3 / 8.0 / 12.3% | 13.8 / 9.2 / 6.7% |
| **Step 26, light / standard / strict** | 1.7 / 1.0 / 1.0% | 6.2 / 5.2 / 4.3% | 4.4 / 7.9 / 12.3% | **10.4 / 7.9 / 5.0%** |

- The share of human frames found at 6% dog cost drops slightly (84.0% ->
  81.8%), but the readable events stay the same and Parakeet reads back fewer
  words at every level.
- alpha = 0.5 lies in between (barks 88%, grunts 89%, recall 83.4%).
- So the v3 model keeps as many or more calls of every type than both earlier
  models, except sneezes, where the old model keeps one more (22 vs 21 of 24),
  at the same privacy. It is the new default of `anonymize_recording`.

*Anonymized copy v3* (in sample; the model saw these participants):

| level | audio muted | human events mostly muted | pants | growls | grunts | barks | moans | whines intact |
|---|---|---|---|---|---|---|---|---|
| light | 17.0% | 444 / 581 | 1017 / 1116 | 188 / 188 | 158 / 159 | 48 / 48 | 32 / 32 | 6 / 7 |
| standard | 22.6% | 514 / 581 | 910 / 1116 | 186 / 188 | 153 / 159 | 48 / 48 | 32 / 32 | 6 / 7 |
| strict | 28.8% | 546 / 581 | 816 / 1116 | 185 / 188 | 146 / 159 | 47 / 48 | 32 / 32 | 6 / 7 |

In sample, v3 keeps fewer pants than v2 (910 vs 952 at standard) and more of
everything else; on unseen participants (above) both keep 82% of the pants.

**Step 27: keeping the dog when a person talks over it (`--keep-dog`)**

```bash
# 27d: the dog extractor, conditioned on Whisper's encoder (two participant folds + the deployable model)
python -m scripts.train_dog_extractor_w --build --group odd       # 3,000 mixtures of real lab audio (~30 min each)
python -m scripts.train_dog_extractor_w --build --group even
python -m scripts.train_dog_extractor_w --train-group odd         # model for the even participants (~70 min)
python -m scripts.train_dog_extractor_w --train-group even
python -m scripts.train_dog_extractor_w --train-group all --steps 5000   # deployable: models/dog_extractor/dog_extractor_w_all.pt
python -m scripts.eval_extractor_synthetic                        # held-out mixtures, where the true dog is known
# 27e: the dog router
python -m scripts.train_dog_router --final                        # one microphone; + models/dog_router_v1.npz
python -m scripts.train_dog_router --inputs wenc+eat+sp           # with the microphone cues
# 27b: real sessions, two attackers (and turbo on the deployable versions)
python -m scripts.eval_dog_extraction --config barkw10_mute_c06 --asr small parakeet --mask-power 3 \
       --router outputs/human_detector_whisper/oof_dog_router_wenc_eat_w5.npz --router-name routed1mic --route-share 0.02 0.05
python -m scripts.eval_dog_extraction --config barkw10_mute_c06 --asr small parakeet --mask-power 3 \
       --router outputs/human_detector_whisper/oof_dog_router_wenc_eat_sp_w5.npz --route-share 0.02 0.05 0.1
python -m scripts.make_listening_pack --n 12                       # outputs/listening_overlap/listen.html
```

**The problem.** 227 s of dog sound lies under a human sound: 223 of the 2,055
annotated calls (178 of them pants). The standard setting mutes 184 s of it,
and with the detector's false alarms 12.1% of all dog sound; about 40% of that
is dog under a person.

**What did not work (27a-c).** A mask network that predicts which
time-frequency bins belong to the dog, trained on mixtures of the lab's own
audio (real dog windows + real human-only events, participant folds):

| model | speech left on held-out mixtures |
|---|---|
| spectrogram only, privacy-weighted signal loss | -19 dB, but the dog was turned down by 9 dB too |
| spectrogram only, mask-regression loss | -5.5 dB (stopped after 1,000 steps: not improving) |
| + level difference between the clip-on M1 (on the dog, see below) and the room mic | -4 to -5 dB (stopped after 1,000 steps) |
| **+ Whisper encoder frames (27d)** | **-6 to -12 dB, dog turned down by only 0.6-3.7 dB** |

With only ~150 real human events per participant group, a network learns the
training voices, not speech in general. Whisper's encoder brings speech
knowledge from 680,000 hours. (The clip-on M1 is on the dog: pants are 8 dB
relatively louder on it, human sounds 4.5 dB, in 25 of 30 sessions.)

On held-out mixtures, where the true dog is known, the Whisper-conditioned
extractor keeps the dog with a median SI-SDR of 14-15 dB (9 dB with the more
conservative mask power 3) and removes 6-12 dB of the speech
(`outputs/dog_extraction/synthetic_heldout.json`).

**The key finding (27b): extract only where there is a dog.** On the real
sessions (each participant processed by an extractor trained on the other
participant group), inside the time the v3 detector mutes at standard:

| inside the muted time | Whisper small: readable / words | Parakeet: readable / words |
|---|---|---|
| silence (Step 26) | 0.9% / 4.4% | 3.3% / 7.9% |
| extractor output everywhere | 2.8% / 5.8% | 6.7% / 16.9% |
| extractor output only inside annotated dog calls (oracle) | 1.2% / 4.4% | 4.1% / 12.5% |

Everywhere, the speech the extractor misses becomes readable again (on the real
recordings the extractor removes only 4-9 dB of person-only sound). Inside
dog calls the dog covers what is left. So the extractor needs a gate.

**The router (27e).** A detector of dog calls that works while a person is
talking: same features as the human detector (Whisper encoder + mn40), trained
on frames with vs without a dog call, frames with a person weighted 5x, leave
one participant out. Inside human frames it tells dog+person from person-only
with AUC 0.78 (0.81 with the microphone cues). Its threshold is set by the
share of person-only frames it lets through.

**Result** (mask power 3; "dog kept" = share of the muted dog time that comes back):

| version | Whisper small: readable / words | Parakeet: readable / words | dog under a person kept | dog false alarms kept | all dog time muted |
|---|---|---|---|---|---|
| Step 25-26, mute | 0.9% / 4.4% | 3.3% / 7.9% | 0% | 0% | 12.1% |
| **`--keep-dog safe`** (1 mic, 2%) | **0.9% / 4.4%** | **2.9% / 7.9%** | 18% | 19% | **9.8%** |
| **`--keep-dog more`** (1 mic, 5%) | 1.0% / 4.4% | 3.4% / 10.4% | 32% | 31% | **8.2%** |
| 3 mics, 2% | 0.9% / 4.4% | 3.1% / 7.9% | 19% | 17% | 9.9% |
| 3 mics, 5% | 1.2% / 4.4% | 3.8% / 10.4% | 38% | 34% | 7.8% |
| 3 mics, 10% | 1.4% / 4.4% | 4.3% / 11.7% | 52% | 49% | 6.0% |

- `safe` brings back a fifth of the muted dog sound with no measurable change
  in privacy: Whisper small reads exactly the same events and words; Parakeet
  reads the same share of words (4 events become readable, 6 stop being
  readable); Whisper turbo reads back 17.9% instead of 17.0% of its words
  (9 vs 8 readable events changing; the gained words are "you", "so", "oh").
  On the 35 events where the attackers agree on clear speech, 2 leak instead of 1.
- `more` brings back a third; Whisper small reads the same words, Parakeet
  10.4% instead of 7.9% (extra words: laughter "ha ha ha"), turbo 18.0%; turbo
  reads one short phrase in full that muting removes. Use `safe` when in doubt.
- These are small numbers (5-54 readable events per attacker): "no measurable
  change" is not proof of equal privacy. That is what the listening check is for.
- Compared with the Step 17 default (16.4% of all dog sound muted), `more`
  halves the dog sound lost (8.2%).
- The kept dog is the extractor's output (16 kHz, about 1-4 dB quieter than
  the original call, speech removed), not the untouched recording.
- Mask power 3 (more conservative than 1) was chosen after power 1 had been
  scored on the same events (power 1, router 2%: Parakeet 10.6% of words), so
  the chosen setting is slightly optimistic.
- The router's threshold is an average: per participant, `safe` lets through
  0-15% of the person-only frames.
- The two extractor folds differ: on held-out mixtures, the remaining speech
  while someone talks is 20-22 dB below the dog in one fold, only 2-4 dB in the
  other.

**Not done yet: a person listening.** `outputs/listening_overlap/listen.html`
has 12 real calls under speech (original / muted / dog kept, `more`). Until a
person confirms that no words are understandable in "dog kept", `--keep-dog`
is off by default.

**Use it:**

```bash
python -m scripts.anonymize_recording input.wav out.wav --keep-dog safe
```

It needs `models/dog_router_v1.npz` and `models/dog_extractor/dog_extractor_w_all.pt`
(trained on all participants, 5,000 steps) next to the v3 detector. The
deployed extractor itself was not evaluated on unseen people (it has seen them
all); the numbers above are from the two fold models. On stereo files the
extractor works on the channel mix and its output goes to every channel.

*Note on baselines:* Step 27's privacy numbers come from their own transcription
run, so the "mute" row (0.9% / 4.4%) differs slightly from Step 26's table
(1.0% / 5.2%) for the same setting; comparisons within each table use the same run.

**Step 28: nothing muted - the dog comes back in all the muted time (`--keep-dog all`)**

```bash
# 28a-b: a router that also listens to the extractor's output (did not help)
python -m scripts.extracted_features                               # extractor output -> Whisper + mn40 features (~20 min)
python -m scripts.train_dog_router2 --inputs x+y
# 28c: many more voices for the extractor (synthetic speech) and a clean-speech test set
python -m scripts.download_models --tts                            # Piper / VITS voices (sherpa-onnx releases on GitHub)
python -m scripts.make_tts_speech --n 1500                         # 1,500 utterances per speaker half (~25 min each)
python -m scripts.train_dog_extractor_w --build --group odd --mix tts --n 3000
python -m scripts.train_dog_extractor_w --build --group even --mix tts --n 3000
python -m scripts.train_dog_extractor_w --build --group odd --mix clean --n 300     # test only: unseen clean voices
python -m scripts.train_dog_extractor_w --build --group even --mix clean --n 300
python -m scripts.train_dog_extractor_w --train-group odd --data ",_tts" --tag _tts --steps 6000
python -m scripts.train_dog_extractor_w --train-group odd --data ",_tts" --tag _w2tts --arch w2 --steps 6000
python -m scripts.eval_extractor_synthetic --data _clean --n 300 --tag _tts --powers 1 2 3 5
python -m scripts.eval_extractor_real --tag _tts                   # real human-only and dog-only events
# 28d-f: room tone under the processed time; four attackers
python -m scripts.room_tone_levels                                 # how loud the dog calls are above the room tone
E="--config barkw10_mute_c06 --kind w --mask-power 3 --room-tone 10 20 --adaptive-tone -6 -12 \
   --router outputs/human_detector_whisper/oof_dog_router_wenc_eat_w5.npz --router-name routed1mic --route-share 0.05"
python -m scripts.eval_dog_extraction $E --asr small parakeet        # Step 27's transcripts are reused (same folder)
python -m scripts.eval_dog_extraction $E --asr turbo --only original mute mute_rt10 extract_rt10 extract_art-12 routed1mic0.05_hyb10
python -m scripts.eval_dog_extraction $E --asr dfn+parakeet --only original mute routed1mic0.05 extract_rt10 extract_art-12 routed1mic0.05_hyb10
python -m scripts.compare_leaks outputs/dog_extraction/barkw10_mute_c06_w_p3   # what each version lets through
```

**Where Step 27 stopped.** `--keep-dog safe` brings back a fifth of the muted
dog sound; 9.8% of all dog sound is still muted. 62% of the muted dog time is
not under a person at all: it is pure dog that the detector flagged (295 s),
the rest is dog under a person (184 s). The router that decides "is there a
dog?" is stuck at AUC 0.78-0.80, so the way forward is to make the extractor's
output safe everywhere.

**28a-b: a better router? No.** Describing the extractor's output with the
same two networks and adding it to the router's input: AUC inside human frames
0.80 (output only), 0.74-0.79 (combined, logistic regression or gradient
boosting). No real gain.

**28c: better extractors? Only along the same trade-off.**

- *Synthetic voices.* The extractor had heard ~150 real voices per group. Text
  to speech gives 780 more (Hungarian Piper voices, LibriTTS-R and VCTK
  speakers, split into two halves by speaker so the test never hears a training
  voice), saying dog-play phrases and LibriSpeech sentences at 0.8-1.3x speed,
  placed in simulated lab rooms at the level of real lab voices.
- *A clean test set.* The old held-out score (6-12 dB of speech removed) was
  measured on real human events, which also carry the room's background; the
  network keeps that background, as it should. On mixtures whose human part is
  only unseen clean voices, the Step 27 extractor removes **14 dB** of the speech
  (mask power 1) and **20 dB** (power 3, as deployed).
- Results on those clean mixtures (speech removed / dog turned down, mask power 1 and 3; each group
  processed by a model trained on the other group):

| extractor | even participants | odd participants |
|---|---|---|
| Step 27 | 13.5 / 0.6 dB; 19.9 / 1.4 dB | 14.5 / 1.4 dB; 20.5 / 2.6 dB |
| + synthetic voices | 14.7 / 0.4 dB; 20.8 / 1.0 dB | 14.6 / 0.8 dB; 20.7 / 1.5 dB |
| + a loss on the leaked speech in dB (fine-tuned) | 21.9 / 2.3 dB; 26.4 / 3.7 dB | not trained |
| frequency decoder (U-Net style) + synthetic voices | 14.7 / 0.2 dB; 19.0 / 0.4 dB | 14.4 / 0.9 dB; 19.0 / 1.2 dB |

  The leak loss removes more speech only by turning the dog down more: at equal
  dog loss it is no better than a higher mask power. The frequency decoder
  keeps the dog better at the same speech removal on clean voices, but on the
  real recordings it removes less of the person-only sound (4.1 dB instead of
  5.6 dB, `real_events*.json`); it would need its own attack evaluation. Step
  27's extractor stays.
- The ceiling is far away: the target mask itself would remove 31 dB.

**28d: why the extractor's leftovers are readable.** The attackers normalize
the level before reading. What matters is not how much quieter the leftover
voice is than the original, but how far it is above everything else in the
output. In the lab the room tone is quiet: person-only speech is about 15 dB
above it, the extractor's output during person-only sound only 4-6 dB, so the
faint leftover is still the loudest thing there.

**28e: room tone.** Inside the processed time, add steady noise with the
spectrum of the recording's own room tone (the quietest 20% of the frames
outside the processed time), a fixed number of dB above it
(`vad_anonymization/comfort_noise.py`). Muted gaps then sound like the room
instead of a dropout, and the leftover voice is under a noise floor.

On the real sessions (detector `standard`, extractor of the other participant
group, mask power 3), words of its own original transcript that each attacker
reads back (`dfn+parakeet`: an adaptive attacker that first cleans the audio
with DeepFilterNet3, a speech enhancer that removes steady noise):

| inside the processed time | Whisper small | Parakeet | DeepFilterNet + Parakeet | Whisper turbo | events with real words beyond muting* |
|---|---|---|---|---|---|
| silence (`standard`) | 4.4% | 7.9% | 10.0% | 17.0% | - |
| room tone +10 dB only | 4.4% | 10.0% | | 16.5% | 0 / 0 / - / 1 |
| extractor everywhere | 5.8% | 16.9% | | | 2 / 1 / - / - |
| extractor everywhere + room tone +10 dB | 4.4% | 11.3% | 15.6% | 17.6% | 0 / 0 / 0 / 4 |
| extractor everywhere + room tone +20 dB | 4.4% | 10.0% | | | 0 / 0 / - / - |
| extractor + room tone following the removed power (-12 dB) | 4.4% | 13.1% | 16.3% | 22.5% | 0 / 0 / 2 / 9 |
| `--keep-dog more` (Step 27) | 4.4% | 10.4% | 10.3% | 18.0% | 0 / 0 / 0 / 4 |
| **`--keep-dog all`**: extractor everywhere, room tone +10 dB where the router hears no dog | **4.4%** | 13.8% | 16.1% | 18.4% | **0 / 0 / 0** / 6 |

\* per attacker (small / Parakeet / DeepFilterNet + Parakeet / turbo): human
events where the attacker reads back more of its own words than from the muted
version, not counting laughter, backchannels ("mm-hmm") and the recognizers'
typical noise words ("you", "thank you"); `scripts/compare_leaks.py` lists them.

- What the extra events contain, with room tone +10 dB and in `--keep-dog all`:
  laughter and "mm-hmm" (the extractor lets laughter through) and "thank you";
  no new words for Whisper small, Parakeet or the DeepFilterNet attacker. Room
  tone alone already makes the recognizers write a few such things.
- Whisper turbo writes English filler ("I'm going to go") for Hungarian speech;
  its extra matches are such phrases and single words ("so", "happy",
  "¿qué?"), plus the "Kiss me, kiss" that `--keep-dog more` already lets through.
- In numbers (95% bootstrap intervals over participants, `leaks_vs_mute.json`),
  `--keep-dog all` against muting: Parakeet +5.8 points (1.9 to 11.8),
  DeepFilterNet + Parakeet +6.1 (1.1 to 10.7) - both above zero, and all of it
  laughter, "mm-hmm" and "thank you"; turbo +1.4 (-0.4 to 3.3); Whisper small 0.
  With 80-238 readable reference events per attacker, "no new words" is not
  proof that nothing leaks.
- The room tone level that follows the removed power keeps the dog more audible
  (half of the calls 6 dB above the noise) but Whisper turbo reads clearly more
  (+5.5 points, interval 2.4-9.0): not used.

**28f: what comes back for the dog.** With `--keep-dog all` no dog time is
muted any more (`standard`: 12.1% of all dog sound; `safe`: 9.8%; `more`:
8.2%). The price is the noise floor. How loud the 432 annotated calls inside the
processed time are compared with the added noise:

| version | median call vs added noise | calls 6+ dB above it | calls with no noise at all |
|---|---|---|---|
| extractor everywhere + room tone +10 dB | -1.2 dB | 21% | 0% |
| **`--keep-dog all`** | **+1.4 dB** | **34%** | **11%** |
| extractor + room tone +20 dB | -11.2 dB | 3% | 0% |

- Per call type with `--keep-dog all` (calls that get some room tone): pants
  -0.4 dB under a person, -0.5 dB in detector false alarms (median, against
  the added noise); other calls -2.8 dB and +1.8 dB.
- Pants are quiet: a pant is 13 dB above the room tone (median); barks 61 dB,
  growls 26 dB, grunts 28 dB, moans 39 dB (`room_tone_levels.json`). Loud calls
  stay well above the room tone; pants under a person end up near it.
- There is no frequency band where a quieter masker would spare the pants:
  pants, person-only speech and the extractor's leftovers are spread evenly
  over 0-8 kHz.
- Compared with `--keep-dog more`, which brings back a third of the muted dog
  time without noise, `--keep-dog all` brings back all of it, with a noise
  floor where the router hears no dog. Which is more useful depends on the
  analysis: for detecting calls and for loud calls, `all`; for the acoustics of
  quiet pants, the pants that `more` keeps clean are kept clean by `all` too.

**Use it:**

```bash
python -m scripts.anonymize_recording input.wav out.wav --keep-dog all             # room tone +10 dB (auto)
python -m scripts.anonymize_recording input.wav out.wav --keep-dog all --room-tone 20
python -m scripts.anonymize_recording input.wav out.wav --room-tone 10            # muting, with room tone in the gaps
```

Audio outside the processed time stays bit-identical in every mode (checked).
The listening pack (`outputs/listening_overlap/listen.html`) now has a fourth
version per clip (`d`: `--keep-dog all`). `--keep-dog` stays off by default
until a person has listened.

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
