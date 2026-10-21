# Speech Anonymization for Multi-Microphone Dog Recordings

**Task 1 of 3** in the dog-sound-recording analysis project (ELTE Modeling
Laboratory, supervised by Dr. Péter Kovács): detect and suppress human
speech in the lab's recordings so they can be anonymized before further
use. 
The other two tasks (Sound-source localization and Echo cancellation) both of them are out of scope for this sub-repository.

## Current approach and results (21 October 2026)

The goal is to remove human speech from the recordings while keeping the
dog vocalizations intact. The current recommended pipeline needs no
annotations at run time:

1. **Detect human sounds** with a frozen audio model, EfficientAT `mn40_as`
   (AudioSet), whose 527 class scores feed a logistic regression trained on the
   lab's expert annotations. The scores are smoothed over 5 frames of 0.24 s
   (Steps 16–17).
2. **Mute** the flagged time (±0.12 s) in the original file. Subtracting a
   speech estimate where a dog overlaps (the hybrid, with a fine-tuned GTCRN)
   was never better than muting at equal dog loss (Steps 12–16).
3. **Check privacy independently**: Whisper acts as the attacker on every
   annotated human event (Steps 7–8, metric fixed in Step 19).

Results on the ELTE BARKS play sessions (581 human events, 17 participants,
every model tested on participants it never saw). The thresholds can also be
chosen without seeing the test participant (nested, Step 19), with the same
results:

| setting (`--strictness`) | human events with 2+ readable words | original words still readable | dog-only time lost |
|---|---|---|---|
| no processing | 19.8% | 100% | 0% |
| Step 9 detector (VGGish), 6% of dog frames flagged | 6.0% | 19.9% | 9.0% |
| `light` | 2.2% | 9.6% | 7.9% |
| `standard` (default) | 1.4% | 5.6% | 12.5% |
| `strict` | 0.9% | 4.4% | 17.9% |

Whisper "reads" 2+ words in 1.0% of background-only windows of the same
length.

**Anonymize a recording** (after the setup below and `git clone https://github.com/fschmid56/EfficientAT models/EfficientAT`):

```bash
python -m scripts.anonymize_recording input.wav anonymized.wav --strictness standard
```

The command runs at about 19x real time on 2 CPU cores and keeps the sample
rate, channels and bit depth of the input. It needs the trained model
`models/human_detector_eat40.npz`, made by `python -m scripts.train_final_detector`.
That model is kept out of git until the dataset question is settled. For
recordings unlike ELTE's, see Step 21: `--model models/human_detector_eat40_aug2.npz`
puts privacy first, `_aug` keeps more dog sound.

Every step, with its commands and full results, is below (Steps 1–21). The
early steps describe the approaches that were tried and replaced: Silero VAD,
VGGish, and hybrid removal with DTLN.

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
├── reports/                    # progress reports (5 Oct, 21 Oct 2026)
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
