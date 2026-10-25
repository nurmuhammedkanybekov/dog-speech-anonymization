"""Step 28c: many more voices for the dog extractor - synthetic speech.

The dog extractor learns what to remove from the human sounds it is trained
on. The lab has ~150 real human-only events per participant group, plus 72
LibriSpeech utterances: too few voices, so it partly learns those voices
instead of speech (Steps 27a-d). Real speech corpora (LibriSpeech in full,
Common Voice, VoxPopuli) cannot be downloaded from this environment, but
neural text-to-speech voices can (sherpa-onnx releases on GitHub):

* Hungarian: three Piper voices (anna, berta, imre) - the participants' language;
* English: Piper LibriTTS-R, 904 speakers; VITS VCTK, 109 speakers.

Texts are short phrases of the kind people say while playing with a dog, and
general sentences, in Hungarian and English; speed varies 0.8-1.3x. Speakers
are split into two halves (odd / even speaker number for the multi-speaker
voices, the Hungarian voices alternate), one per participant group, so the
held-out evaluation never hears a training voice.

Writes ``outputs/tts_speech/<group>.npz`` (dry 16 kHz clips; the room and the
level are added when mixtures are made).

Usage (from inside speech_anonymization/):
    python -m scripts.make_tts_speech --n 1500        # per group, ~25 min each
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

SR = 16_000
MODELS = Path(__file__).resolve().parent.parent / "models" / "tts"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "tts_speech"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

HU = """Gyere ide! | Jó kutya! | Ügyes vagy! | Hol a labda? | Hozd ide! | Fogd meg! | Ereszd el! | Nem, nem, nem! |
Na gyere, gyere! | Ki a jó kutya? | Szép volt! | Várj! | Ül! | Fekszik! | Pacsi! | Keresd meg! | Hozd vissza! |
Mit csinálsz? | Gyere, játsszunk! | Most én jövök! | Vigyázz! | Ne harapd meg! | Jaj, de aranyos vagy! |
Na, mi lesz? | Nézd csak! | Itt van, itt van! | Gyere, kapd el! | Szaladj! | Állj meg! | Hát ez nagyon jó volt. |
Most figyelj rám. | Megvan a kötél? | Húzd erősebben! | Elég lesz, pihenjünk egy kicsit. | Ügyes kislány! |
Ügyes fiú! | Szeretlek, te kis bolond. | Gyere, megsimogatlak. | Mutasd a pocakod! | Hol van a játék? |
Ez a tiéd? | Mindjárt dobom! | Egy, kettő, három! | Na, ki nyert? | Nem baj, majd legközelebb. |
Holnap megint jövünk. | Ma nagyon fáradt vagy, ugye? | A kamera most is felvesz? | Kezdhetjük a következőt? |
Szerintem ez jól ment. | Mennyi idő van még? | Kérsz vizet? | Csak nyugodtan. | Mindjárt végzünk. |
Köszönöm szépen. | Hol van a gazdi? | Gyere vissza hozzám! | Nem szabad! | Jól van, jól van. |
Megint elszaladt. | Olyan gyors vagy! | Csak egyszer még! | Ez nagyon vicces volt. | Ki az ügyes?""".replace("\n", " ")
EN = """Come here! | Good dog! | Where is the ball? | Bring it back! | Drop it! | Get it! | Sit! | Stay! | Who's a good boy? |
Let's play! | Ready? Go! | Wait for it. | Look at you! | Are you tired? | One more time. | That was great. |
Can you hear me? | Is the camera still recording? | How much time do we have? | Let's take a short break. |
I think she likes it. | He is so fast today. | Do you want some water? | Okay, let's start again. |
No, no, not the shoes! | Catch it! | Over here! | Gentle, gentle. | Thank you so much. | That's enough for today.""".replace("\n", " ")


def phrases(text: str) -> list[str]:
    return [p.strip() for p in text.split("|") if p.strip()]


def libri_sentences() -> list[str]:
    f = PROJECT_ROOT / "data" / "librispeech_subset" / "transcripts.txt"
    out = []
    for line in open(f):
        words = line.strip().split(" ", 1)[-1].lower().split()
        for a in range(0, max(1, len(words) - 6), 8):
            out.append(" ".join(words[a : a + random.randint(4, 12)]).capitalize() + ".")
    return out


def tts(name: str, model: str, data_dir: bool = True, lexicon: bool = False):
    import sherpa_onnx

    d = MODELS / name
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(model=str(d / model), tokens=str(d / "tokens.txt"),
                                                 data_dir=str(d / "espeak-ng-data") if data_dir else "",
                                                 lexicon=str(d / "lexicon.txt") if lexicon else "")
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=2)))


def main() -> None:
    import torch
    import torchaudio

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1500, help="utterances per participant group")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    hu, en, libri = phrases(HU), phrases(EN), libri_sentences()
    hu_voices = [("vits-piper-hu_HU-anna-medium", "hu_HU-anna-medium.onnx"),
                 ("vits-piper-hu_HU-berta-medium", "hu_HU-berta-medium.onnx"),
                 ("vits-piper-hu_HU-imre-medium", "hu_HU-imre-medium.onnx")]
    engines = {"hu": [tts(n, m) for n, m in hu_voices],
               "libritts": tts("vits-piper-en_US-libritts_r-medium", "en_US-libritts_r-medium.onnx"),
               "vctk": tts("vits-vctk", "vits-vctk.int8.onnx", data_dir=False, lexicon=True)}
    for gi, group in enumerate(("odd", "even")):
        rng = random.Random(100 + gi)
        clips, meta, t0 = [], [], time.time()
        for k in range(args.n):
            r = rng.random()
            if r < 0.4:                                  # Hungarian, voices alternate between the groups
                vi = [v for v in range(3) if v % 2 == gi] or [gi]
                v = rng.choice(vi)
                text = " ".join(rng.sample(hu, rng.randint(1, 3)))
                eng, sid, tag = engines["hu"][v], 0, f"hu{v}"
            elif r < 0.8:
                sid = rng.randrange(gi, 904, 2)
                text = rng.choice(libri) if rng.random() < 0.6 else " ".join(rng.sample(en, rng.randint(1, 3)))
                eng, tag = engines["libritts"], f"libritts{sid}"
            else:
                sid = rng.randrange(gi, 109, 2)
                text = rng.choice(libri) if rng.random() < 0.6 else " ".join(rng.sample(en, rng.randint(1, 3)))
                eng, tag = engines["vctk"], f"vctk{sid}"
            a = eng.generate(text, sid=sid, speed=rng.uniform(0.8, 1.3))
            x = torch.tensor(a.samples, dtype=torch.float32)
            x = torchaudio.functional.resample(x, a.sample_rate, SR).numpy()
            if len(x) > SR // 4:
                clips.append((x[: 4 * SR] / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float16))
                meta.append(tag)
            if k % 250 == 0:
                print(f"{group}: {k}/{args.n} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(OUT / f"{group}.npz", **{f"c{i}": c for i, c in enumerate(clips)}, voices=np.array(meta))
        print(f"saved {OUT / group}.npz: {len(clips)} clips, {len(set(meta))} voices", flush=True)


if __name__ == "__main__":
    main()
