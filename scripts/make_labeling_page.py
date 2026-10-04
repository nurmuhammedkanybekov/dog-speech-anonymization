"""Build a local listening page to tag every annotated human event by tier.

The ELTE annotations mark all human sounds with one label ("Human"). To know
which ones are actual speech (tier A), voiced non-speech (tier B) or other
human noises (tier C) - see docs/target_definition.md - someone has to
listen. This script cuts every human event (+0.5 s context on each side),
makes it loud enough to hear (the recordings are very quiet), and writes a
self-contained HTML page that plays them one by one with keyboard shortcuts
and exports the tags as CSV.

Everything stays on your computer (these are real people's voices) - open
outputs/labeling/label_human_events.html in a browser.

With ``--asr-results`` (outputs/asr_real/per_event.csv from run_asr_real) the
page is ordered so the work takes ~30 min instead of ~1 h:

1. priority  - events where Whisper read 2+ words or longer than 1.5 s (label all)
2. spot-check - 30 random short events Whisper found nothing in (checks the
                assumption that these are not speech)
3. leak audit - the audio AFTER oracle hybrid removal, for every event where
                Whisper still read something: real words, or Whisper inventing?
4. optional  - the remaining short events

Usage (from inside speech_anonymization/):
    python -m scripts.make_labeling_page
    python -m scripts.make_labeling_page --asr-results outputs/asr_real/per_event.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch

from vad_anonymization.audio_io import load_audio, save_audio
from vad_anonymization.elte_barks import load_sessions

SAMPLE_RATE = 16_000
CONTEXT_S = 0.5
MAX_CLIP_S = 12.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Label human events</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; --card:#ffffff; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#ffffff; --muted:#c3c2b7; --line:#3a3a37; --accent:#3987e5; --card:#242422; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.muted { color: var(--muted); }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; margin:16px 0; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
.bar > div { height:100%; background:var(--accent); width:0; }
.meta { display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--muted); }
.timeline { position:relative; height:28px; background:var(--line); border-radius:6px; margin:12px 0; }
.timeline .event { position:absolute; top:0; bottom:0; background:var(--accent); opacity:.35; border-radius:6px; }
.timeline .head { position:absolute; top:0; bottom:0; width:2px; background:var(--ink); }
.buttons { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:8px; margin-top:12px; }
button { font:inherit; padding:10px 12px; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--ink); cursor:pointer; text-align:left; }
button:hover { border-color: var(--accent); }
button.chosen { outline: 2px solid var(--accent); }
kbd { font: 12px ui-monospace, monospace; border:1px solid var(--line); border-radius:4px; padding:0 5px; margin-right:6px; }
.row { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
textarea { width:100%; font:inherit; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:8px; }
</style></head>
<body><main>
<h1>Label the human sounds</h1>
<p class="muted">Each clip is one annotated human sound (blue band) with 0.5 s around it, turned up so it's audible.
Pick what the <b>human</b> part is. Keys: <kbd>1</kbd>-<kbd>5</kbd> label + next, <kbd>Space</kbd>/<kbd>R</kbd> replay,
<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> back/forward, <kbd>W</kbd> show what Whisper heard (try to decide before looking). Progress is saved in this browser; press <b>Export CSV</b> when done.</p>
<div class="bar"><div id="prog"></div></div>
<div class="card">
  <div class="meta"><b id="phase"></b><span id="pos"></span><span id="sess"></span><span id="dur"></span><span id="done"></span></div>
  <div class="timeline" id="tl"><div class="event" id="ev"></div><div class="head" id="head"></div></div>
  <p class="muted" id="help" style="margin:4px 0"></p>
  <p id="hint" style="display:none;margin:4px 0;font-size:13px"></p>
  <audio id="player" controls preload="auto" style="width:100%"></audio>
  <div class="buttons" id="btns"></div>
  <div class="row"><textarea id="note" rows="1" placeholder="optional note (what you heard)"></textarea></div>
  <div class="row"><button id="prev">&larr; Back</button><button id="next">Skip &rarr;</button><button id="nextOpen">Next unlabeled</button><button id="export">Export CSV</button></div>
</div>
<p class="muted" id="summary"></p>
</main>
<script>
const EVENTS = __EVENTS__;
const LABEL_SETS = {
  label: [
    ["A_speech", "Speech: words / name / command (\\"hey\\", \\"good boy\\")"],
    ["B_voiced", "Voiced, no words: laugh, \\"ah\\", \\"oh\\", hum"],
    ["C_nonvocal", "Not voice: kiss/click, whistle, cough, clap"],
    ["D_not_human", "Can't hear a human at all / only dog or noise"],
    ["E_unsure", "Unsure"],
  ],
  audit: [
    ["R_real_leak", "I can understand words in this (privacy leak)"],
    ["V_voice_no_words", "A voice is there, but I can't make out words"],
    ["N_no_voice", "No human voice: Whisper made it up"],
    ["U_unsure", "Unsure"],
  ],
};
const PHASES = { priority: "1/4 Label", spotcheck: "2/4 Spot-check", audit: "3/4 Leak audit (after removal)", optional: "4/4 Optional" };
const AUDIT_HELP = "This is the audio AFTER the human sound was removed (turned up loud). Question: can a person still understand words?";
const KEY = "elte_human_labels_v1";
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} };
let i = Math.max(0, EVENTS.findIndex(e => !store[e.id]));
let showHint = false;
const player = document.getElementById("player");
const btns = document.getElementById("btns");
let LABELS = [];
function drawButtons(kind) {
  LABELS = LABEL_SETS[kind || "label"]; btns.innerHTML = "";
  LABELS.forEach(([code, text], k) => {
    const b = document.createElement("button"); b.dataset.code = code;
    b.innerHTML = `<kbd>${k + 1}</kbd>${text}`; b.onclick = () => choose(code); btns.appendChild(b);
  });
}
function render() {
  const e = EVENTS[i];
  drawButtons(e.kind);
  document.getElementById("phase").textContent = e.phase ? PHASES[e.phase] : "";
  document.getElementById("help").textContent = e.kind === "audit" ? AUDIT_HELP : "";
  const hint = document.getElementById("hint");
  hint.textContent = e.whisper ? `Whisper heard: "${e.whisper}"` : "Whisper heard: (nothing)";
  hint.style.display = (showHint || e.kind === "audit") ? "block" : "none";
  player.src = e.clip; player.play().catch(() => {});
  document.getElementById("pos").textContent = `${i + 1} / ${EVENTS.length}`;
  document.getElementById("sess").textContent = `${e.session} @ ${e.start.toFixed(2)} s`;
  document.getElementById("dur").textContent = `human part ${e.dur.toFixed(2)} s`;
  const ev = document.getElementById("ev");
  ev.style.left = (100 * e.ev0 / e.clip_s) + "%"; ev.style.width = (100 * (e.ev1 - e.ev0) / e.clip_s) + "%";
  const cur = store[e.id];
  document.getElementById("note").value = cur ? (cur.note || "") : "";
  [...btns.children].forEach(b => b.classList.toggle("chosen", cur && cur.label === b.dataset.code));
  const n = Object.keys(store).length;
  document.getElementById("done").textContent = `${n} labeled`;
  document.getElementById("prog").style.width = (100 * n / EVENTS.length) + "%";
  const counts = {}; Object.values(store).forEach(v => counts[v.label] = (counts[v.label] || 0) + 1);
  const all = [...LABEL_SETS.label, ...LABEL_SETS.audit];
  const left = EVENTS.filter(x => x.phase !== "optional" && !store[x.id]).length;
  document.getElementById("summary").textContent = (EVENTS[0].phase ? `${left} left before the optional part · ` : "") +
    all.filter(([c]) => counts[c]).map(([c]) => `${c}: ${counts[c]}`).join("  ·  ");
}
function choose(code) {
  const e = EVENTS[i]; store[e.id] = { label: code, note: document.getElementById("note").value }; save();
  if (i < EVENTS.length - 1) i++; render();
}
player.ontimeupdate = () => {
  const e = EVENTS[i]; document.getElementById("head").style.left = (100 * player.currentTime / e.clip_s) + "%";
};
document.getElementById("prev").onclick = () => { if (i > 0) { i--; render(); } };
document.getElementById("next").onclick = () => { if (i < EVENTS.length - 1) { i++; render(); } };
document.getElementById("nextOpen").onclick = () => { const k = EVENTS.findIndex(e => !store[e.id]); if (k >= 0) { i = k; render(); } };
document.getElementById("note").addEventListener("keydown", ev => ev.stopPropagation());
document.addEventListener("keydown", ev => {
  if (ev.key >= "1" && ev.key <= String(LABELS.length)) { choose(LABELS[+ev.key - 1][0]); ev.preventDefault(); }
  else if (ev.key === " " || ev.key === "r" || ev.key === "R") { player.currentTime = 0; player.play(); ev.preventDefault(); }
  else if (ev.key === "w" || ev.key === "W") { showHint = !showHint; render(); }
  else if (ev.key === "ArrowLeft") { if (i > 0) { i--; render(); } }
  else if (ev.key === "ArrowRight") { if (i < EVENTS.length - 1) { i++; render(); } }
});
document.getElementById("export").onclick = () => {
  const rows = [["id", "kind", "phase", "session", "start_s", "end_s", "label", "note"]];
  EVENTS.forEach(e => { const v = store[e.id]; rows.push([e.id, e.kind || "label", e.phase || "", e.session, e.start, e.end, v ? v.label : "", v ? (v.note || "").replace(/[\\n,]/g, " ") : ""]); });
  const blob = new Blob([rows.map(r => r.join(",")).join("\\n")], { type: "text/csv" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "human_event_labels.csv"; a.click();
};
render();
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "elte_barks")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "labeling")
    p.add_argument("--asr-results", type=Path, default=None, help="per_event.csv from run_asr_real: order the page by priority")
    p.add_argument("--enhancer", default="dtln", help="speech estimate used for the oracle hybrid in the leak audit")
    p.add_argument("--n-spotcheck", type=int, default=30)
    args = p.parse_args()
    clip_dir = args.output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    asr = {r["id"]: r for r in csv.DictReader(open(args.asr_results, encoding="utf-8"))} if args.asr_results else {}
    audit_ids = {i for i, r in asr.items() if int(r["words_hybrid"]) >= 1}
    if audit_ids:
        from scripts.run_oracle_real import PAD_S, speech_estimate
        from vad_anonymization.anonymize import hybrid_remove
        from vad_anonymization.detection import SpeechSegment
        from vad_anonymization.enhancers import ENHANCERS

        enhancer = ENHANCERS[args.enhancer]()
        (args.output_dir / "audit_clips").mkdir(parents=True, exist_ok=True)

    events, audits = [], []
    for sess in load_sessions(args.data_root):
        audio = load_audio(sess.audio_path, SAMPLE_RATE)
        total_s = audio.shape[-1] / SAMPLE_RATE
        hybrid = None
        if any(f"{sess.name}_{k:03d}" in audit_ids for k in range(len(sess.human))):
            segs = [SpeechSegment(max(0.0, e.start_s - PAD_S), min(total_s, e.end_s + PAD_S)) for e in sess.human]
            dogs = [SpeechSegment(d.start_s, d.end_s) for d in sess.dog]
            hybrid = hybrid_remove(audio, speech_estimate(audio, segs, enhancer, False), segs, dogs)
        for k, e in enumerate(sess.human):
            t0 = max(0.0, e.start_s - CONTEXT_S)
            t1 = min(total_s, e.end_s + CONTEXT_S, t0 + MAX_CLIP_S)
            clip = audio[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)].clone()
            ev0, ev1 = e.start_s - t0, min(e.end_s, t1) - t0
            human_part = clip[int(ev0 * SAMPLE_RATE) : int(ev1 * SAMPLE_RATE)]
            # make the human part audible: its RMS to -20 dBFS (max +60 dB), soft-limit peaks
            rms = float(human_part.double().pow(2).mean().sqrt()) if human_part.numel() else 0.0
            gain = min(10 ** 3, (10 ** (-20 / 20)) / rms) if rms > 0 else 1.0
            clip = torch.tanh(clip * gain)
            name = f"{sess.name}_{k:03d}.wav"
            save_audio(clip_dir / name, clip, SAMPLE_RATE)
            item = {
                "id": f"{sess.name}_{k:03d}", "session": sess.name, "start": round(e.start_s, 3), "end": round(e.end_s, 3),
                "dur": round(e.duration_s, 3), "clip": f"clips/{name}", "clip_s": round(t1 - t0, 3),
                "ev0": round(ev0, 3), "ev1": round(ev1, 3), "gain_db": round(20 * math.log10(gain), 1),
                "kind": "label", "whisper": asr.get(item_id := f"{sess.name}_{k:03d}", {}).get("before", ""),
            }
            events.append(item)
            if item_id in audit_ids:
                after = hybrid[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)]
                a_rms = float(after.double().pow(2).mean().sqrt())
                a_gain = min(10 ** 3, (10 ** (-20 / 20)) / a_rms) if a_rms > 0 else 1.0
                save_audio(args.output_dir / "audit_clips" / name, torch.tanh(after * a_gain), SAMPLE_RATE)
                audits.append({**item, "id": f"audit:{item_id}", "kind": "audit", "clip": f"audit_clips/{name}",
                               "whisper": asr[item_id]["hybrid"]})

    if asr:  # order by priority (see module docstring)
        def is_priority(it):
            r = asr.get(it["id"])
            return r is not None and (int(r["words_before"]) >= 2 or it["dur"] > 1.5)

        priority = [dict(it, phase="priority") for it in events if is_priority(it)]
        rest = [it for it in events if not is_priority(it)]
        quiet = [it for it in rest if it["id"] in asr and int(asr[it["id"]]["words_before"]) == 0]
        spot_ids = {it["id"] for it in random.Random(0).sample(quiet, min(args.n_spotcheck, len(quiet)))}
        spot = [dict(it, phase="spotcheck") for it in rest if it["id"] in spot_ids]
        optional = [dict(it, phase="optional") for it in rest if it["id"] not in spot_ids]
        events = priority + spot + [dict(a, phase="audit") for a in audits] + optional
        print(f"priority {len(priority)}, spot-check {len(spot)}, leak audit {len(audits)}, optional {len(optional)}")
    (args.output_dir / "label_human_events.html").write_text(PAGE.replace("__EVENTS__", json.dumps(events)), encoding="utf-8")
    print(f"{len(events)} clips + label_human_events.html written to {args.output_dir}")


if __name__ == "__main__":
    main()
