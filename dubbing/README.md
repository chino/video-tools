# dubbing

Auto-dubs the foreign-language portions of a video into English (TTS), keeps
existing English-spoken portions untouched voice-wise, ducks background
music under all dialogue, and generates a full-episode English subtitle
file.

**What this actually is:** a wrapper script (`run.sh`) that runs 7
python/shell steps in sequence. Each step takes an `outdir` argument, so
multiple videos can be processed side by side without clobbering each
other's data (`output/<video-slug>/`). It's a real script you point at a
video, not just a list of steps to run by hand — but it's not an installed
CLI tool either: no pip package, no `--help`, positional arguments only.

The source language (default `zh`) and the passthrough language (default
`en`) are genuine parameters, not hardcoded. The *target* language is fixed
to English, though — the TTS engine (Kokoro) only synthesizes a handful of
languages, so translating to anything else wouldn't be usable downstream.
See "What's still missing" below for the rest of the gap versus a real
installed tool.

Setup: see [SETUP.md](SETUP.md) for GPU/torch version gotchas and the
package list.

## Pipeline

```
source.mkv  (video + stereo audio, no subtitle streams)
  │
  ▼
ffmpeg: extract 16kHz mono (ASR) + 48kHz stereo (final mix)
  │
  ▼
faster-whisper "small" (GPU): VAD + per-window language ID
  → spans.tsv (contiguous dubbed-lang / passthrough-lang / other spans)
  │
  ▼
faster-whisper "large-v3" (GPU): task=transcribe, per span
  → <lang>_transcripts.json, one file per language (literal text)
  │
  ▼
claude -p (text-only, no tool access): translates each dubbed-lang transcript
  → translated_spans.json
  (entries that look like a Whisper hallucination — e.g. silence or outro
  music transcribed as fake content — come back null and are skipped)
  │
  ▼
SUBTITLES: merge translations + passthrough transcripts, sort by time
  → dub_generated.srt (no OCR involved)
  │
  ▼
Demucs "htdemucs" (GPU, per-span padded clips, both languages)
  → isolated music/ambience bed for every speech span
  │
  ▼
dubbed-lang spans:                    passthrough-lang spans:
  Kokoro TTS (GPU) synthesizes          original voice kept
  English speech, time-stretched        completely untouched
  to fit the original slot
                                         SUBTRACT a ramped fraction of
  REPLACE the span: original →          the isolated bed from the
  (ducked bed + TTS) → original,        original mix, ramped in/out —
  crossfaded so the real voice          only the music is reduced, so
  and the TTS voice never overlap       separation artifacts can't
                                         reach the voice
  │                                      │
  └──────────────┬───────────────────────┘
                  ▼
       output/<slug>/dubbed_audio.wav
                  │
                  ▼
       run.sh's mux step → output/<slug>/<name>_dubbed.mkv
```

## Layout

```
run.sh                    the wrapper — runs everything below in order
segment_lang.py           step 1: VAD + language ID
transcribe_literal.py     step 2: literal transcript, one language per call
translate_with_claude.py  step 3: shells out to `claude -p` to translate —
                                  no episode content hardcoded here, only
                                  the prompt
build_srt.py              step 4: merges translations + passthrough
                                  transcripts into a subtitle file
separate_stems.py         step 5: per-span Demucs vocal/music separation
synth_and_splice.py       step 6: TTS + duck/replace → dub audio track
mux.sh                    step 7: mux the dub track onto the source video

output/<slug>/            gitignored, one directory per video (slug is the
                          source filename, sanitized). Everything generated
                          lives here: audio.wav, audio_hq.wav, spans.tsv,
                          <lang>_transcripts.json, translated_spans.json,
                          dub_generated.srt, span_clips_<lang>/,
                          demucs_out_<lang>/, dubbed_audio.wav,
                          <name>_dubbed.mkv
```

## Usage

```bash
./run.sh /path/to/video.mkv
```

That's the whole thing: it extracts audio, runs all 7 steps, and writes the
final muxed video to `output/<slug>/<name>_dubbed.mkv`.

Optional positional arguments:

```bash
./run.sh video.mkv [dubbed_lang] [passthrough_lang] [glossary_file]
```

Defaults: `zh`, `en`, no glossary. The glossary file is a comma-separated
list of proper nouns or jargon specific to this video, used to bias
Whisper's recognition of terms it would otherwise mishear — a show's own
vocabulary, say. Write your own per video; none is checked into this repo,
since it would be video-specific content. If you omit the argument,
`output/<slug>/glossary_<lang>.txt` is used automatically when it exists.

### Running steps individually

Useful for debugging, or for resuming after a failure without redoing
earlier steps. Every script takes the same `outdir` as its first argument:

```bash
SOURCE=/path/to/video.mkv
OUTDIR=output/my_video_slug

python3 segment_lang.py "$OUTDIR"
python3 transcribe_literal.py "$OUTDIR" zh [glossary_file]
python3 transcribe_literal.py "$OUTDIR" en
python3 translate_with_claude.py "$OUTDIR" zh
python3 build_srt.py "$OUTDIR" zh en
python3 separate_stems.py "$OUTDIR" zh en
python3 synth_and_splice.py "$OUTDIR" zh en [--voice am_michael]
./mux.sh "$SOURCE" "$OUTDIR"
```

Re-running just `synth_and_splice.py` with a different `--voice` is fast
(~2 minutes for a ~50-span video), since step 5's Demucs stems are cached
on disk and reused as-is. Delete `demucs_out_*/` under the outdir to force
a clean re-separation.

## Known caveats

- Ducking only triggers during detected speech (dub or original dialogue).
  Music-only stretches with no dialogue are left at full volume by design,
  not lowered globally.
- `DUCK_GAIN = 0.45` (~ -7dB) in `synth_and_splice.py` was chosen by ear on
  a handful of spans, not rigorously tuned.
- **Never run Demucs on a full-length video in one call.** It OOM-killed a
  memory-constrained (~9.7GB) VM at ~100% completion, purely from its own
  peak RAM while assembling a full-length output (confirmed via `dmesg`,
  not a system crash). `separate_stems.py` always splits into per-span
  padded clips first, keeping memory bounded and only processing the audio
  actually needed.
- Only a handful of spans have been spot-checked by ear per run — not a
  full watch-through.
- `run.sh` has no resume/skip-if-exists logic. If step 5 fails partway
  through, re-running `run.sh` redoes steps 1–4 too; use the individual-step
  commands above to resume from a specific step instead.

## What's still missing

`run.sh` genuinely handles "run this on an arbitrary video with an
arbitrary source/passthrough language pair" — that part is real and
tested. Versus an actual installed tool, still missing:

- Skip-if-already-done logic in `run.sh` (see the caveat above).
- Support for a non-English target language — would mean swapping in a
  different TTS engine for the synth step, since Kokoro only covers a
  handful of languages, not just adding a parameter.
- A packaging decision for `translate_with_claude.py`. Right now it shells
  out to a `claude` CLI the user must have installed and authenticated —
  reasonable for personal use, worth reconsidering for wider distribution.
- Actual packaging: this is a folder of scripts you clone and run, not
  `pip install video-dubber && dub video.mkv`.
- A decision on what belongs in the repo versus `output/` for a video whose
  data you *do* want to keep long-term. Right now everything video-specific
  is gitignored, full stop — right for "this repo ships no one's specific
  video content," but there's no supported way to check in a finished
  project's data if you wanted to.
