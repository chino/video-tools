"""
Isolates the music/ambience bed (vocals removed) for every speech span, so
the next step can duck the music under the dub or the original dialogue
without ever losing the score entirely. This is the slow, GPU-heavy step.

Runs Demucs per-span on padded clips, not on the whole file at once --
running it on a full-length video OOM-killed a memory-constrained (~9.7GB)
VM at ~100% completion, purely from Demucs' own peak RAM assembling a
full-length output (confirmed via dmesg, not a system crash). Per-span
clips keep memory bounded and only process the ~15-25 minutes of audio
that's actually speech, not the whole runtime.

Usage: python3 separate_stems.py <outdir> [dubbed_lang] [passthrough_lang]

  <outdir>            the video's working directory
  [dubbed_lang]        language that gets fully replaced by TTS (default: zh)
  [passthrough_lang]   language whose original voice is kept, music ducked
                        underneath it (default: en)
"""
import glob
import json
import os
import subprocess
import sys

import soundfile as sf

SR = 48000
PAD = 2.0  # seconds of context on each side of a span, given to Demucs for
           # better separation quality; also leaves room for the ducking
           # crossfade/ramp zones used later without ever touching the very
           # edge of an independently-processed clip


def extract_clips(outdir, spans, clip_dir):
    os.makedirs(clip_dir, exist_ok=True)
    audio, sr = sf.read(f'{outdir}/audio_hq.wav', dtype='float32')
    assert sr == SR
    n = len(audio)
    for i, (start, end) in enumerate(spans):
        a = max(0, int((start - PAD) * SR))
        b = min(n, int((end + PAD) * SR))
        sf.write(f'{clip_dir}/span_{i:03d}.wav', audio[a:b], SR)
    return len(spans)


def run_demucs(clip_dir, out_dir):
    clips = sorted(glob.glob(f'{clip_dir}/*.wav'))
    subprocess.run(
        ['python3', '-m', 'demucs', '--two-stems=vocals', '-n', 'htdemucs',
         '-d', 'cuda', *clips, '-o', out_dir],
        check=True,
    )


def main():
    outdir = sys.argv[1]
    dubbed_lang = sys.argv[2] if len(sys.argv) > 2 else 'zh'
    passthrough_lang = sys.argv[3] if len(sys.argv) > 3 else 'en'

    spans = json.load(open(f'{outdir}/translated_spans.json'))
    dubbed_spans = [(s['start'], s['end']) for s in spans if s['lang'] == dubbed_lang and s['text']]
    passthrough_spans = [(s['start'], s['end']) for s in spans if s['lang'] == passthrough_lang]

    print(f'extracting {len(dubbed_spans)} {dubbed_lang} + '
          f'{len(passthrough_spans)} {passthrough_lang} padded clips...', flush=True)
    extract_clips(outdir, dubbed_spans, f'{outdir}/span_clips_{dubbed_lang}')
    extract_clips(outdir, passthrough_spans, f'{outdir}/span_clips_{passthrough_lang}')

    print(f'running Demucs on {dubbed_lang} clips...', flush=True)
    run_demucs(f'{outdir}/span_clips_{dubbed_lang}', f'{outdir}/demucs_out_{dubbed_lang}')
    print(f'running Demucs on {passthrough_lang} clips...', flush=True)
    run_demucs(f'{outdir}/span_clips_{passthrough_lang}', f'{outdir}/demucs_out_{passthrough_lang}')
    print('done')


if __name__ == '__main__':
    main()
