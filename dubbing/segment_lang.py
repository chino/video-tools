"""
Finds contiguous same-language spans in the audio -- the "where does each
language start/stop" map every later step works from.

Two-pass approach, not a single language-tagged transcription: pass 1 finds
speech boundaries with VAD alone (language-agnostic, so it isn't biased by
whatever language a naive single-pass transcription might lock onto first).
Pass 2 runs Whisper's detect_language() independently on each resulting
window, then merges adjacent same-language windows into spans. This avoids
a real Whisper failure mode: transcribing the whole file in one pass tends
to detect a language once (e.g. from the first 30s) and then force that
language on the rest of the audio, mistranscribing anything in a different
language for the rest of the file.

Usage: python3 segment_lang.py <outdir>

Reads <outdir>/audio.wav, writes <outdir>/spans.tsv.
"""
import sys
from collections import Counter

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.vad import get_speech_timestamps, VadOptions

SR = 16000


def main():
    outdir = sys.argv[1]
    model = WhisperModel("small", device="cpu", compute_type="int8")

    audio, sr = sf.read(f"{outdir}/audio.wav", dtype="float32")
    assert sr == SR
    print(f'loaded {len(audio)/SR:.1f}s of audio')

    # 1. language-agnostic VAD boundaries
    raw_chunks = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=400))
    print(f'{len(raw_chunks)} raw VAD chunks')

    # 2. group raw VAD chunks into ~8s windows for reliable per-window langID
    windows = []
    cur_start = None
    cur_end = None
    for ch in raw_chunks:
        s, e = ch['start'], ch['end']
        if cur_start is None:
            cur_start, cur_end = s, e
        elif s - cur_end > SR * 1.5 or (e - cur_start) > SR * 12:
            windows.append((cur_start, cur_end))
            cur_start, cur_end = s, e
        else:
            cur_end = e
    if cur_start is not None:
        windows.append((cur_start, cur_end))
    print(f'{len(windows)} grouped windows for langID')

    # 3. detect language per window independently
    labeled = []
    for s, e in windows:
        clip = audio[s:e]
        if len(clip) < SR * 0.5:
            continue
        lang, prob, _ = model.detect_language(clip)
        labeled.append((s, e, lang, prob))

    # 4. merge adjacent windows sharing the same language into contiguous spans
    spans = []
    for s, e, lang, prob in labeled:
        if spans and spans[-1][2] == lang and s - spans[-1][1] < SR * 2:
            spans[-1] = (spans[-1][0], e, lang)
        else:
            spans.append((s, e, lang))

    print(f'{len(spans)} contiguous language spans')
    totals = Counter()
    for s, e, lang in spans:
        totals[lang] += (e - s) / SR
    print('  ' + ', '.join(f'{lang}={secs:.1f}s' for lang, secs in totals.most_common()))

    with open(f'{outdir}/spans.tsv', 'w') as f:
        for s, e, lang in spans:
            f.write(f'{s/SR:.2f}\t{e/SR:.2f}\t{lang}\n')
    print(f'wrote {outdir}/spans.tsv')


if __name__ == '__main__':
    main()
