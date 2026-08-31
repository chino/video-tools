"""
Literal (task=transcribe, not task=translate) transcription of every span of
one language, using faster-whisper large-v3. Produces raw text in that
language -- translation, if needed, is a separate step (see
translate_with_claude.py).

Usage: python3 transcribe_literal.py <outdir> <lang> [glossary_file]

  <outdir>        the video's working directory (has audio.wav, spans.tsv)
  <lang>          language code matching spans.tsv, e.g. zh or en
  [glossary_file] optional path to a plain-text file of topic-specific terms
                  (comma-separated) to bias recognition of proper nouns/
                  jargon via Whisper's initial_prompt -- e.g. a show's
                  specific vocabulary. If omitted, <outdir>/glossary_<lang>.txt
                  is used automatically if it exists.
"""
import json
import os
import sys

import soundfile as sf
from faster_whisper import WhisperModel

SR = 16000


def main():
    outdir = sys.argv[1]
    lang = sys.argv[2]
    glossary_file = sys.argv[3] if len(sys.argv) > 3 else f'{outdir}/glossary_{lang}.txt'
    glossary = None
    if os.path.exists(glossary_file):
        glossary = open(glossary_file, encoding='utf-8').read().strip()

    model = WhisperModel("large-v3", device="cuda", compute_type="int8")

    audio, sr = sf.read(f"{outdir}/audio.wav", dtype="float32")
    assert sr == SR

    spans = []
    with open(f"{outdir}/spans.tsv") as f:
        for line in f:
            s, e, span_lang = line.strip().split("\t")
            if span_lang == lang:
                spans.append((float(s), float(e)))

    print(f'{len(spans)} {lang} spans to transcribe'
          + (f' (glossary: {glossary_file})' if glossary else ''), flush=True)

    results = []
    for i, (s, e) in enumerate(spans):
        clip = audio[int(s * SR):int(e * SR)]
        segs, info = model.transcribe(
            clip, language=lang, task='transcribe', beam_size=5,
            initial_prompt=glossary,
            condition_on_previous_text=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        text = ''.join(seg.text.strip() for seg in segs)
        results.append({'start': s, 'end': e, 'text': text})
        print(f'[{i+1}/{len(spans)}] {s:.1f}-{e:.1f}: {text}', flush=True)

    out_path = f'{outdir}/{lang}_transcripts.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
