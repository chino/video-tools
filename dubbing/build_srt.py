"""
Builds a full-episode English subtitle file directly from the transcripts
already produced for dubbing -- no OCR, no separate subtitle pipeline.
Just merges the dubbed language's translations (translated_spans.json) with
the passthrough language's literal transcript (already English) by
timestamp, since between the two every spoken span in the video is already
covered in English.

Usage: python3 build_srt.py <outdir> [dubbed_lang] [passthrough_lang]

  <outdir>            the video's working directory
  [dubbed_lang]        language that got translated (default: zh)
  [passthrough_lang]   language transcribed as-is, no translation (default: en)
"""
import json
import sys


def fmt_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def main():
    outdir = sys.argv[1]
    dubbed_lang = sys.argv[2] if len(sys.argv) > 2 else 'zh'
    passthrough_lang = sys.argv[3] if len(sys.argv) > 3 else 'en'

    entries = []

    for s in json.load(open(f'{outdir}/translated_spans.json')):
        if s['lang'] == dubbed_lang and s['text']:
            entries.append((s['start'], s['end'], s['text']))

    for s in json.load(open(f'{outdir}/{passthrough_lang}_transcripts.json')):
        text = s['text'].strip()
        if text:
            entries.append((s['start'], s['end'], text))

    entries.sort(key=lambda e: e[0])

    out_path = f'{outdir}/dub_generated.srt'
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, (start, end, text) in enumerate(entries, 1):
            f.write(f'{i}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{text}\n\n')

    print(f'wrote {len(entries)} subtitle entries to {out_path}')


if __name__ == '__main__':
    main()
