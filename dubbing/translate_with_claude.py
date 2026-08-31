"""
Translates each foreign-language span's literal transcript into English by
shelling out to `claude -p` -- a real text-to-text translation judgment
call, not something faster-whisper's own task=translate mode is reliable at
(it drifts on figurative language and mishears rare proper nouns; see
README's "Known caveats"). No episode content is hardcoded here -- this
script contains no translations, only the prompt and the plumbing.

Target language is fixed to English (not a parameter) because the TTS step
downstream (Kokoro) only synthesizes English -- translating to anything else
wouldn't currently be usable by the rest of the pipeline.

Usage: python3 translate_with_claude.py <outdir> [src_lang]

  <outdir>   the video's working directory (has <src_lang>_transcripts.json,
             spans.tsv)
  [src_lang] language code to translate from, matching spans.tsv (default: zh)
"""
import json
import re
import subprocess
import sys

PROMPT_TEMPLATE = """You are translating literal ASR transcripts from a video into
natural, faithful English, for dubbing over the original audio.

Each transcript below came from an automatic speech recognizer and may contain
mis-transcribed characters -- ASR often substitutes a near-homophone for the
word actually spoken (e.g. a technical term or proper noun it doesn't
recognize). Use surrounding context to recover the intended meaning rather
than translating a nonsensical homophone literally.

Some entries may not be real spoken content at all -- ASR models sometimes
hallucinate boilerplate (e.g. subtitle-credit text, repeated looping phrases)
during silence or music-only audio, especially near the start or end of a
file. If an entry looks like this rather than genuine narration, output null
for it instead of translating it.

Translate faithfully and literally -- preserve the speaker's actual claims and
tone, don't embellish or add descriptive language that wasn't said.

Input is a JSON array of {"i": <index>, "text": <source-language text>}
objects. Output ONLY a JSON array of strings/nulls, same length and order as
the input, one English translation (or null) per entry. No other text.

Input:
%%ENTRIES_JSON%%
"""


def main():
    outdir = sys.argv[1]
    src_lang = sys.argv[2] if len(sys.argv) > 2 else 'zh'

    transcripts = json.load(open(f'{outdir}/{src_lang}_transcripts.json', encoding='utf-8'))
    entries = [{'i': i, 'text': t['text']} for i, t in enumerate(transcripts)]
    entries_json = json.dumps(entries, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.replace('%%ENTRIES_JSON%%', entries_json)

    print(f'translating {len(entries)} {src_lang} spans via claude -p...', flush=True)
    proc = subprocess.run(
        ['claude', '-p', '--output-format', 'json', '--restricted', '--model', 'sonnet', prompt],
        capture_output=True, text=True, check=True,
    )
    envelope = json.loads(proc.stdout)
    if envelope.get('is_error'):
        raise RuntimeError(f"claude -p reported an error: {envelope}")
    result_text = envelope['result']
    # models sometimes wrap JSON output in a markdown code fence despite
    # being told not to -- strip it defensively rather than fight the prompt
    fenced = re.match(r'^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$', result_text, re.DOTALL)
    if fenced:
        result_text = fenced.group(1)
    try:
        translations = json.loads(result_text)
    except json.JSONDecodeError:
        open('/tmp/claude_translate_debug.txt', 'w').write(result_text)
        raise RuntimeError(
            f'could not parse result as JSON (len={len(result_text)}); '
            f'raw text written to /tmp/claude_translate_debug.txt'
        )
    if len(translations) != len(transcripts):
        raise RuntimeError(f'expected {len(transcripts)} translations, got {len(translations)}')

    spans = []
    with open(f'{outdir}/spans.tsv') as f:
        for line in f:
            s, e, lang = line.strip().split('\t')
            spans.append({'start': float(s), 'end': float(e), 'lang': lang, 'text': None})

    src_starts = [t['start'] for t in transcripts]
    for start, text in zip(src_starts, translations):
        for span in spans:
            if span['lang'] == src_lang and span['start'] == start:
                span['text'] = text
                break

    out_path = f'{outdir}/translated_spans.json'
    with open(out_path, 'w') as f:
        json.dump(spans, f, indent=2)

    n_translated = sum(1 for s in spans if s['lang'] == src_lang and s['text'])
    n_dropped = sum(1 for s in spans if s['lang'] == src_lang) - n_translated
    print(f'wrote {out_path} '
          f'({n_translated} translated, {n_dropped} dropped as likely hallucinations)')


if __name__ == '__main__':
    main()
