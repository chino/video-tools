"""
Synthesizes the dub and applies music ducking -- the step that actually
produces the final mixed audio track.

Usage: python3 synth_and_splice.py <outdir> [dubbed_lang] [passthrough_lang]
                                    [--voice VOICE]

  <outdir>            the video's working directory (has translated_spans.json,
                       audio_hq.wav, and demucs_out_<lang>/ for both languages
                       from separate_stems.py)
  [dubbed_lang]        language that gets fully replaced by TTS (default: zh)
  [passthrough_lang]   language whose original voice is kept, music ducked
                        underneath it (default: en)
  --voice              Kokoro voice name (default: am_michael)
"""
import argparse
import json
import numpy as np
import soundfile as sf
import librosa
from kokoro import KPipeline

SR = 48000
TTS_SR = 24000
CROSSFADE_SEC = 0.30   # transition zone for the dubbed-language (replace) path
DUCK_RAMP_SEC = 0.50   # slower ramp for the passthrough-language (subtractive
                        # duck) path -- no signal swap happening there, so a
                        # slightly longer ramp just makes the "music comes
                        # down/goes back up" feel more natural
TTS_EDGE_SEC = 0.08    # short fade on the TTS clip itself so it doesn't click in/out
DUCK_GAIN = 0.45       # music/ambience bed level under speech (~ -7dB), both paths
PAD = 2.0              # must match the padding used when span_clips*/*.wav were extracted


def synth(pipe, voice, text):
    chunks = []
    for _, _, audio in pipe(text, voice=voice):
        chunks.append(audio)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def fit_duration(mono_24k, target_n):
    target_sec = target_n / SR
    if len(mono_24k) < TTS_SR * 0.05 or target_n < int(SR * 0.05):
        return np.zeros(target_n, dtype=np.float32)
    resampled = librosa.resample(mono_24k, orig_sr=TTS_SR, target_sr=SR)
    src_dur = len(resampled) / SR
    ratio = src_dur / target_sec
    rate = max(1 / 3.0, min(ratio, 3.0))
    stretched = librosa.effects.time_stretch(resampled, rate=rate)
    if len(stretched) >= target_n:
        out = stretched[:target_n].copy()
    else:
        out = np.zeros(target_n, dtype=np.float32)
        out[:len(stretched)] = stretched
    edge_n = min(int(TTS_EDGE_SEC * SR), target_n // 2)
    if edge_n > 0:
        env = np.linspace(0, 1, edge_n, dtype=np.float32)
        out[:edge_n] *= env
        out[-edge_n:] *= env[::-1]
    return out


def equal_power_fade(n):
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    return np.sin(t), np.cos(t)  # fade-in, fade-out


def load_bed_wide(demucs_dir, idx, clip_start_sec, start_i, end_i, fade_n):
    """Load the isolated no-vocals bed for span `idx`, covering
    [start_i - fade_n, end_i + fade_n), resampled to SR if needed."""
    target_n = end_i - start_i
    clip_a = max(0, int((clip_start_sec - PAD) * SR))
    bed_path = f'{demucs_dir}/htdemucs/span_{idx:03d}/no_vocals.wav'
    bed_clip, bsr = sf.read(bed_path, dtype='float32')
    if bsr != SR:
        bed_clip = librosa.resample(bed_clip.T, orig_sr=bsr, target_sr=SR).T
    offset_in_clip = start_i - clip_a
    wide_a = offset_in_clip - fade_n
    wide_b = offset_in_clip + target_n + fade_n
    bed_wide = bed_clip[max(0, wide_a):wide_b]
    needed = target_n + 2 * fade_n
    if wide_a < 0 or len(bed_wide) < needed:
        padded = np.zeros((needed, 2), dtype=np.float32)
        copy_n = min(len(bed_wide), needed)
        padded[:copy_n] = bed_wide[:copy_n]
        bed_wide = padded
    return bed_wide


def replace_span(base_full, bed_wide, start_i, end_i, tts_mono, fade_n):
    """dubbed-language path: fully replace [start_i, end_i) with (ducked bed +
    TTS), crossfaded against the original mix at each edge -- real voice and
    TTS voice never overlap."""
    core_n = end_i - start_i
    tts_stereo = np.stack([tts_mono, tts_mono], axis=1)
    ducked_bed = bed_wide * DUCK_GAIN
    m = min(core_n, len(ducked_bed) - 2 * fade_n)
    core = ducked_bed[fade_n:fade_n + core_n].copy()
    core[:m] += tts_stereo[:m] * 1.15

    fade_in, fade_out = equal_power_fade(fade_n)
    fade_in = fade_in[:, None]
    fade_out = fade_out[:, None]

    pre = base_full[start_i - fade_n:start_i] * fade_out + ducked_bed[:fade_n] * fade_in
    post = ducked_bed[fade_n + core_n:fade_n + core_n + fade_n] * fade_out + \
        base_full[end_i:end_i + fade_n] * fade_in

    base_full[start_i - fade_n:start_i] = pre
    base_full[start_i:end_i] = core
    base_full[end_i:end_i + fade_n] = post


def duck_span_subtractive(base_full, bed_wide, start_i, end_i, fade_n):
    """passthrough-language path: keep the original mix (real voice untouched)
    and subtract a ramped fraction of the isolated bed from it --
    final = original - (1-DUCK)*bed = original_vocal + DUCK*bed. The real
    voice is never re-synthesized or reconstructed, only the already-isolated
    music/ambience component is reduced, so voice fidelity can't be hurt by
    separation artifacts."""
    core_n = end_i - start_i
    subtract_amt = (1 - DUCK_GAIN) * bed_wide  # what we'd remove at full duck

    ramp_up = np.linspace(0, 1, fade_n, dtype=np.float32)[:, None]
    ramp_down = ramp_up[::-1]
    flat = np.ones((core_n, 1), dtype=np.float32)
    envelope = np.concatenate([ramp_up, flat, ramp_down], axis=0)

    region = base_full[start_i - fade_n:end_i + fade_n]
    base_full[start_i - fade_n:end_i + fade_n] = region - subtract_amt * envelope


def process_dubbed(pipe, voice, outdir, dubbed_lang, out, spans, n_total, fade_n):
    dubbed_spans = [s for s in spans if s['lang'] == dubbed_lang and s['text']]
    print(f'{dubbed_lang}: synthesizing {len(dubbed_spans)} spans on GPU (voice={voice}), '
          f'full replace...', flush=True)
    demucs_dir = f'{outdir}/demucs_out_{dubbed_lang}'
    for i, s in enumerate(dubbed_spans):
        start_i = int(s['start'] * SR)
        end_i = min(int(s['end'] * SR), n_total)
        target_n = end_i - start_i
        this_fade_n = min(fade_n, start_i, n_total - end_i)
        bed_wide = load_bed_wide(demucs_dir, i, s['start'], start_i, end_i, this_fade_n)
        tts_24k = synth(pipe, voice, s['text'])
        fitted = fit_duration(tts_24k, target_n)
        replace_span(out, bed_wide, start_i, end_i, fitted, this_fade_n)
        print(f'  [{i+1}/{len(dubbed_spans)}] {s["start"]:.1f}-{s["end"]:.1f} '
              f'tts={len(tts_24k)/TTS_SR:.1f}s -> fit={len(fitted)/SR:.1f}s of {target_n/SR:.1f}s slot', flush=True)


def process_passthrough(outdir, passthrough_lang, out, spans, n_total, fade_n):
    passthrough_spans = [s for s in spans if s['lang'] == passthrough_lang]
    print(f'{passthrough_lang}: ducking music bed under {len(passthrough_spans)} spans, '
          f'voice left untouched...', flush=True)
    demucs_dir = f'{outdir}/demucs_out_{passthrough_lang}'
    for i, s in enumerate(passthrough_spans):
        start_i = int(s['start'] * SR)
        end_i = min(int(s['end'] * SR), n_total)
        this_fade_n = min(fade_n, start_i, n_total - end_i)
        bed_wide = load_bed_wide(demucs_dir, i, s['start'], start_i, end_i, this_fade_n)
        duck_span_subtractive(out, bed_wide, start_i, end_i, this_fade_n)
        print(f'  [{i+1}/{len(passthrough_spans)}] {s["start"]:.1f}-{s["end"]:.1f} ducked', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('dubbed_lang', nargs='?', default='zh')
    ap.add_argument('passthrough_lang', nargs='?', default='en')
    ap.add_argument('--voice', default='am_michael', help='Kokoro voice name')
    args = ap.parse_args()

    pipe = KPipeline(lang_code='a', device='cuda')

    original, sr = sf.read(f'{args.outdir}/audio_hq.wav', dtype='float32')
    assert sr == SR
    if original.ndim == 1:
        original = np.stack([original, original], axis=1)
    n_total = len(original)
    out = original.copy()

    spans = json.load(open(f'{args.outdir}/translated_spans.json'))

    process_dubbed(pipe, args.voice, args.outdir, args.dubbed_lang, out, spans, n_total,
                   int(CROSSFADE_SEC * SR))
    process_passthrough(args.outdir, args.passthrough_lang, out, spans, n_total,
                         int(DUCK_RAMP_SEC * SR))

    out_path = f'{args.outdir}/dubbed_audio.wav'
    sf.write(out_path, out, SR)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
