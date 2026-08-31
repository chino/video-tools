#!/usr/bin/env bash
# The top-level entry point: runs all 7 pipeline steps on one video, start
# to finish, writing everything to output/<slug>/ (slug derived from the
# source filename) so multiple videos can be processed without one run's
# files overwriting another's. No skip-if-already-done logic -- if a step
# fails partway through, re-running this script redoes everything before
# it too; see the individual scripts if you need to resume from one step.
#
# Usage: ./run.sh /path/to/video.mkv [dubbed_lang] [passthrough_lang] [glossary_file]
#
#   dubbed_lang       language to translate + TTS-dub over (default: zh)
#   passthrough_lang  language whose original voice is kept, music ducked
#                      underneath it (default: en)
#   glossary_file     optional path to a topic-glossary file for the
#                      dubbed-language transcription step (see
#                      transcribe_literal.py)
set -euo pipefail

SOURCE="$1"
DUBBED_LANG="${2:-zh}"
PASSTHROUGH_LANG="${3:-en}"
GLOSSARY="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLUG=$(basename "$SOURCE" | sed -E 's/\.[^.]+$//' | tr -c 'A-Za-z0-9_-' '_')
OUTDIR="$SCRIPT_DIR/output/$SLUG"
mkdir -p "$OUTDIR"

echo "== [0/7] extracting audio -> $OUTDIR =="
ffmpeg -y -i "$SOURCE" -vn -ac 1 -ar 16000 "$OUTDIR/audio.wav" -loglevel error
ffmpeg -y -i "$SOURCE" -vn -ac 2 -ar 48000 "$OUTDIR/audio_hq.wav" -loglevel error

echo "== [1/7] segmenting language spans =="
python3 "$SCRIPT_DIR/segment_lang.py" "$OUTDIR"

echo "== [2/7] transcribing $DUBBED_LANG spans =="
if [ -n "$GLOSSARY" ]; then
  python3 "$SCRIPT_DIR/transcribe_literal.py" "$OUTDIR" "$DUBBED_LANG" "$GLOSSARY"
else
  python3 "$SCRIPT_DIR/transcribe_literal.py" "$OUTDIR" "$DUBBED_LANG"
fi

echo "== [2/7] transcribing $PASSTHROUGH_LANG spans =="
python3 "$SCRIPT_DIR/transcribe_literal.py" "$OUTDIR" "$PASSTHROUGH_LANG"

echo "== [3/7] translating $DUBBED_LANG -> English =="
python3 "$SCRIPT_DIR/translate_with_claude.py" "$OUTDIR" "$DUBBED_LANG"

echo "== [4/7] building subtitle file =="
python3 "$SCRIPT_DIR/build_srt.py" "$OUTDIR" "$DUBBED_LANG" "$PASSTHROUGH_LANG"

echo "== [5/7] separating music/vocal stems (slow step) =="
python3 "$SCRIPT_DIR/separate_stems.py" "$OUTDIR" "$DUBBED_LANG" "$PASSTHROUGH_LANG"

echo "== [6/7] synthesizing dub + ducking =="
python3 "$SCRIPT_DIR/synth_and_splice.py" "$OUTDIR" "$DUBBED_LANG" "$PASSTHROUGH_LANG"

echo "== [7/7] muxing final video =="
"$SCRIPT_DIR/mux.sh" "$SOURCE" "$OUTDIR"

echo "done -- see $OUTDIR"
