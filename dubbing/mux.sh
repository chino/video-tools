#!/usr/bin/env bash
# Adds the synthesized dub as a second audio track alongside the original --
# the video stream and original audio are stream-copied untouched, so this
# never re-encodes anything and the source is never modified in place.
#
# Usage: ./mux.sh /path/to/source.mkv <outdir>
set -euo pipefail

SOURCE="$1"
OUT_DIR="$2"
WAV="$OUT_DIR/dubbed_audio.wav"
AAC="$OUT_DIR/dubbed_audio.aac"
FINAL="$OUT_DIR/$(basename "$SOURCE" | sed -E 's/\.[^.]+$//')_dubbed.mkv"

ffmpeg -y -i "$WAV" -ar 48000 -ac 2 -c:a aac -b:a 256k "$AAC" -loglevel error

ffmpeg -y -i "$SOURCE" -i "$AAC" \
  -map 0:v -map 0:a -map 1:a \
  -c:v copy -c:a:0 copy -c:a:1 copy \
  -metadata:s:a:0 title="Original" \
  -metadata:s:a:1 language=eng -metadata:s:a:1 title="English Dub" \
  -disposition:a:0 0 -disposition:a:1 default \
  "$FINAL" -loglevel error

rm -f "$AAC"
echo "wrote $FINAL"
