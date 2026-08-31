# video-tools

A home for small video-processing tools, built as needed. Each one lives in
its own subfolder with its own README.

## Tools

- **[dubbing/](dubbing/README.md)** — auto-dub the foreign-language portions
  of a video into English (ASR → translation → TTS → music-preserving mix)
  and generate a subtitle file from the same transcripts. GPU-accelerated
  (faster-whisper, Demucs, Kokoro). Run with `./run.sh video.mkv`; see its
  README for language options and what's still missing versus a proper
  installed CLI tool.

Generated audio/video output never gets committed — each tool keeps its own
gitignored `output/` folder for that.
