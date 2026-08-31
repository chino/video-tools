# Setup

Tested on: Ubuntu, Python 3.10, dual Pascal-generation GPUs (compute
capability sm_61, 8GB VRAM each). If you're on newer GPUs (Turing/Ampere+)
the `torch` pin below is unnecessary — just `pip install torch` normally.

```bash
pip install faster-whisper soundfile librosa demucs kokoro
```

## Pascal-GPU gotcha (sm_61 cards only)

A stock recent `pip install torch` may ship **no working kernels for sm_61**
at all, while still silently reporting `torch.cuda.is_available() == True` --
any real op just fails at kernel-launch time, not at import time. Fix:

```bash
pip install --user torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

(2.5.1 is the floor version `kokoro`'s `transformers` dependency will
accept; the `cu121` build still includes sm_61 kernels, unlike more recent
`cu13x` builds.) Verify with:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); x=torch.randn(10,device='cuda'); print((x@x).item())"
```

If the matmul actually runs (not just `cuda.is_available()==True`), you're
good.

Also on these cards: `faster-whisper`'s `WhisperModel(..., compute_type=...)`
needs `int8`, not `float16` -- Pascal has no efficient fp16 path and
CTranslate2 raises `ValueError` on load if you ask for it.

## Other requirements

- `ffmpeg` with NVENC support if you want GPU-accelerated encode/decode for
  quick spot-check clips (not required for the pipeline itself, which only
  uses `ffmpeg` for plain audio extraction/muxing).
- A CUDA GPU. Everything here (`faster-whisper`, `demucs`, `kokoro`) runs on
  CPU too, just much slower -- change `device='cuda'` to `device='cpu'` in
  each script if needed.
- `claude` CLI on `PATH`, authenticated, for the translation step
  (`translate_with_claude.py`).
