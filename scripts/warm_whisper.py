#!/usr/bin/env python3
"""Warm up faster-whisper model download (tiny).

Usage:
  python scripts/warm_whisper.py /path/to/some.wav
"""

from __future__ import annotations
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/warm_whisper.py /path/to/audio.wav")
        sys.exit(2)

    wav = Path(sys.argv[1])
    if not wav.exists():
        print(f"File not found: {wav}")
        sys.exit(2)

    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    _, info = model.transcribe(str(wav), beam_size=1, vad_filter=True)
    print(f"Detected language: {info.language} (p={getattr(info,'language_probability',None)})")

if __name__ == "__main__":
    main()
