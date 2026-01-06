#!/usr/bin/env python3
"""Download MediaPipe FaceLandmarker model bundle (.task) to the default path.

Usage:
  python scripts/download_face_landmarker.py

Output:
  models/mediapipe/face_landmarker.task
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

def main():
    out = Path("models/mediapipe/face_landmarker.task")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FaceLandmarker model to: {out}")
    urllib.request.urlretrieve(URL, out)
    print("Done.")

if __name__ == "__main__":
    main()
