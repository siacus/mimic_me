from __future__ import annotations
import os
import hashlib
import subprocess
from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class UploadInfo:
    stored_path: str
    sha256: str

def sha256_file(path: str, chunk_size: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def store_upload(src_path: str, dest_dir: str) -> UploadInfo:
    os.makedirs(dest_dir, exist_ok=True)
    h = sha256_file(src_path)
    ext = os.path.splitext(src_path)[1].lower() or ".bin"
    dest = os.path.join(dest_dir, f"{h}{ext}")
    if not os.path.exists(dest):
        with open(src_path, "rb") as r, open(dest, "wb") as w:
            w.write(r.read())
    return UploadInfo(stored_path=dest, sha256=h)

def extract_audio_ffmpeg(video_path: str, wav_path: str, sr: int = 22050) -> bool:
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", wav_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            logger.warning("ffmpeg failed: %s", (p.stderr or "")[:2000])
            return False
        return True
    except FileNotFoundError:
        logger.warning("ffmpeg not found; skipping audio extraction.")
        return False

def try_extract_audio(video_path: str, wav_path: str) -> Tuple[bool, str]:
    ok = extract_audio_ffmpeg(video_path, wav_path)
    if ok:
        return True, "Audio extracted with ffmpeg."
    return False, "Audio extraction failed (ffmpeg missing or error)."
