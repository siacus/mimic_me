from __future__ import annotations
import os
import hashlib
import subprocess
import shutil
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
    """Extract audio from video using ffmpeg"""
    
    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.error("❌ ffmpeg not found in PATH. Install with: brew install ffmpeg")
        return False
    
    # Check if input video exists
    if not os.path.exists(video_path):
        logger.error(f"❌ Video file not found: {video_path}")
        return False
    
    logger.info(f"🎬 Extracting audio from: {video_path}")
    logger.info(f"   Using ffmpeg: {ffmpeg_path}")
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,  # Input file
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", str(sr),  # Sample rate
        "-f", "wav",  # Output format
        wav_path
    ]
    
    try:
        logger.info(f"   Running: {' '.join(cmd)}")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if p.returncode != 0:
            logger.error(f"❌ ffmpeg failed with return code {p.returncode}")
            logger.error(f"   stderr: {(p.stderr or '')[:1000]}")
            return False
        
        # Verify output was created
        if not os.path.exists(wav_path):
            logger.error(f"❌ Output WAV not created: {wav_path}")
            return False
        
        # Check file size
        size = os.path.getsize(wav_path)
        if size < 1000:
            logger.warning(f"⚠️  WAV file is very small ({size} bytes) - might be empty")
        
        logger.info(f"✅ Audio extracted successfully: {wav_path} ({size} bytes)")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("❌ ffmpeg timed out after 30 seconds")
        return False
    except FileNotFoundError:
        logger.error("❌ ffmpeg command not found")
        return False
    except Exception as e:
        logger.exception(f"❌ Unexpected error during audio extraction: {e}")
        return False


def try_extract_audio(video_path: str, wav_path: str) -> Tuple[bool, str]:
    """Try to extract audio, return success status and message"""
    ok = extract_audio_ffmpeg(video_path, wav_path)
    if ok:
        return True, "Audio extracted with ffmpeg."
    return False, "Audio extraction failed (check logs for details)."

