#!/usr/bin/env python3
"""
Test if ffmpeg can extract audio from your video
"""

import os
import sys
import subprocess
import shutil

# Your video file from the logs
VIDEO_PATH = "/Users/jago/github/mimic_me/data/uploads/4a6e1d39/be7a13615a11da65a0e230579a13aa0b39e70c4d9feab748eeed38491e40f411.webm"
OUTPUT_WAV = "/tmp/test_audio.wav"


def test_ffmpeg():
    print("=" * 70)
    print("TESTING FFMPEG AUDIO EXTRACTION")
    print("=" * 70)
    
    # Check if ffmpeg is installed
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("❌ ERROR: ffmpeg not found in PATH")
        print("\nInstall ffmpeg:")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt-get install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return False
    
    print(f"✅ Found ffmpeg: {ffmpeg_path}")
    
    # Check ffmpeg version
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        version_line = result.stdout.split('\n')[0]
        print(f"✅ Version: {version_line}")
    except Exception as e:
        print(f"⚠️  Could not get version: {e}")
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"\n❌ ERROR: Video file not found:")
        print(f"   {VIDEO_PATH}")
        print("\nThis means the video upload worked, but the file is in a different location.")
        print("Check: ls -la data/uploads/*/")
        return False
    
    print(f"\n✅ Found video: {VIDEO_PATH}")
    video_size = os.path.getsize(VIDEO_PATH)
    print(f"   Size: {video_size:,} bytes ({video_size / 1024 / 1024:.2f} MB)")
    
    # Try to extract audio
    print(f"\n🎬 Extracting audio to: {OUTPUT_WAV}")
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-i", VIDEO_PATH,
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", "22050",  # 22kHz
        "-f", "wav",
        OUTPUT_WAV
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    print("\n--- ffmpeg output ---")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Print stderr (ffmpeg outputs to stderr even on success)
        if result.stderr:
            print(result.stderr[:2000])
        
        if result.returncode != 0:
            print(f"\n❌ ffmpeg failed with return code {result.returncode}")
            return False
        
        # Check if output was created
        if not os.path.exists(OUTPUT_WAV):
            print(f"\n❌ Output file was not created: {OUTPUT_WAV}")
            return False
        
        wav_size = os.path.getsize(OUTPUT_WAV)
        print(f"\n✅ SUCCESS! Audio extracted:")
        print(f"   Output: {OUTPUT_WAV}")
        print(f"   Size: {wav_size:,} bytes ({wav_size / 1024:.2f} KB)")
        
        if wav_size < 1000:
            print(f"   ⚠️  WARNING: File is very small, might be empty")
        
        print(f"\n💡 You can play it with: afplay {OUTPUT_WAV}")
        return True
        
    except subprocess.TimeoutExpired:
        print("\n❌ ffmpeg timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    success = test_ffmpeg()
    print("\n" + "=" * 70)
    if success:
        print("✅ FFMPEG WORKS! Audio extraction is functional.")
        print("\nThe issue is likely in how app_gradio.py calls the extraction.")
    else:
        print("❌ FFMPEG TEST FAILED")
        print("\nFix the errors above before continuing.")
    print("=" * 70)
    sys.exit(0 if success else 1)
    
