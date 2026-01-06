#!/usr/bin/env python3
"""Environment sanity-check for mimic_me.

Usage:
  python scripts/test_env.py
Optionally set:
  MEDIAPIPE_FACE_LANDMARKER_MODEL=/path/to/face_landmarker.task
"""

from __future__ import annotations

import os
import sys
import platform
import importlib
import subprocess
from pathlib import Path


REQUIRED_PKGS = [
    "numpy",
    "torch",
    "transformers",
    "TTS",
    "mediapipe",
    "librosa",
    "soundfile",
    "cv2",
    "gradio",
    "lingua",
    "faster_whisper",
    "torchcodec",
]


def banner(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def ok(msg: str) -> None:
    print(f"[OK]  {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def get_version(modname: str) -> str:
    mod = importlib.import_module(modname)
    return getattr(mod, "__version__", "unknown")


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout.strip()
    except FileNotFoundError:
        return 127, ""


def check_imports_and_versions() -> None:
    banner("Python / Platform")
    ok(f"Python: {sys.version.split()[0]}")
    ok(f"Executable: {sys.executable}")
    ok(f"Platform: {platform.platform()}")

    banner("Package imports + versions")
    for pkg in REQUIRED_PKGS:
        try:
            ver = get_version(pkg)
            ok(f"{pkg}=={ver}")
        except Exception as e:
            fail(f"{pkg} import failed: {e}")


def check_ffmpeg() -> None:
    banner("ffmpeg")
    code, out = run_cmd(["ffmpeg", "-version"])
    if code == 0 and out:
        ok(out.splitlines()[0])
    else:
        warn("ffmpeg not found on PATH. (Video/audio extraction may fail.)")


def check_torch_mps() -> None:
    banner("PyTorch / MPS")
    try:
        import torch  # noqa
        ok(f"torch=={torch.__version__}")

        has_mps_attr = hasattr(torch.backends, "mps")
        if not has_mps_attr:
            warn("torch.backends.mps not present (likely non-mac build).")
            return

        ok(f"MPS built: {torch.backends.mps.is_built()}")
        ok(f"MPS available: {torch.backends.mps.is_available()}")

        if torch.backends.mps.is_available():
            x = torch.ones(3, device="mps")
            y = (x * 2).sum().item()
            ok(f"MPS tensor test succeeded (result={y}).")
    except Exception as e:
        fail(f"torch check failed: {e}")


def check_transformers() -> None:
    banner("Transformers sanity")
    try:
        import transformers  # noqa
        ok(f"transformers=={transformers.__version__}")
        from transformers import BeamSearchScorer  # noqa
        ok("BeamSearchScorer import: OK")
    except Exception as e:
        fail(f"transformers check failed: {e}")


def check_tts() -> None:
    banner("Coqui TTS sanity")
    try:
        from TTS.api import TTS  # noqa
        ok("from TTS.api import TTS: OK")
    except Exception as e:
        fail(f"TTS import/init check failed: {e}")


def check_mediapipe_tasks_face_landmarker() -> None:
    banner("MediaPipe Tasks / FaceLandmarker")
    try:
        import mediapipe as mp  # noqa
        ok(f"mediapipe=={mp.__version__}")
        from mediapipe.tasks import python as mp_python  # noqa
        from mediapipe.tasks.python import vision as mp_vision  # noqa
        ok("mediapipe.tasks.python.vision import: OK")
    except Exception as e:
        fail(f"MediaPipe Tasks import failed: {e}")
        return

    default_model = Path("models/mediapipe/face_landmarker.task")
    model_path = Path(os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL", str(default_model))).expanduser()

    if not model_path.exists():
        warn(f"FaceLandmarker model not found: {model_path}")
        warn("Run: python scripts/download_face_landmarker.py")
        return

    ok(f"FaceLandmarker model found: {model_path}")

    try:
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)
        ok("FaceLandmarker.create_from_options: OK")
        detector.close()
        ok("FaceLandmarker.close: OK")
    except Exception as e:
        fail(f"FaceLandmarker load failed: {e}")


def check_language_detection() -> None:
    banner("Language detection")
    try:
        from ml.lang import detect_text_language, detect_audio_language
        ok("ml.lang import: OK")
        res = detect_text_language("sono molto arrabbiato oggi")
        ok(f"text detect -> {res.lang} (source={res.source}, conf={res.confidence})")
    except Exception as e:
        fail(f"language detection check failed: {e}")


def main() -> None:
    check_imports_and_versions()
    check_ffmpeg()
    check_torch_mps()
    check_transformers()
    check_tts()
    check_mediapipe_tasks_face_landmarker()
    check_language_detection()

    banner("Done")
    print("If anything above is [FAIL], paste it here and I’ll pinpoint the fix.")


if __name__ == "__main__":
    main()
