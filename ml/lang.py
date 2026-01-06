from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Language codes we pass to XTTS / general TTS selection.
# Keep it small and explicit to avoid unexpected codes.
SUPPORTED_LANGS = {"it", "en", "es", "fr", "de"}

@dataclass
class LangResult:
    lang: str
    confidence: float = 0.0
    source: str = "fallback"  # manual|text_detect|audio_detect|fallback

    def as_dict(self) -> dict:
        return {"lang": self.lang, "confidence": float(self.confidence), "source": self.source}


def _normalize_lang(code: str) -> str:
    code = (code or "").strip().lower()
    if code in SUPPORTED_LANGS:
        return code
    # common variants
    if code.startswith("it"):
        return "it"
    if code.startswith("en"):
        return "en"
    if code.startswith("es"):
        return "es"
    if code.startswith("fr"):
        return "fr"
    if code.startswith("de"):
        return "de"
    return "en"


def detect_text_language(text: str) -> LangResult:
    """Fast, offline detection from text using Lingua if available."""
    t = (text or "").strip()
    if len(t) < 8:
        return LangResult(lang="en", confidence=0.0, source="fallback")

    try:
        from lingua import Language, LanguageDetectorBuilder
    except Exception:
        return LangResult(lang="en", confidence=0.0, source="fallback")

    detector = LanguageDetectorBuilder.from_languages(
        Language.ITALIAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
    ).build()

    lang = detector.detect_language_of(t)
    if lang is None:
        return LangResult(lang="en", confidence=0.0, source="fallback")

    mapping = {
        Language.ITALIAN: "it",
        Language.ENGLISH: "en",
        Language.SPANISH: "es",
        Language.FRENCH: "fr",
        Language.GERMAN: "de",
    }
    code = mapping.get(lang, "en")

    # Lingua has probability APIs, but they vary by version; keep it simple.
    # This "confidence" is a heuristic for logs/UI.
    return LangResult(lang=code, confidence=0.6, source="text_detect")


# Keep a single WhisperModel in memory (loading is expensive).
_WHISPER_MODEL = None

def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    from faster_whisper import WhisperModel
    _WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


@lru_cache(maxsize=128)
def _audio_lang_cached(path: str, size: int, mtime: int) -> Optional[LangResult]:
    """Internal cached audio language detection."""
    try:
        model = _get_whisper_model()
    except Exception:
        return None

    try:
        # We only need language; transcribe with tiny beam and VAD.
        _, info = model.transcribe(path, beam_size=1, vad_filter=True)
        lang = _normalize_lang(getattr(info, "language", "en"))
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        return LangResult(lang=lang, confidence=prob, source="audio_detect")
    except Exception:
        return None


def detect_audio_language(wav_path: str) -> Optional[LangResult]:
    """Detect language from reference WAV using faster-whisper, if installed."""
    p = Path(wav_path)
    if not p.exists():
        return None
    try:
        st = p.stat()
        return _audio_lang_cached(str(p), int(st.st_size), int(st.st_mtime))
    except Exception:
        return None


def resolve_language(
    *,
    text: str,
    text_lang_choice: str,
    audio_lang_choice: str,
    reference_wav: Optional[str],
) -> tuple[LangResult, LangResult, Optional[LangResult]]:
    """
    Resolve one language code to use for TTS.

    Returns: (final, text_detected, audio_detected)

    Policy:
    1) If text_lang_choice is explicit (not 'auto'), use it.
    2) Else detect from text.
    3) If audio_lang_choice is explicit language, use it.
    4) Else if audio_lang_choice == 'auto' and reference_wav exists, try audio detect and use it if confidence >= 0.5.
    5) Otherwise use detected text.
    """

    text_choice = (text_lang_choice or "auto").strip().lower()
    audio_choice = (audio_lang_choice or "auto").strip().lower()

    # Manual text choice wins
    if text_choice not in ("auto", ""):
        final = LangResult(lang=_normalize_lang(text_choice), confidence=1.0, source="manual")
        text_det = detect_text_language(text)
        aud_det = detect_audio_language(reference_wav) if reference_wav else None
        return final, text_det, aud_det

    text_det = detect_text_language(text)

    # Manual audio choice (explicit language) overrides
    if audio_choice not in ("auto", "same_as_text", ""):
        final = LangResult(lang=_normalize_lang(audio_choice), confidence=1.0, source="manual")
        aud_det = detect_audio_language(reference_wav) if reference_wav else None
        return final, text_det, aud_det

    aud_det = None
    if audio_choice == "auto" and reference_wav:
        aud_det = detect_audio_language(reference_wav)
        if aud_det and aud_det.confidence >= 0.5:
            return aud_det, text_det, aud_det

    # Same-as-text or fallback
    return text_det, text_det, aud_det
