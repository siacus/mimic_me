"""
Pluggable ML providers for voice synthesis and voice cloning.

Supports:
- OpenAI TTS API
- ElevenLabs API  
- Coqui XTTS (local, open source)
- Edge TTS (free, Microsoft)

Configure via environment variables or config file.
"""

from __future__ import annotations

import os
import json
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result from TTS synthesis"""
    audio_path: str
    duration_seconds: float
    sample_rate: int
    provider: str
    cached: bool = False


@dataclass
class VoiceProfile:
    """Voice profile for cloning"""
    profile_id: str
    reference_audio_paths: List[str]
    embedding_path: Optional[str] = None
    provider_voice_id: Optional[str] = None  # For API-based cloning (ElevenLabs)


class TTSProvider(ABC):
    """Abstract base class for TTS providers"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def supports_cloning(self) -> bool:
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_profile: Optional[VoiceProfile] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        """Synthesize speech from text"""
        pass
    
    def clone_voice(
        self,
        reference_audio_paths: List[str],
        profile_id: str,
    ) -> VoiceProfile:
        """Create a voice profile from reference audio (if supported)"""
        raise NotImplementedError(f"{self.name} does not support voice cloning")


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "tts-1-hd"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None
        
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def supports_cloning(self) -> bool:
        return False  # OpenAI doesn't support custom voice cloning yet
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_profile: Optional[VoiceProfile] = None,
        emotion: Optional[str] = None,
        voice: str = "alloy",
        speed: float = 1.0,
        **kwargs
    ) -> TTSResult:
        """
        Synthesize using OpenAI TTS.
        
        Voices: alloy, echo, fable, onyx, nova, shimmer
        """
        client = self._get_client()
        
        # Map emotions to voices (rough approximation)
        emotion_voice_map = {
            "Happy": "nova",
            "Excited": "nova", 
            "Angry": "onyx",
            "Sarcastic": "fable",
            "Whisper": "shimmer",
            "Deadpan": "echo",
        }
        
        if emotion and emotion in emotion_voice_map:
            voice = emotion_voice_map[emotion]
        
        logger.info(f"OpenAI TTS: synthesizing with voice={voice}, speed={speed}")
        
        response = client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text,
            speed=speed,
        )
        
        response.stream_to_file(output_path)
        
        # Get duration from file
        import soundfile as sf
        info = sf.info(output_path)
        
        return TTSResult(
            audio_path=output_path,
            duration_seconds=info.duration,
            sample_rate=info.samplerate,
            provider=self.name,
        )


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs API provider with voice cloning support"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self._client = None
        
    @property
    def name(self) -> str:
        return "elevenlabs"
    
    @property
    def supports_cloning(self) -> bool:
        return True
    
    def _get_client(self):
        if self._client is None:
            try:
                from elevenlabs.client import ElevenLabs
                self._client = ElevenLabs(api_key=self.api_key)
            except ImportError:
                raise ImportError("elevenlabs package not installed. Run: pip install elevenlabs")
        return self._client
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_profile: Optional[VoiceProfile] = None,
        emotion: Optional[str] = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel (default)
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        **kwargs
    ) -> TTSResult:
        """Synthesize using ElevenLabs API"""
        client = self._get_client()
        
        # Use cloned voice if available
        if voice_profile and voice_profile.provider_voice_id:
            voice_id = voice_profile.provider_voice_id
            logger.info(f"ElevenLabs: using cloned voice {voice_id}")
        
        audio = client.generate(
            text=text,
            voice=voice_id,
            model="eleven_monolingual_v1",
        )
        
        # Write audio bytes to file
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        import soundfile as sf
        info = sf.info(output_path)
        
        return TTSResult(
            audio_path=output_path,
            duration_seconds=info.duration,
            sample_rate=info.samplerate,
            provider=self.name,
        )
    
    def clone_voice(
        self,
        reference_audio_paths: List[str],
        profile_id: str,
    ) -> VoiceProfile:
        """Clone voice using ElevenLabs Instant Voice Cloning"""
        client = self._get_client()
        
        # ElevenLabs requires audio files
        from elevenlabs import Voice, VoiceSettings
        
        voice = client.clone(
            name=f"mimic_me_{profile_id[:8]}",
            files=reference_audio_paths[:25],  # Max 25 files
            description=f"Cloned voice for Mimic Me profile {profile_id}",
        )
        
        logger.info(f"ElevenLabs: cloned voice with ID {voice.voice_id}")
        
        return VoiceProfile(
            profile_id=profile_id,
            reference_audio_paths=reference_audio_paths,
            provider_voice_id=voice.voice_id,
        )


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS (free, no API key required)"""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "edge"
    
    @property
    def supports_cloning(self) -> bool:
        return False
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_profile: Optional[VoiceProfile] = None,
        emotion: Optional[str] = None,
        language: str = "en",
        voice: str = "en-US-AriaNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        **kwargs
    ) -> TTSResult:
        """
        Synthesize using Edge TTS (free).
        
        Popular voices:
        - en-US-AriaNeural (female, expressive)
        - en-US-GuyNeural (male)
        - en-US-JennyNeural (female)
        - en-GB-SoniaNeural (British female)
        """
        try:
            import edge_tts
            import asyncio
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")
        EDGE_VOICE_BY_LANG = {
            "en": "en-US-AriaNeural",
            "it": "it-IT-ElsaNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
        }
        # If caller didn't explicitly pick a voice, choose one by language.
        # (Edge does not clone voices; this only changes the synthetic voice to match language.)
        lang_code = (language or "en").lower()
        if voice == "en-US-AriaNeural":
            voice = EDGE_VOICE_BY_LANG.get(lang_code, "en-US-AriaNeural")
        # Map emotions to voice styles (Edge supports some styles)
        emotion_style_map = {
            "Happy": ("cheerful", "+10%", "+5Hz"),
            "Excited": ("excited", "+20%", "+10Hz"),
            "Angry": ("angry", "+5%", "-5Hz"),
            "Sarcastic": ("ironic", "+0%", "+0Hz"),
            "Whisper": ("whispering", "-10%", "-5Hz"),
            "Deadpan": ("neutral", "-5%", "-2Hz"),
        }
        
        if emotion and emotion in emotion_style_map:
            _, rate, pitch = emotion_style_map[emotion]
        
        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(output_path)
        
        asyncio.run(_synthesize())
        
        import soundfile as sf
        info = sf.info(output_path)
        
        return TTSResult(
            audio_path=output_path,
            duration_seconds=info.duration,
            sample_rate=info.samplerate,
            provider=self.name,
        )


class CoquiXTTSProvider(TTSProvider):
    """
    Coqui XTTS v2 - Local, open source, supports voice cloning.
    
    Works great on Apple Silicon M1/M2/M3 with MPS acceleration!
    
    Requires: pip install TTS torch
    First run will download the model (~2GB).
    """
    
    def __init__(self, device: str = "auto", model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.device = device
        self.model_name = model_name
        self._tts = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        
    @property
    def name(self) -> str:
        return "coqui_xtts"
    
    @property
    def supports_cloning(self) -> bool:
        return True
    
    def _get_device(self) -> str:
        """Detect best device (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)"""
        if self.device != "auto":
            return self.device
            
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("Using Apple Silicon MPS acceleration for Coqui XTTS")
                return "mps"
            elif torch.cuda.is_available():
                logger.info("Using CUDA acceleration for Coqui XTTS")
                return "cuda"
        except ImportError:
            pass
        
        return "cpu"
    
    def _get_tts(self):
        if self._tts is None:
            try:
                from TTS.api import TTS
                import torch
                import re
                import importlib
                
                device = self._get_device()

                def _allowlist_from_error(err: Exception) -> bool:
                    """Allowlist globals required by torch.load(weights_only=True) for XTTS checkpoints.

                    PyTorch 2.6+ defaults `torch.load(weights_only=True)` which can block loading objects
                    inside trusted checkpoints unless they are allowlisted. Coqui TTS XTTS checkpoints
                    can trigger errors like:
                      "Unsupported global: GLOBAL TTS.tts.configs.xtts_config.XttsConfig"

                    We only auto-allowlist objects explicitly referenced in the error message.
                    """
                    msg = str(err)
                    if "Weights only load failed" not in msg or "Unsupported global: GLOBAL" not in msg:
                        return False

                    # Collect all "GLOBAL module.path.Class" occurrences
                    globals_paths = re.findall(r"Unsupported global: GLOBAL ([A-Za-z0-9_\.]+)", msg)
                    if not globals_paths:
                        return False

                    added_any = False
                    for dotted in globals_paths:
                        try:
                            module_path, attr = dotted.rsplit(".", 1)
                            mod = importlib.import_module(module_path)
                            obj = getattr(mod, attr)
                            torch.serialization.add_safe_globals([obj])
                            logger.warning(
                                "Allowlisted %s for torch.load(weights_only=True). "
                                "Only do this if you trust the XTTS model source.",
                                dotted,
                            )
                            added_any = True
                        except Exception as e2:
                            logger.warning("Failed to allowlist %s: %s", dotted, e2)
                    return added_any

                def _load_tts_on(dev: str):
                    """Load XTTS with a small retry loop to add safe globals if required."""
                    attempts = 0
                    while True:
                        try:
                            logger.info(f"Loading Coqui XTTS on device: {dev}")
                            return TTS(self.model_name).to(dev)
                        except Exception as e:
                            # Handle PyTorch weights_only safe-global errors (PyTorch 2.6+)
                            if attempts < 5 and _allowlist_from_error(e):
                                attempts += 1
                                continue
                            raise
                
                # Note: Some Coqui models have limited MPS support
                # Fall back to CPU if MPS fails
                try:
                    self._tts = _load_tts_on(device)
                except Exception as e:
                    if device == "mps":
                        logger.warning(f"MPS failed ({e}), falling back to CPU")
                        self._tts = _load_tts_on("cpu")
                    else:
                        raise
                
                logger.info("Coqui XTTS loaded successfully")
                
            except ImportError:
                raise ImportError(
                    "Coqui XTTS initialization failed (import error). This is usually a dependency/version mismatch (commonly `transformers`). Try pinning transformers (e.g., 4.46.2) and reinstalling TTS/torch in a clean env.\n"
                    "Note: First run will download ~2GB model."
                )
        return self._tts
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_profile: Optional[VoiceProfile] = None,
        emotion: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """
        Synthesize using Coqui XTTS.
        
        If voice_profile is provided, uses voice cloning.
        Otherwise uses default speaker.
        """
        tts = self._get_tts()
        
        # Get speaker reference for cloning
        speaker_wav = None
        if voice_profile and voice_profile.reference_audio_paths:
            speaker_wav = voice_profile.reference_audio_paths[0]
            if not os.path.exists(speaker_wav):
                logger.warning(f"Coqui XTTS: requested cloning reference does not exist: {speaker_wav}")
                speaker_wav = None
            else:
                logger.info(f"Coqui XTTS: cloning voice from {speaker_wav}")
        if not speaker_wav:
            logger.info("Coqui XTTS: no cloning reference provided, using default speaker")

        logger.info(f"Coqui XTTS: language={language}")
        
        if speaker_wav:
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=language,
            )
        else:
            # Use default speaker
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=language,
            )
        
        import soundfile as sf
        info = sf.info(output_path)
        
        return TTSResult(
            audio_path=output_path,
            duration_seconds=info.duration,
            sample_rate=info.samplerate,
            provider=self.name,
        )
    
    def clone_voice(
        self,
        reference_audio_paths: List[str],
        profile_id: str,
    ) -> VoiceProfile:
        """
        Create voice profile for XTTS cloning.
        
        XTTS does real-time cloning, so we just store the reference paths.
        For better quality, you can extract and cache speaker embeddings.
        """
        # XTTS can clone from a single reference, but more is better
        # We'll just store the paths - XTTS does real-time cloning
        return VoiceProfile(
            profile_id=profile_id,
            reference_audio_paths=reference_audio_paths,
        )


class PlaceholderTTSProvider(TTSProvider):
    """Fallback placeholder that generates tones (for testing)"""
    
    @property
    def name(self) -> str:
        return "placeholder"
    
    @property
    def supports_cloning(self) -> bool:
        return False
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_profile: Optional[VoiceProfile] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        """Generate a placeholder tone"""
        import soundfile as sf
        
        # Estimate duration from text (rough: 150 words/min)
        words = len(text.split())
        duration = max(1.0, words / 2.5)  # ~150 wpm
        
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Different tones for different emotions
        freq_map = {
            "Happy": 523,      # C5
            "Excited": 659,    # E5
            "Angry": 330,      # E4
            "Sarcastic": 440,  # A4
            "Whisper": 294,    # D4
            "Deadpan": 392,    # G4
        }
        
        freq = freq_map.get(emotion, 440)
        
        # Generate tone with envelope
        tone = 0.1 * np.sin(2 * np.pi * freq * t)
        envelope = np.minimum(1.0, np.minimum(t / 0.05, (duration - t) / 0.05))
        audio = tone * envelope
        
        sf.write(output_path, audio, sr)
        
        logger.warning(f"PlaceholderTTS: Generated {duration:.1f}s tone at {freq}Hz (text: {text[:30]}...)")
        
        return TTSResult(
            audio_path=output_path,
            duration_seconds=duration,
            sample_rate=sr,
            provider=self.name,
        )


class TTSManager:
    """
    Manager class for TTS providers with caching and fallback.
    
    Usage:
        manager = TTSManager()
        manager.set_provider("edge")  # or "openai", "elevenlabs", "coqui_xtts"
        result = manager.synthesize("Hello world", "output.wav")
    """
    
    PROVIDERS = {
        "openai": OpenAITTSProvider,
        "elevenlabs": ElevenLabsTTSProvider,
        "edge": EdgeTTSProvider,
        "coqui_xtts": CoquiXTTSProvider,
        "placeholder": PlaceholderTTSProvider,
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._providers: Dict[str, TTSProvider] = {}
        self._active_provider: Optional[str] = None
        
    def set_provider(
        self,
        name: str,
        **kwargs
    ) -> None:
        """Set the active TTS provider"""
        if name not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {name}. Available: {list(self.PROVIDERS.keys())}")
        
        if name not in self._providers:
            self._providers[name] = self.PROVIDERS[name](**kwargs)
        
        self._active_provider = name
        logger.info(f"TTSManager: active provider set to {name}")
    
    def get_provider(self) -> TTSProvider:
        """Get the active provider, with fallback to placeholder"""
        if self._active_provider and self._active_provider in self._providers:
            return self._providers[self._active_provider]
        
        # Try to auto-detect available provider
        for name in ["edge", "coqui_xtts", "openai", "elevenlabs"]:
            try:
                self.set_provider(name)
                return self._providers[name]
            except Exception as e:
                logger.debug(f"Provider {name} not available: {e}")
                continue
        
        # Fallback to placeholder
        logger.warning("No TTS providers available, using placeholder")
        self.set_provider("placeholder")
        return self._providers["placeholder"]
    
    def _get_cache_key(self, text: str, provider: str, **kwargs) -> str:
        """Generate cache key for TTS output"""
        data = json.dumps({"text": text, "provider": provider, **kwargs}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _voice_signature(self, voice_profile: Optional[VoiceProfile]) -> Optional[str]:
        """Create a stable-ish signature for the cloning reference.

        We intentionally avoid hashing full file contents (slow).
        Using (path, size, mtime) is enough to invalidate cache when the user uploads a new reference.
        """
        if not voice_profile or not voice_profile.reference_audio_paths:
            return None
        parts: List[str] = [voice_profile.profile_id]
        for p in voice_profile.reference_audio_paths:
            try:
                st = os.stat(p)
                parts.append(f"{p}|{st.st_size}|{int(st.st_mtime)}")
            except Exception:
                parts.append(str(p))
        return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_profile: Optional[VoiceProfile] = None,
        emotion: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> TTSResult:
        """
        Synthesize speech with optional caching.
        """
        provider = self.get_provider()
        
        # Check cache
        voice_sig = self._voice_signature(voice_profile)

        if use_cache and self.cache_dir:
            cache_key = self._get_cache_key(
                text,
                provider.name,
                emotion=emotion,
                voice_sig=voice_sig,
                language=kwargs.get("language"),
            )
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.wav")
            
            if os.path.exists(cache_path):
                import shutil
                shutil.copy(cache_path, output_path)
                
                import soundfile as sf
                info = sf.info(output_path)
                
                logger.info(f"TTSManager: cache hit for {cache_key}")
                return TTSResult(
                    audio_path=output_path,
                    duration_seconds=info.duration,
                    sample_rate=info.samplerate,
                    provider=provider.name,
                    cached=True,
                )
        
        # Synthesize
        try:
            result = provider.synthesize(
                text=text,
                output_path=output_path,
                voice_profile=voice_profile,
                emotion=emotion,
                **kwargs
            )
            
            # Cache result
            if use_cache and self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                import shutil
                shutil.copy(output_path, cache_path)
            
            return result
            
        except Exception as e:
            logger.error(f"TTS synthesis failed with {provider.name}: {e}")
            
            # Fallback to placeholder
            if provider.name != "placeholder":
                logger.warning("Falling back to placeholder TTS")
                placeholder = PlaceholderTTSProvider()
                return placeholder.synthesize(text, output_path, voice_profile, emotion)
            raise
    
    def clone_voice(
        self,
        reference_audio_paths: List[str],
        profile_id: str,
    ) -> Optional[VoiceProfile]:
        """Clone voice if provider supports it"""
        provider = self.get_provider()
        
        if not provider.supports_cloning:
            logger.warning(f"Provider {provider.name} does not support voice cloning")
            return None
        
        return provider.clone_voice(reference_audio_paths, profile_id)
