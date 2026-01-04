"""
Configuration management for Mimic Me.

Supports:
- Environment variables
- Config file (config.json)
- Runtime overrides
- Apple Silicon (M1/M2/M3) MPS detection
"""

from __future__ import annotations

import os
import json
import logging
import platform
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """
    Detect the best available device for ML operations.
    
    Returns: 'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' otherwise
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("Detected Apple Silicon - using MPS acceleration")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("Detected NVIDIA GPU - using CUDA")
            return "cuda"
    except ImportError:
        pass
    
    # Check if we're on Apple Silicon even without torch
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        logger.info("Detected Apple Silicon Mac (torch not installed, will use MPS when available)")
        return "mps"
    
    return "cpu"


@dataclass
class TTSConfig:
    """TTS provider configuration"""
    provider: str = "edge"  # edge, openai, elevenlabs, coqui_xtts, placeholder
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "tts-1-hd"
    openai_voice: str = "alloy"
    
    # ElevenLabs settings
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    
    # Coqui XTTS settings
    coqui_device: str = "auto"  # auto, mps, cuda, cpu
    coqui_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    # Edge TTS settings
    edge_voice: str = "en-US-AriaNeural"
    
    # General
    cache_enabled: bool = True
    cache_dir: Optional[str] = None


@dataclass
class MotionConfig:
    """Motion extraction and generation configuration"""
    device: str = "auto"  # auto, mps, cuda, cpu (auto detects Apple Silicon)
    use_gpu: bool = True  # Enable GPU/MPS acceleration
    target_fps: float = 30.0
    
    # MediaPipe settings
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class LearningConfig:
    """Learning pipeline configuration"""
    sync_learning_enabled: bool = True
    preference_learning_enabled: bool = True
    auto_train_on_approve: bool = True
    min_examples_for_training: int = 3


@dataclass
class Config:
    """Main configuration"""
    tts: TTSConfig = field(default_factory=TTSConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    
    # Paths
    data_dir: str = "data"
    models_dir: str = "models"
    cache_dir: str = "cache"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tts": {
                "provider": self.tts.provider,
                "openai_model": self.tts.openai_model,
                "openai_voice": self.tts.openai_voice,
                "elevenlabs_voice_id": self.tts.elevenlabs_voice_id,
                "coqui_device": self.tts.coqui_device,
                "coqui_model": self.tts.coqui_model,
                "edge_voice": self.tts.edge_voice,
                "cache_enabled": self.tts.cache_enabled,
            },
            "motion": {
                "device": self.motion.device,
                "use_gpu": self.motion.use_gpu,
                "target_fps": self.motion.target_fps,
                "min_detection_confidence": self.motion.min_detection_confidence,
                "min_tracking_confidence": self.motion.min_tracking_confidence,
            },
            "learning": {
                "sync_learning_enabled": self.learning.sync_learning_enabled,
                "preference_learning_enabled": self.learning.preference_learning_enabled,
                "auto_train_on_approve": self.learning.auto_train_on_approve,
                "min_examples_for_training": self.learning.min_examples_for_training,
            },
            "data_dir": self.data_dir,
            "models_dir": self.models_dir,
            "cache_dir": self.cache_dir,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        config = cls()
        
        if "tts" in data:
            tts = data["tts"]
            config.tts.provider = tts.get("provider", config.tts.provider)
            config.tts.openai_model = tts.get("openai_model", config.tts.openai_model)
            config.tts.openai_voice = tts.get("openai_voice", config.tts.openai_voice)
            config.tts.elevenlabs_voice_id = tts.get("elevenlabs_voice_id", config.tts.elevenlabs_voice_id)
            config.tts.coqui_device = tts.get("coqui_device", config.tts.coqui_device)
            config.tts.coqui_model = tts.get("coqui_model", config.tts.coqui_model)
            config.tts.edge_voice = tts.get("edge_voice", config.tts.edge_voice)
            config.tts.cache_enabled = tts.get("cache_enabled", config.tts.cache_enabled)
        
        if "motion" in data:
            motion = data["motion"]
            config.motion.device = motion.get("device", config.motion.device)
            config.motion.use_gpu = motion.get("use_gpu", config.motion.use_gpu)
            config.motion.target_fps = motion.get("target_fps", config.motion.target_fps)
            config.motion.min_detection_confidence = motion.get("min_detection_confidence", config.motion.min_detection_confidence)
            config.motion.min_tracking_confidence = motion.get("min_tracking_confidence", config.motion.min_tracking_confidence)
        
        if "learning" in data:
            learning = data["learning"]
            config.learning.sync_learning_enabled = learning.get("sync_learning_enabled", config.learning.sync_learning_enabled)
            config.learning.preference_learning_enabled = learning.get("preference_learning_enabled", config.learning.preference_learning_enabled)
            config.learning.auto_train_on_approve = learning.get("auto_train_on_approve", config.learning.auto_train_on_approve)
            config.learning.min_examples_for_training = learning.get("min_examples_for_training", config.learning.min_examples_for_training)
        
        config.data_dir = data.get("data_dir", config.data_dir)
        config.models_dir = data.get("models_dir", config.models_dir)
        config.cache_dir = data.get("cache_dir", config.cache_dir)
        
        return config


class ConfigManager:
    """
    Manages configuration loading and saving.
    
    Priority (highest to lowest):
    1. Runtime overrides
    2. Environment variables
    3. Config file
    4. Defaults
    """
    
    ENV_PREFIX = "MIMIC_ME_"
    
    def __init__(self, root_dir: str, config_file: str = "config.json"):
        self.root_dir = root_dir
        self.config_file = os.path.join(root_dir, config_file)
        self._config: Optional[Config] = None
    
    def load(self) -> Config:
        """Load configuration from all sources"""
        # Start with defaults
        config = Config()
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                config = Config.from_dict(data)
                logger.info(f"Loaded config from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        # Set absolute paths
        config.data_dir = os.path.join(self.root_dir, config.data_dir)
        config.models_dir = os.path.join(self.root_dir, config.models_dir)
        config.cache_dir = os.path.join(self.root_dir, config.cache_dir)
        
        if config.tts.cache_enabled and not config.tts.cache_dir:
            config.tts.cache_dir = os.path.join(config.cache_dir, "tts")
        
        # Create directories
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.models_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
        
        self._config = config
        return config
    
    def _apply_env_overrides(self, config: Config) -> Config:
        """Apply environment variable overrides"""
        
        # TTS provider
        provider = os.environ.get(f"{self.ENV_PREFIX}TTS_PROVIDER")
        if provider:
            config.tts.provider = provider
        
        # API keys (always check env for security)
        config.tts.openai_api_key = os.environ.get("OPENAI_API_KEY")
        config.tts.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
        
        # Auto-select provider based on available API keys
        if config.tts.provider == "auto":
            if config.tts.openai_api_key:
                config.tts.provider = "openai"
                logger.info("Auto-selected OpenAI TTS (API key found)")
            elif config.tts.elevenlabs_api_key:
                config.tts.provider = "elevenlabs"
                logger.info("Auto-selected ElevenLabs TTS (API key found)")
            else:
                config.tts.provider = "edge"
                logger.info("Auto-selected Edge TTS (no API keys found)")
        
        # Other env overrides
        if os.environ.get(f"{self.ENV_PREFIX}USE_GPU"):
            config.motion.use_gpu = os.environ.get(f"{self.ENV_PREFIX}USE_GPU", "").lower() == "true"
        
        return config
    
    def save(self, config: Optional[Config] = None) -> None:
        """Save configuration to file"""
        config = config or self._config
        if not config:
            return
        
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved config to {self.config_file}")
    
    def get(self) -> Config:
        """Get current configuration"""
        if self._config is None:
            return self.load()
        return self._config
    
    def update(self, **kwargs) -> Config:
        """Update configuration values"""
        config = self.get()
        
        # Handle nested updates
        for key, value in kwargs.items():
            if "." in key:
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            elif hasattr(config.tts, key):
                setattr(config.tts, key, value)
            elif hasattr(config.motion, key):
                setattr(config.motion, key, value)
            elif hasattr(config.learning, key):
                setattr(config.learning, key, value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        self._config = config
        return config


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(root_dir: Optional[str] = None) -> ConfigManager:
    """Get or create the global config manager"""
    global _config_manager
    
    if _config_manager is None:
        if root_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _config_manager = ConfigManager(root_dir)
    
    return _config_manager


def get_config() -> Config:
    """Get the current configuration"""
    return get_config_manager().get()
