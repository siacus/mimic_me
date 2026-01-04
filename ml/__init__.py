"""
ML module for Reachy Mimic Me.

Contains:
- providers.py: Pluggable TTS backends (OpenAI, ElevenLabs, Coqui, Edge)
- motion.py: Motion extraction and generation
- sync.py: Audio-motion synchronization learning
- config.py: Configuration management
"""

from .providers import (
    TTSManager,
    TTSProvider,
    TTSResult,
    VoiceProfile,
    OpenAITTSProvider,
    ElevenLabsTTSProvider,
    EdgeTTSProvider,
    CoquiXTTSProvider,
    PlaceholderTTSProvider,
)

from .motion import (
    MotionExtractor,
    MotionGenerator,
    MotionSequence,
    HeadPose,
    FacialFeatures,
)

from .sync import (
    SyncLearner,
    SyncModel,
    SyncExample,
    PreferenceLearner,
)

from .config import (
    Config,
    TTSConfig,
    MotionConfig,
    LearningConfig,
    ConfigManager,
    get_config,
    get_config_manager,
)
