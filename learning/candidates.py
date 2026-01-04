"""
Candidate generation with real ML pipeline.

Generates voice + motion candidates using:
- Pluggable TTS providers (OpenAI, ElevenLabs, Coqui, Edge)
- Motion extraction from reference video
- Audio-motion synchronization (learned or rule-based)
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import soundfile as sf

from core.reachy_driver import Pose, TimelinePoint, ReachyMiniDriver

logger = logging.getLogger(__name__)


@dataclass
class GeneratedTake:
    """A single generated candidate"""
    episode_id: str
    preview_audio_path: Optional[str]
    timeline: List[TimelinePoint]
    metadata: Dict[str, Any] = None


class CandidateGenerator:
    """
    Generates voice + motion candidates using ML pipeline.
    
    Pipeline:
    1. Synthesize voice (TTS with optional cloning)
    2. Extract motion from reference video (if provided)
    3. Generate motion from audio features
    4. Apply emotion and comedian modifiers
    """
    
    def __init__(self, storage, reachy: ReachyMiniDriver, config=None):
        self.storage = storage
        self.reachy = reachy
        self.config = config
        
        # Lazy-loaded ML components
        self._tts_manager = None
        self._motion_extractor = None
        self._motion_generator = None
        self._sync_learner = None
    
    def _get_tts_manager(self):
        """Lazy-load TTS manager"""
        if self._tts_manager is None:
            try:
                from ml.providers import TTSManager
                from ml.config import get_config
                
                config = get_config()
                cache_dir = config.tts.cache_dir if config.tts.cache_enabled else None
                
                self._tts_manager = TTSManager(cache_dir=cache_dir)
                self._tts_manager.set_provider(config.tts.provider)
                logger.info(f"Initialized TTS manager with provider: {config.tts.provider}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize TTS manager: {e}")
                from ml.providers import TTSManager
                self._tts_manager = TTSManager()
                self._tts_manager.set_provider("placeholder")
        
        return self._tts_manager
    
    def _get_motion_extractor(self):
        """Lazy-load motion extractor"""
        if self._motion_extractor is None:
            try:
                from ml.motion import MotionExtractor
                from ml.config import get_config
                
                config = get_config()
                self._motion_extractor = MotionExtractor(use_gpu=config.motion.use_gpu)
                logger.info("Initialized motion extractor")
                
            except Exception as e:
                logger.warning(f"Failed to initialize motion extractor: {e}")
                self._motion_extractor = None
        
        return self._motion_extractor
    
    def _get_motion_generator(self):
        """Lazy-load motion generator"""
        if self._motion_generator is None:
            try:
                from ml.motion import MotionGenerator
                self._motion_generator = MotionGenerator()
                logger.info("Initialized motion generator")
            except Exception as e:
                logger.warning(f"Failed to initialize motion generator: {e}")
        
        return self._motion_generator
    
    def _get_sync_learner(self):
        """Lazy-load sync learner"""
        if self._sync_learner is None:
            try:
                from ml.sync import SyncLearner
                from ml.config import get_config
                
                config = get_config()
                models_dir = os.path.join(config.models_dir, "sync")
                self._sync_learner = SyncLearner(models_dir)
                logger.info("Initialized sync learner")
                
            except Exception as e:
                logger.warning(f"Failed to initialize sync learner: {e}")
        
        return self._sync_learner
    
    def _make_placeholder_audio(self, out_wav: str, text: str, emotion: str, seconds: float = 3.0) -> str:
        """Fallback: generate placeholder tone"""
        sr = 22050
        
        words = len(text.split())
        duration = max(1.0, min(seconds, words / 2.5))
        
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        freq_map = {
            "Happy": 523,
            "Excited": 659,
            "Angry": 330,
            "Sarcastic": 440,
            "Whisper": 294,
            "Deadpan": 392,
        }
        
        base_emotion = emotion.split(":")[0] if ":" in emotion else emotion
        freq = freq_map.get(base_emotion, 440)
        
        tone = 0.08 * np.sin(2 * np.pi * freq * t)
        envelope = np.minimum(1.0, np.minimum(t / 0.05, (duration - t) / 0.05))
        audio = tone * envelope
        
        sf.write(out_wav, audio, sr)
        logger.warning(f"Generated placeholder audio ({duration:.1f}s at {freq}Hz)")
        
        return out_wav
    
    def _synthesize_voice(
        self,
        text: str,
        output_path: str,
        voice_mode: str,
        emotion: str,
        profile_id: str,
        reference_audio_path: Optional[str] = None,
    ) -> Optional[str]:
        """Synthesize voice using configured TTS provider"""
        
        tts = self._get_tts_manager()
        if tts is None:
            return self._make_placeholder_audio(output_path, text, emotion)
        
        try:
            voice_profile = None
            if voice_mode == "imitate" and reference_audio_path:
                from ml.providers import VoiceProfile
                voice_profile = VoiceProfile(
                    profile_id=profile_id,
                    reference_audio_paths=[reference_audio_path],
                )
            
            result = tts.synthesize(
                text=text,
                output_path=output_path,
                voice_profile=voice_profile,
                emotion=emotion,
            )
            
            logger.info(f"Synthesized voice: {result.duration_seconds:.2f}s via {result.provider}")
            return result.audio_path
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return self._make_placeholder_audio(output_path, text, emotion)
    
    def _extract_reference_motion(
        self,
        video_path: str,
        assets_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract motion from reference video"""
        
        extractor = self._get_motion_extractor()
        if extractor is None:
            logger.warning("Motion extractor not available")
            return None
        
        try:
            sequence = extractor.extract_from_video(video_path)
            if sequence:
                motion_path = os.path.join(assets_dir, "reference_motion.json")
                extractor.save_sequence(sequence, motion_path)
                logger.info(f"Extracted motion: {len(sequence.head_poses)} frames")
                return sequence.to_dict()
        except Exception as e:
            logger.error(f"Motion extraction failed: {e}")
        
        return None
    
    def _generate_motion(
        self,
        audio_path: str,
        emotion: str,
        comedian: float,
        antenna_role: str = "Eyebrows",
        reference_motion: Optional[Dict[str, Any]] = None,
    ) -> List[TimelinePoint]:
        """Generate motion timeline from audio"""
        
        generator = self._get_motion_generator()
        sync_learner = self._get_sync_learner()
        
        audio_features = {}
        if generator:
            try:
                audio_features = generator.extract_audio_features(audio_path)
            except Exception as e:
                logger.warning(f"Audio feature extraction failed: {e}")
        
        timeline_data = None
        if sync_learner and audio_features:
            try:
                profile = sync_learner.get_profile(emotion)
                if profile.get("n_examples", 0) > 0:
                    timeline_data = sync_learner.apply_profile(
                        audio_features, emotion, comedian
                    )
                    logger.info(f"Using learned sync profile for {emotion}")
            except Exception as e:
                logger.warning(f"Sync learner failed: {e}")
        
        if timeline_data is None and generator:
            try:
                if reference_motion:
                    from ml.motion import MotionSequence
                    ref_seq = MotionSequence.from_dict(reference_motion)
                    duration = audio_features.get("duration", 3.0)
                    timeline_data = generator.transfer_motion(ref_seq, duration, comedian)
                    logger.info("Using transferred motion from reference")
                else:
                    timeline_data = generator.synthesize_motion(
                        audio_features, emotion, comedian, antenna_role
                    )
                    logger.info("Using synthesized motion from audio")
            except Exception as e:
                logger.warning(f"Motion generation failed: {e}")
        
        if timeline_data is None:
            timeline_data = self._generate_random_motion(
                duration=audio_features.get("duration", 3.0),
                comedian=comedian,
            )
            logger.info("Using random fallback motion")
        
        timeline = []
        for pt in timeline_data:
            pose = pt.get("pose", {})
            timeline.append(TimelinePoint(
                t=float(pt.get("t", 0)),
                pose=Pose(
                    pitch=float(pose.get("pitch", 0)),
                    roll=float(pose.get("roll", 0)),
                    yaw=float(pose.get("yaw", 0)),
                    base=float(pose.get("base", 0)),
                    antenna_l=float(pose.get("antenna_l", 0)),
                    antenna_r=float(pose.get("antenna_r", 0)),
                )
            ))
        
        return timeline
    
    def _generate_random_motion(
        self,
        duration: float = 3.0,
        comedian: float = 1.0,
        fps: int = 30,
    ) -> List[Dict[str, Any]]:
        """Generate smooth random motion (fallback)"""
        n = int(duration * fps)
        ts = np.linspace(0, duration, n)
        rng = np.random.default_rng()
        
        def curve(scale, limit):
            x = rng.normal(0, 1, size=n)
            x = np.cumsum(x)
            x = (x - x.mean()) / (x.std() + 1e-6)
            x = np.clip(scale * x, -limit, limit)
            for _ in range(2):
                x = np.convolve(x, np.ones(7) / 7, mode="same")
            return x
        
        pitch = curve(8, 18) * comedian
        yaw = curve(10, 30) * comedian
        roll = curve(6, 15) * comedian
        al = curve(6, 20) * comedian
        ar = curve(6, 20) * comedian
        
        timeline = []
        for i, t in enumerate(ts):
            timeline.append({
                "t": float(t),
                "pose": {
                    "pitch": float(np.clip(pitch[i], -25, 25)),
                    "yaw": float(np.clip(yaw[i], -45, 45)),
                    "roll": float(np.clip(roll[i], -25, 25)),
                    "base": 0.0,
                    "antenna_l": float(np.clip(al[i], -30, 30)),
                    "antenna_r": float(np.clip(ar[i], -30, 30)),
                }
            })
        
        return timeline
    
    def generate_candidates(
        self,
        profile_id: str,
        emotion: str,
        text: str,
        voice_mode: str,
        comedian: float,
        n_candidates: int = 2,
        source_type: str = "live",
        reference_video_path: Optional[str] = None,
        reference_audio_path: Optional[str] = None,
        antenna_role: str = "Eyebrows",
    ) -> List[GeneratedTake]:
        """Generate candidate takes with voice + motion."""
        logger.info("=" * 60)
        logger.info(f"GENERATING {n_candidates} CANDIDATES")
        logger.info(f"  Profile: {profile_id[:8]}")
        logger.info(f"  Emotion: {emotion}")
        logger.info(f"  Voice mode: {voice_mode}")
        logger.info(f"  Text: {text[:50]}...")
        logger.info(f"  Reference video: {'YES' if reference_video_path else 'NO'}")
        logger.info(f"  Reference audio: {'YES' if reference_audio_path else 'NO'}")
        
        n_candidates = max(1, min(int(n_candidates), 4))
        takes: List[GeneratedTake] = []
        
        reference_motion = None
        if reference_video_path:
            # Cache extracted motion next to the uploaded video so it survives across sessions
            cache_dir = os.path.join(self.storage.paths.uploads_dir, profile_id)
            os.makedirs(cache_dir, exist_ok=True)
            reference_motion = self._extract_reference_motion(
                reference_video_path,
                cache_dir,
            )
        
        for i in range(n_candidates):
            logger.info(f"\n--- Candidate {i+1}/{n_candidates} ---")
            
            episode_id = self.storage.create_episode(
                profile_id=profile_id,
                source_type=source_type,
                emotion=emotion,
                text=text,
                voice_mode=voice_mode,
                comedian=comedian,
                approved=False,
                notes=f"candidate_{i}",
            )
            logger.info(f"Created episode: {episode_id[:8]}")
            
            assets_dir = os.path.join(self.storage.paths.episodes_dir, episode_id)
            os.makedirs(assets_dir, exist_ok=True)
            
            conditioning = {
                "reference_video_path": reference_video_path,
                "reference_audio_path": reference_audio_path,
                "voice_mode": voice_mode,
                "emotion": emotion,
                "text": text,
                "comedian": float(comedian),
                "antenna_role": antenna_role,
                "candidate_index": i,
            }
            with open(os.path.join(assets_dir, "conditioning.json"), "w") as f:
                json.dump(conditioning, f, indent=2)
            
            wav_path = os.path.join(assets_dir, "preview.wav")
            self._synthesize_voice(
                text=text,
                output_path=wav_path,
                voice_mode=voice_mode,
                emotion=emotion,
                profile_id=profile_id,
                reference_audio_path=reference_audio_path,
            )
            
            timeline = self._generate_motion(
                audio_path=wav_path,
                emotion=emotion,
                comedian=comedian,
                antenna_role=antenna_role,
                reference_motion=reference_motion,
            )
            
            timeline = self.reachy.exaggerate_timeline(timeline, comedian=comedian)
            
            timeline_path = os.path.join(assets_dir, "timeline.json")
            with open(timeline_path, "w") as f:
                json.dump([
                    {"t": pt.t, "pose": pt.pose.__dict__}
                    for pt in timeline
                ], f, indent=2)
            
            takes.append(GeneratedTake(
                episode_id=episode_id,
                preview_audio_path=wav_path,
                timeline=timeline,
                metadata=conditioning,
            ))
            
            logger.info(f"✓ Generated candidate {i+1}: {len(timeline)} motion frames")
        
        logger.info(f"\n✓ GENERATED {len(takes)} CANDIDATES")
        logger.info("=" * 60)
        
        return takes
