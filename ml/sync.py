"""
Audio-motion synchronization learning.

Learns the mapping between audio features and motion from approved examples.
Uses a lightweight approach suitable for incremental learning.
"""

from __future__ import annotations

import os
import json
import logging
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyncExample:
    """A single example of audio-motion synchronization"""
    episode_id: str
    emotion: str
    audio_features: Dict[str, np.ndarray]
    motion_timeline: List[Dict[str, Any]]
    antenna_role: str
    approval_rank: int = 0  # Lower is better (0 = approved as best)


@dataclass  
class SyncModel:
    """
    Lightweight sync model that learns per-emotion mappings.
    
    For each emotion, stores:
    - Mean audio feature profiles
    - Motion response patterns
    - Learned scaling factors
    """
    emotion_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    global_scale: float = 1.0
    version: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion_profiles": {
                k: {
                    "n_examples": v.get("n_examples", 0),
                    "energy_to_pitch": v.get("energy_to_pitch", 1.0),
                    "energy_to_yaw": v.get("energy_to_yaw", 0.5),
                    "pitch_to_pitch": v.get("pitch_to_pitch", 1.0),
                    "brightness_to_antenna": v.get("brightness_to_antenna", 1.0),
                    "antenna_role_weights": v.get("antenna_role_weights", {}),
                }
                for k, v in self.emotion_profiles.items()
            },
            "global_scale": self.global_scale,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncModel":
        model = cls()
        model.emotion_profiles = data.get("emotion_profiles", {})
        model.global_scale = data.get("global_scale", 1.0)
        model.version = data.get("version", 0)
        return model


class SyncLearner:
    """
    Learns audio-motion synchronization from approved examples.
    
    Key features:
    - Incremental learning (add examples without full retraining)
    - Per-emotion profiles
    - Preference learning from approval rankings
    - Lightweight (no deep learning required)
    """
    
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.model = SyncModel()
        self._examples: List[SyncExample] = []
        
        os.makedirs(storage_dir, exist_ok=True)
        self._load_model()
    
    def _model_path(self) -> str:
        return os.path.join(self.storage_dir, "sync_model.json")
    
    def _examples_path(self) -> str:
        return os.path.join(self.storage_dir, "sync_examples.pkl")
    
    def _load_model(self) -> None:
        """Load model from disk if exists"""
        path = self._model_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.model = SyncModel.from_dict(data)
                logger.info(f"Loaded sync model v{self.model.version}")
            except Exception as e:
                logger.warning(f"Failed to load sync model: {e}")
        
        # Load examples
        examples_path = self._examples_path()
        if os.path.exists(examples_path):
            try:
                with open(examples_path, "rb") as f:
                    self._examples = pickle.load(f)
                logger.info(f"Loaded {len(self._examples)} sync examples")
            except Exception as e:
                logger.warning(f"Failed to load examples: {e}")
    
    def _save_model(self) -> None:
        """Save model to disk"""
        with open(self._model_path(), "w", encoding="utf-8") as f:
            json.dump(self.model.to_dict(), f, indent=2)
        
        with open(self._examples_path(), "wb") as f:
            pickle.dump(self._examples, f)
        
        logger.info(f"Saved sync model v{self.model.version}")
    
    def add_example(
        self,
        episode_id: str,
        emotion: str,
        audio_features: Dict[str, np.ndarray],
        motion_timeline: List[Dict[str, Any]],
        antenna_role: str,
        approval_rank: int = 0,
    ) -> None:
        """Add a new training example"""
        example = SyncExample(
            episode_id=episode_id,
            emotion=emotion,
            audio_features=audio_features,
            motion_timeline=motion_timeline,
            antenna_role=antenna_role,
            approval_rank=approval_rank,
        )
        self._examples.append(example)
        logger.info(f"Added sync example: {episode_id[:8]}, emotion={emotion}")
    
    def learn(self) -> Dict[str, Any]:
        """
        Learn from all examples.
        
        For each emotion:
        1. Analyze approved examples
        2. Compute mapping coefficients
        3. Weight by approval rank
        
        Returns learning statistics.
        """
        if not self._examples:
            return {"status": "no_examples", "version": self.model.version}
        
        # Group examples by emotion
        by_emotion: Dict[str, List[SyncExample]] = {}
        for ex in self._examples:
            by_emotion.setdefault(ex.emotion, []).append(ex)
        
        stats = {"emotions": {}, "total_examples": len(self._examples)}
        
        for emotion, examples in by_emotion.items():
            # Weight examples by approval rank (lower = better)
            weights = []
            for ex in examples:
                # Rank 0 (best) gets weight 1.0, rank 1 gets 0.5, etc.
                w = 1.0 / (1 + ex.approval_rank)
                weights.append(w)
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Compute weighted average mappings
            profile = self._learn_emotion_profile(examples, weights)
            self.model.emotion_profiles[emotion] = profile
            
            stats["emotions"][emotion] = {
                "n_examples": len(examples),
                "profile": profile,
            }
        
        self.model.version += 1
        self._save_model()
        
        stats["version"] = self.model.version
        stats["status"] = "success"
        
        return stats
    
    def _learn_emotion_profile(
        self,
        examples: List[SyncExample],
        weights: np.ndarray,
    ) -> Dict[str, Any]:
        """Learn profile for a single emotion from weighted examples"""
        
        # Analyze correlations in each example
        energy_pitch_corrs = []
        energy_yaw_corrs = []
        pitch_pitch_corrs = []
        brightness_antenna_corrs = []
        antenna_role_counts: Dict[str, int] = {}
        
        for ex in examples:
            # Count antenna roles
            antenna_role_counts[ex.antenna_role] = antenna_role_counts.get(ex.antenna_role, 0) + 1
            
            audio = ex.audio_features
            motion = ex.motion_timeline
            
            if not audio or not motion:
                continue
            
            # Extract motion values
            n_frames = len(motion)
            pitches = np.array([m["pose"]["pitch"] for m in motion])
            yaws = np.array([m["pose"]["yaw"] for m in motion])
            antennas = np.array([
                (m["pose"]["antenna_l"] + m["pose"]["antenna_r"]) / 2 
                for m in motion
            ])
            
            # Get audio features (resample to match motion length)
            energy = audio.get("energy", np.ones(n_frames) * 0.5)
            audio_pitch = audio.get("pitch", np.ones(n_frames) * 0.5)
            brightness = audio.get("brightness", np.ones(n_frames) * 0.5)
            
            # Resample audio to motion length
            if len(energy) != n_frames:
                energy = np.interp(
                    np.linspace(0, 1, n_frames),
                    np.linspace(0, 1, len(energy)),
                    energy
                )
                audio_pitch = np.interp(
                    np.linspace(0, 1, n_frames),
                    np.linspace(0, 1, len(audio_pitch)),
                    audio_pitch
                )
                brightness = np.interp(
                    np.linspace(0, 1, n_frames),
                    np.linspace(0, 1, len(brightness)),
                    brightness
                )
            
            # Compute correlations (with safety for zero variance)
            def safe_corr(a, b):
                if np.std(a) < 1e-6 or np.std(b) < 1e-6:
                    return 0.0
                return float(np.corrcoef(a, b)[0, 1])
            
            energy_pitch_corrs.append(safe_corr(energy, pitches))
            energy_yaw_corrs.append(safe_corr(energy, np.abs(yaws)))
            pitch_pitch_corrs.append(safe_corr(audio_pitch, pitches))
            brightness_antenna_corrs.append(safe_corr(brightness, antennas))
        
        # Compute weighted averages
        def weighted_avg(values, weights):
            values = np.array(values)
            valid = ~np.isnan(values)
            if not np.any(valid):
                return 0.5
            return float(np.average(values[valid], weights=weights[valid]))
        
        # Convert correlations to scaling factors (0-2 range)
        profile = {
            "n_examples": len(examples),
            "energy_to_pitch": 1.0 + weighted_avg(energy_pitch_corrs, weights),
            "energy_to_yaw": 0.5 + 0.5 * weighted_avg(energy_yaw_corrs, weights),
            "pitch_to_pitch": 1.0 + weighted_avg(pitch_pitch_corrs, weights),
            "brightness_to_antenna": 1.0 + weighted_avg(brightness_antenna_corrs, weights),
            "antenna_role_weights": antenna_role_counts,
        }
        
        return profile
    
    def get_profile(self, emotion: str) -> Dict[str, Any]:
        """Get learned profile for an emotion (with fallback to defaults)"""
        if emotion in self.model.emotion_profiles:
            return self.model.emotion_profiles[emotion]
        
        # Check for custom emotion base
        if ":" in emotion:
            base = emotion.split(":")[0]
            if base in self.model.emotion_profiles:
                return self.model.emotion_profiles[base]
        
        # Return defaults
        return {
            "n_examples": 0,
            "energy_to_pitch": 1.0,
            "energy_to_yaw": 0.5,
            "pitch_to_pitch": 1.0,
            "brightness_to_antenna": 1.0,
            "antenna_role_weights": {},
        }
    
    def apply_profile(
        self,
        audio_features: Dict[str, np.ndarray],
        emotion: str,
        comedian: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate motion using learned profile.
        
        This is the inference step that applies learned mappings.
        """
        profile = self.get_profile(emotion)
        
        duration = audio_features.get("duration", 3.0)
        time_axis = audio_features.get("time", np.linspace(0, duration, 90))
        energy = audio_features.get("energy", np.ones_like(time_axis) * 0.5)
        pitch = audio_features.get("pitch", np.ones_like(time_axis) * 0.5)
        brightness = audio_features.get("brightness", np.ones_like(time_axis) * 0.5)
        onset = audio_features.get("onset", np.zeros_like(time_axis))
        
        # Get most common antenna role
        antenna_weights = profile.get("antenna_role_weights", {})
        antenna_role = max(antenna_weights, key=antenna_weights.get) if antenna_weights else "Eyebrows"
        
        # Get learned scaling factors
        e2p = profile.get("energy_to_pitch", 1.0)
        e2y = profile.get("energy_to_yaw", 0.5)
        p2p = profile.get("pitch_to_pitch", 1.0)
        b2a = profile.get("brightness_to_antenna", 1.0)
        
        timeline = []
        
        for i, t in enumerate(time_axis):
            e = energy[min(i, len(energy) - 1)]
            p = pitch[min(i, len(pitch) - 1)]
            b = brightness[min(i, len(brightness) - 1)]
            o = onset[min(i, len(onset) - 1)]
            
            # Apply learned mappings
            head_pitch = (p - 0.5) * 20 * p2p + e * 5 * e2p
            head_yaw = np.sin(t * 2) * 5 + o * 10 * e2y
            head_roll = np.sin(t * 1.5) * 3 * e
            
            # Antenna based on learned role preference
            if antenna_role == "Eyebrows":
                antenna_l = b * 20 * b2a
                antenna_r = b * 20 * b2a
            elif antenna_role == "Arms":
                antenna_l = e * 25 * b2a
                antenna_r = -e * 25 * b2a
            else:
                antenna_l = 0
                antenna_r = 0
            
            # Apply comedian scaling
            timeline.append({
                "t": float(t),
                "pose": {
                    "pitch": float(np.clip(head_pitch * comedian, -25, 25)),
                    "yaw": float(np.clip(head_yaw * comedian, -45, 45)),
                    "roll": float(np.clip(head_roll * comedian, -25, 25)),
                    "base": 0.0,
                    "antenna_l": float(np.clip(antenna_l * comedian, -30, 30)),
                    "antenna_r": float(np.clip(antenna_r * comedian, -30, 30)),
                }
            })
        
        return timeline


class PreferenceLearner:
    """
    Learns from pairwise preferences (approval rankings).
    
    Uses a simple Bradley-Terry model to learn which motion patterns
    are preferred for each emotion.
    """
    
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self._preferences: List[Tuple[str, str, str]] = []  # (winner_id, loser_id, emotion)
        
        os.makedirs(storage_dir, exist_ok=True)
        self._load()
    
    def _path(self) -> str:
        return os.path.join(self.storage_dir, "preferences.json")
    
    def _load(self) -> None:
        path = self._path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._preferences = [tuple(x) for x in json.load(f)]
    
    def _save(self) -> None:
        with open(self._path(), "w", encoding="utf-8") as f:
            json.dump(self._preferences, f)
    
    def add_preference(
        self,
        winner_id: str,
        loser_ids: List[str],
        emotion: str,
    ) -> None:
        """Record preference from approval ranking"""
        for loser_id in loser_ids:
            self._preferences.append((winner_id, loser_id, emotion))
        self._save()
        logger.info(f"Added {len(loser_ids)} preferences for {emotion}")
    
    def get_win_rate(self, emotion: str) -> Dict[str, float]:
        """Get win rates for all candidates in an emotion"""
        wins: Dict[str, int] = {}
        total: Dict[str, int] = {}
        
        for winner, loser, em in self._preferences:
            if em != emotion:
                continue
            wins[winner] = wins.get(winner, 0) + 1
            total[winner] = total.get(winner, 0) + 1
            total[loser] = total.get(loser, 0) + 1
        
        rates = {}
        for eid in total:
            rates[eid] = wins.get(eid, 0) / total[eid]
        
        return rates
