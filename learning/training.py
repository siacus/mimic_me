"""
Incremental training for Reachy Mimic Me.

Learns from approved examples to improve:
- Audio-motion synchronization
- Per-emotion motion patterns
- Preference rankings
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.storage import StorageManager

logger = logging.getLogger(__name__)


class Trainer:
    """
    Incremental trainer for audio-motion sync and preferences.
    
    Key features:
    - Incremental: add examples without full retraining
    - Per-emotion profiles: learn different patterns per mood
    - Preference learning: incorporate approval rankings
    - Replay buffer: avoid forgetting old examples
    """
    
    def __init__(self, storage: StorageManager, models_dir: Optional[str] = None):
        self.storage = storage
        self.models_dir = models_dir or os.path.join(storage.paths.root, "models")
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Lazy-loaded learners
        self._sync_learner = None
        self._preference_learner = None
    
    def _get_sync_learner(self):
        """Lazy-load sync learner"""
        if self._sync_learner is None:
            try:
                from ml.sync import SyncLearner
                sync_dir = os.path.join(self.models_dir, "sync")
                self._sync_learner = SyncLearner(sync_dir)
            except Exception as e:
                logger.warning(f"Failed to initialize sync learner: {e}")
        return self._sync_learner
    
    def _get_preference_learner(self):
        """Lazy-load preference learner"""
        if self._preference_learner is None:
            try:
                from ml.sync import PreferenceLearner
                pref_dir = os.path.join(self.models_dir, "preferences")
                self._preference_learner = PreferenceLearner(pref_dir)
            except Exception as e:
                logger.warning(f"Failed to initialize preference learner: {e}")
        return self._preference_learner
    
    def _load_episode_data(self, episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load all data for an episode"""
        episode_id = episode["episode_id"]
        assets_dir = episode.get("assets_dir") or os.path.join(
            self.storage.paths.episodes_dir, episode_id
        )
        
        data = {"episode": episode}
        
        # Load conditioning
        conditioning_path = os.path.join(assets_dir, "conditioning.json")
        if os.path.exists(conditioning_path):
            with open(conditioning_path, "r") as f:
                data["conditioning"] = json.load(f)
        
        # Load audio features (if extracted)
        features_path = os.path.join(assets_dir, "audio_features.json")
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                data["audio_features"] = json.load(f)
        else:
            # Try to extract features from audio
            audio_path = os.path.join(assets_dir, "preview.wav")
            if os.path.exists(audio_path):
                data["audio_features"] = self._extract_audio_features(audio_path, features_path)
        
        # Load timeline
        timeline_path = os.path.join(assets_dir, "timeline.json")
        if os.path.exists(timeline_path):
            with open(timeline_path, "r") as f:
                data["timeline"] = json.load(f)
        
        # Load reference motion (if extracted)
        ref_motion_path = os.path.join(assets_dir, "reference_motion.json")
        if os.path.exists(ref_motion_path):
            with open(ref_motion_path, "r") as f:
                data["reference_motion"] = json.load(f)
        
        return data
    
    def _extract_audio_features(self, audio_path: str, output_path: str) -> Optional[Dict[str, Any]]:
        """Extract audio features for training"""
        try:
            from ml.motion import MotionGenerator
            import numpy as np
            
            generator = MotionGenerator()
            features = generator.extract_audio_features(audio_path)
            
            # Convert numpy arrays to lists for JSON serialization
            features_json = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in features.items()
            }
            
            with open(output_path, "w") as f:
                json.dump(features_json, f)
            
            return features_json
            
        except Exception as e:
            logger.warning(f"Failed to extract audio features: {e}")
            return None
    
    def add_approved_example(
        self,
        episode_id: str,
        profile_id: str,
        emotion: str,
        ranking: List[int],
        antenna_role: str,
    ) -> Dict[str, Any]:
        """
        Add an approved example to the training data.
        
        Called automatically when user approves a take.
        """
        import numpy as np
        
        result = {"episode_id": episode_id, "added": False}
        
        # Load episode data
        episodes = self.storage.list_episodes(profile_id, limit=1000)
        episode = next((e for e in episodes if e["episode_id"] == episode_id), None)
        
        if not episode:
            result["error"] = "Episode not found"
            return result
        
        data = self._load_episode_data(episode)
        if not data:
            result["error"] = "Failed to load episode data"
            return result
        
        # Add to sync learner
        sync_learner = self._get_sync_learner()
        if sync_learner and data.get("audio_features") and data.get("timeline"):
            try:
                # Convert lists back to numpy arrays
                audio_features = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in data["audio_features"].items()
                }
                
                sync_learner.add_example(
                    episode_id=episode_id,
                    emotion=emotion,
                    audio_features=audio_features,
                    motion_timeline=data["timeline"],
                    antenna_role=antenna_role,
                    approval_rank=0,  # This was the approved one
                )
                result["sync_example_added"] = True
            except Exception as e:
                logger.warning(f"Failed to add sync example: {e}")
        
        # Add to preference learner
        pref_learner = self._get_preference_learner()
        if pref_learner and len(ranking) > 1:
            try:
                # Get loser episode IDs (the ones not approved)
                loser_indices = [i for i in ranking[1:] if i != ranking[0]]
                # We need the episode IDs, not indices - this is handled in approvals.py
                pref_learner.add_preference(
                    winner_id=episode_id,
                    loser_ids=[],  # Will be populated by approvals manager
                    emotion=emotion,
                )
                result["preference_added"] = True
            except Exception as e:
                logger.warning(f"Failed to add preference: {e}")
        
        result["added"] = True
        return result
    
    def update_profile_models(self, profile_id: str) -> Dict[str, Any]:
        """
        Update all models for a profile.
        
        This runs the actual learning step:
        1. Gather all approved examples for this profile
        2. Train sync model
        3. Update preference rankings
        4. Increment model version
        """
        import numpy as np
        
        result = {
            "profile_id": profile_id,
            "timestamp": datetime.utcnow().isoformat(),
            "episodes_processed": 0,
            "sync_training": None,
            "preference_stats": None,
        }
        
        # Get profile
        prof = self.storage.get_profile(profile_id)
        if not prof:
            result["error"] = "Unknown profile"
            return result
        
        # Get all episodes
        episodes = self.storage.list_episodes(profile_id, limit=10000)
        approved = [e for e in episodes if int(e.get("approved", 0)) == 1]
        
        result["episodes_total"] = len(episodes)
        result["episodes_approved"] = len(approved)
        
        # Statistics by emotion
        by_emotion = {}
        for e in episodes:
            em = e["emotion"]
            by_emotion.setdefault(em, {"total": 0, "approved": 0})
            by_emotion[em]["total"] += 1
            by_emotion[em]["approved"] += int(e.get("approved", 0))
        result["by_emotion"] = by_emotion
        
        # Load and process approved examples
        sync_learner = self._get_sync_learner()
        processed = 0
        
        for episode in approved:
            data = self._load_episode_data(episode)
            if not data:
                continue
            
            # Add to sync learner if not already added
            if sync_learner and data.get("audio_features") and data.get("timeline"):
                try:
                    audio_features = {
                        k: np.array(v) if isinstance(v, list) else v
                        for k, v in data["audio_features"].items()
                    }
                    
                    # Get approval rank from episode
                    approval_rank_json = episode.get("approval_rank")
                    approval_rank = 0
                    if approval_rank_json:
                        try:
                            ranks = json.loads(approval_rank_json)
                            approval_rank = ranks[0] if ranks else 0
                        except:
                            pass
                    
                    sync_learner.add_example(
                        episode_id=episode["episode_id"],
                        emotion=episode["emotion"],
                        audio_features=audio_features,
                        motion_timeline=data["timeline"],
                        antenna_role=episode.get("antenna_role", "Eyebrows"),
                        approval_rank=approval_rank,
                    )
                    processed += 1
                except Exception as e:
                    logger.debug(f"Failed to add example {episode['episode_id'][:8]}: {e}")
        
        result["episodes_processed"] = processed
        
        # Run sync learning
        if sync_learner and processed > 0:
            try:
                sync_stats = sync_learner.learn()
                result["sync_training"] = sync_stats
                logger.info(f"Sync learning complete: {sync_stats}")
            except Exception as e:
                logger.error(f"Sync learning failed: {e}")
                result["sync_training_error"] = str(e)
        
        # Get preference stats
        pref_learner = self._get_preference_learner()
        if pref_learner:
            try:
                pref_stats = {}
                for em in by_emotion.keys():
                    rates = pref_learner.get_win_rate(em)
                    if rates:
                        pref_stats[em] = {
                            "n_comparisons": len(rates),
                            "top_win_rate": max(rates.values()) if rates else 0,
                        }
                result["preference_stats"] = pref_stats
            except Exception as e:
                logger.warning(f"Failed to get preference stats: {e}")
        
        # Update model version
        model_version = int(prof.get("model_version", 0)) + 1
        self.storage.update_profile(profile_id, model_version=model_version)
        result["model_version"] = model_version
        
        # Save training record
        training_log_path = os.path.join(
            self.models_dir, 
            f"training_log_{profile_id[:8]}.jsonl"
        )
        with open(training_log_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        return result
    
    def get_training_status(self, profile_id: str) -> Dict[str, Any]:
        """Get training status for a profile"""
        prof = self.storage.get_profile(profile_id)
        if not prof:
            return {"error": "Unknown profile"}
        
        episodes = self.storage.list_episodes(profile_id, limit=10000)
        
        status = {
            "profile_id": profile_id,
            "model_version": prof.get("model_version", 0),
            "total_episodes": len(episodes),
            "approved_episodes": sum(1 for e in episodes if int(e.get("approved", 0)) == 1),
        }
        
        # Check sync model status
        sync_learner = self._get_sync_learner()
        if sync_learner:
            status["sync_model_version"] = sync_learner.model.version
            status["sync_emotions"] = list(sync_learner.model.emotion_profiles.keys())
        
        return status
