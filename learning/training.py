from __future__ import annotations
from typing import Dict, Any
from core.storage import StorageManager


class Trainer:
    """
    Incremental trainer scaffold.
    Intended behavior:
    - Teach 1 mood first, add more later.
    - Avoid forgetting via replay of previously approved episodes.
    - Update only lightweight per-profile/per-mood parameters (embeddings/adapters).
    Current stub increments model_version and reports stats.
    """
    def __init__(self, storage: StorageManager):
        self.storage = storage
    
    def update_profile_models(self, profile_id: str) -> Dict[str, Any]:
        prof = self.storage.get_profile(profile_id)
        if not prof:
            raise ValueError("Unknown profile")
        
        episodes = self.storage.list_episodes(profile_id=profile_id, limit=10000)
        n_approved = sum(1 for e in episodes if int(e["approved"]) == 1)
        
        by_em = {}
        for e in episodes:
            by_em.setdefault(e["emotion"], {"total": 0, "approved": 0})
            by_em[e["emotion"]]["total"] += 1
            by_em[e["emotion"]]["approved"] += int(e["approved"])
        
        model_version = int(prof["model_version"]) + 1
        self.storage.update_profile(profile_id, model_version=model_version)
        
        return {
            "profile_id": profile_id,
            "model_version": model_version,
            "episodes_total": len(episodes),
            "episodes_approved": n_approved,
            "by_emotion": by_em,
            "note": "Stub update only. Plug your JEPA/gesture/voice training here.",
        }
