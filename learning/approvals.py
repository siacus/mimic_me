from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from core.storage import StorageManager
from core.profiles import ProfileManager

@dataclass
class Approval:
    episode_id: str
    ranking: List[int]
    antenna_role: Optional[str] = None
    notes: str = ""

class ApprovalManager:
    def __init__(self, storage: StorageManager, profiles: ProfileManager):
        self.storage = storage
        self.profiles = profiles

    def approve(self, profile_id: str, episode_ids: List[str], approval: Approval) -> str:
        if not episode_ids:
            raise ValueError("No candidate episodes provided.")
        if approval.episode_id not in episode_ids:
            raise ValueError("Approved episode_id not among candidates.")

        self.storage.set_episode_approval(
            episode_id=approval.episode_id,
            approved=True,
            approval_rank=approval.ranking,
            antenna_role=approval.antenna_role,
            notes=approval.notes,
        )

        # Incremental mood progress
        ep = None
        for row in self.storage.list_episodes(profile_id=profile_id, limit=500):
            if row["episode_id"] == approval.episode_id:
                ep = row
                break
        if ep:
            self.profiles.update_mood_stats(profile_id=profile_id, emotion=ep["emotion"], approved_delta=1)

        # Keep non-approved candidates as negatives, but mark them not approved
        for eid in episode_ids:
            if eid != approval.episode_id:
                self.storage.set_episode_approval(episode_id=eid, approved=False)

        return approval.episode_id
