from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import re
from .storage import StorageManager
from .constants import STANDARD_EMOTIONS


def normalize_custom_mood(word: str) -> str:
    w = (word or "").strip()
    if not w:
        return "neutral"
    w = re.split(r"\s+", w)[0]
    w = re.sub(r"[^A-Za-z0-9_\-]", "", w)
    return w or "neutral"


def normalize_emotion(choice: str, custom_word: str) -> str:
    if choice in STANDARD_EMOTIONS:
        return choice
    if choice == "Custom":
        return f"Custom:{normalize_custom_mood(custom_word)}"
    return "Custom:neutral"


@dataclass
class ProfileSummary:
    profile_id: str
    name: str
    consent_voice: bool
    default_voice_mode: str
    model_version: int


class ProfileManager:
    def __init__(self, storage: StorageManager):
        self.storage = storage
    
    def list_profiles(self) -> List[ProfileSummary]:
        out: List[ProfileSummary] = []
        for p in self.storage.list_profiles():
            out.append(ProfileSummary(
                profile_id=p["profile_id"],
                name=p["name"],
                consent_voice=bool(p["consent_voice"]),
                default_voice_mode=p["default_voice_mode"],
                model_version=int(p["model_version"]),
            ))
        return out
    
    def create_profile(self, name: str, consent_voice: bool) -> str:
        default_voice_mode = "imitate" if consent_voice else "synthetic"
        return self.storage.create_profile(name=name, consent_voice=consent_voice, default_voice_mode=default_voice_mode)
    
    def update_mood_stats(self, profile_id: str, emotion: str, approved_delta: int = 0) -> Dict[str, Any]:
        pj = self.storage.read_profile_json(profile_id) or {}
        moods = pj.setdefault("moods", {})
        m = moods.setdefault(emotion, {"approved": 0})
        m["approved"] = int(m.get("approved", 0)) + int(approved_delta)
        self.storage.write_profile_json(profile_id, pj)
        return pj
    
    def mood_progress_table(self, profile_id: str):
        pj = self.storage.read_profile_json(profile_id) or {}
        moods = pj.get("moods", {})
        keys = list(STANDARD_EMOTIONS)
        for k in moods.keys():
            if k.startswith("Custom:") and k not in keys:
                keys.append(k)
        return [{"mood": k, "approved": int(moods.get(k, {}).get("approved", 0))} for k in keys]
    
    def resolve_voice_mode(self, profile_id: str, requested_mode: str) -> str:
        prof = self.storage.get_profile(profile_id)
        if not prof:
            return "synthetic"
        
        consent = bool(prof["consent_voice"])
        default_mode = prof["default_voice_mode"]
        mode = (requested_mode or "auto").lower()
        
        if mode == "auto":
            mode = default_mode
        
        if mode == "imitate" and not consent:
            return "synthetic"
        
        if mode not in {"synthetic", "imitate"}:
            return "synthetic"
        
        return mode
    
