from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

@dataclass
class CandidateTake:
    candidate_id: str
    episode_id: str
    emotion: str
    text: str
    voice_mode: str
    comedian: float
    preview_audio_path: Optional[str] = None
    notes: str = ""

@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    current_profile_id: Optional[str] = None
    candidates: List[CandidateTake] = field(default_factory=list)

@dataclass
class AppState:
    sessions: Dict[str, SessionState] = field(default_factory=dict)

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]
