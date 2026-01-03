from __future__ import annotations
import os
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

@dataclass
class StoragePaths:
    root: str

    @property
    def data_dir(self) -> str:
        return os.path.join(self.root, "data")

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "app.db")

    @property
    def profiles_dir(self) -> str:
        return os.path.join(self.data_dir, "profiles")

    @property
    def episodes_dir(self) -> str:
        return os.path.join(self.data_dir, "episodes")

    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.data_dir, "uploads")

class StorageManager:
    """
    Robust local storage:
    - SQLite for metadata + indexes
    - filesystem directories for large assets
    Thread-safe via an internal lock.
    """
    def __init__(self, root: str):
        self.paths = StoragePaths(root=root)
        os.makedirs(self.paths.data_dir, exist_ok=True)
        os.makedirs(self.paths.profiles_dir, exist_ok=True)
        os.makedirs(self.paths.episodes_dir, exist_ok=True)
        os.makedirs(self.paths.uploads_dir, exist_ok=True)

        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.paths.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consent_voice INTEGER NOT NULL DEFAULT 0,
                default_voice_mode TEXT NOT NULL DEFAULT 'synthetic',
                model_version INTEGER NOT NULL DEFAULT 0,
                mirror_roi_front TEXT,
                mirror_roi_side TEXT
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                emotion TEXT NOT NULL,
                text TEXT NOT NULL,
                voice_mode TEXT NOT NULL,
                comedian REAL NOT NULL DEFAULT 1.0,
                approved INTEGER NOT NULL DEFAULT 0,
                approval_rank TEXT,
                antenna_role TEXT,
                notes TEXT,
                assets_dir TEXT NOT NULL,
                FOREIGN KEY(profile_id) REFERENCES profiles(profile_id)
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_profile ON episodes(profile_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_profile_emotion ON episodes(profile_id, emotion);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_approved ON episodes(approved);")

    # Profiles
    def create_profile(self, name: str, consent_voice: bool = False, default_voice_mode: str = "synthetic") -> str:
        profile_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        assets_dir = os.path.join(self.paths.profiles_dir, profile_id)
        os.makedirs(assets_dir, exist_ok=True)

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO profiles(profile_id,name,created_at,consent_voice,default_voice_mode) VALUES(?,?,?,?,?)",
                (profile_id, name, now, 1 if consent_voice else 0, default_voice_mode),
            )

        profile_json = {
            "profile_id": profile_id,
            "name": name,
            "consent_voice": bool(consent_voice),
            "default_voice_mode": default_voice_mode,
            "created_at": now,
            "moods": {},
        }
        with open(os.path.join(assets_dir, "profile.json"), "w", encoding="utf-8") as f:
            json.dump(profile_json, f, indent=2)

        return profile_id

    def list_profiles(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM profiles ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE profile_id=?", (profile_id,)).fetchone()
        return dict(row) if row else None

    def update_profile(self, profile_id: str, **fields) -> None:
        allowed = {"name","consent_voice","default_voice_mode","model_version","mirror_roi_front","mirror_roi_side"}
        keys = [k for k in fields.keys() if k in allowed]
        if not keys:
            return
        sets = ", ".join([f"{k}=?" for k in keys])
        vals = [fields[k] for k in keys] + [profile_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE profiles SET {sets} WHERE profile_id=?", vals)

    def read_profile_json(self, profile_id: str) -> Dict[str, Any]:
        path = os.path.join(self.paths.profiles_dir, profile_id, "profile.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_profile_json(self, profile_id: str, data: Dict[str, Any]) -> None:
        path = os.path.join(self.paths.profiles_dir, profile_id, "profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Episodes
    def new_episode_dir(self, episode_id: str) -> str:
        d = os.path.join(self.paths.episodes_dir, episode_id)
        os.makedirs(d, exist_ok=True)
        return d

    def create_episode(
        self,
        profile_id: str,
        source_type: str,
        emotion: str,
        text: str,
        voice_mode: str,
        comedian: float = 1.0,
        approved: bool = False,
        assets_dir: Optional[str] = None,
        notes: str = "",
    ) -> str:
        episode_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        assets_dir = assets_dir or self.new_episode_dir(episode_id)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO episodes(episode_id,profile_id,created_at,source_type,emotion,text,voice_mode,comedian,approved,assets_dir,notes)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (episode_id, profile_id, now, source_type, emotion, text, voice_mode, float(comedian), 1 if approved else 0, assets_dir, notes),
            )
        return episode_id

    def set_episode_approval(
        self,
        episode_id: str,
        approved: bool,
        approval_rank: Optional[List[int]] = None,
        antenna_role: Optional[str] = None,
        notes: str = "",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE episodes SET approved=?, approval_rank=?, antenna_role=?, notes=? WHERE episode_id=?",
                (
                    1 if approved else 0,
                    json.dumps(approval_rank) if approval_rank is not None else None,
                    antenna_role,
                    notes,
                    episode_id,
                ),
            )

    def list_episodes(self, profile_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE profile_id=? ORDER BY created_at DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
