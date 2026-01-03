from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
import soundfile as sf

from core.reachy_driver import Pose, TimelinePoint, ReachyMiniDriver


@dataclass
class GeneratedTake:
    episode_id: str
    preview_audio_path: Optional[str]
    timeline: List[TimelinePoint]


class CandidateGenerator:
    """
    Stub candidate generator.

    Updated behavior:
    - Accepts reference_video_path and reference_audio_path, which are stored per-episode
      so your future JEPA/voice model can learn from them.
    - For now it generates placeholder audio + a smooth random motion timeline.
    """

    def __init__(self, storage, reachy: ReachyMiniDriver):
        self.storage = storage
        self.reachy = reachy

    def _make_placeholder_audio(self, out_wav: str, seconds: float = 3.0, sr: int = 22050) -> str:
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        tone = 0.08 * np.sin(2 * np.pi * 440 * t)
        fade = np.minimum(1.0, np.minimum(t / 0.08, (seconds - t) / 0.08))
        y = tone * fade
        sf.write(out_wav, y, sr)
        return out_wav

    def _smooth_random_timeline(self, seconds: float = 3.0, fps: int = 30) -> List[TimelinePoint]:
        n = int(seconds * fps)
        ts = np.linspace(0, seconds, n)
        rng = np.random.default_rng()

        def curve(scale, limit):
            x = rng.normal(0, 1, size=n)
            x = np.cumsum(x)
            x = (x - x.mean()) / (x.std() + 1e-6)
            x = np.clip(scale * x, -limit, limit)
            for _ in range(2):
                x = np.convolve(x, np.ones(7) / 7, mode="same")
            return x

        pitch = curve(8, 18)
        yaw = curve(10, 30)
        roll = curve(6, 15)
        al = curve(6, 20)
        ar = curve(6, 20)

        out: List[TimelinePoint] = []
        for i, t in enumerate(ts):
            out.append(
                TimelinePoint(
                    t=float(t),
                    pose=Pose(
                        pitch=float(pitch[i]),
                        roll=float(roll[i]),
                        yaw=float(yaw[i]),
                        base=0.0,
                        antenna_l=float(al[i]),
                        antenna_r=float(ar[i]),
                    ),
                )
            )
        return out

    def _write_conditioning(self, assets_dir: str, conditioning: Dict[str, Any]) -> None:
        path = os.path.join(assets_dir, "conditioning.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(conditioning, f, indent=2)

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
    ) -> List[GeneratedTake]:
        takes: List[GeneratedTake] = []
        n_candidates = max(1, min(int(n_candidates), 4))

        for _ in range(n_candidates):
            episode_id = self.storage.create_episode(
                profile_id=profile_id,
                source_type=source_type,
                emotion=emotion,
                text=text,
                voice_mode=voice_mode,
                comedian=comedian,
                approved=False,
                notes="candidate_take",
            )

            assets_dir = os.path.join(self.storage.paths.episodes_dir, episode_id)
            os.makedirs(assets_dir, exist_ok=True)

            conditioning = {
                "reference_video_path": reference_video_path,
                "reference_audio_path": reference_audio_path,
                "voice_mode": voice_mode,
                "emotion": emotion,
                "text": text,
                "comedian": float(comedian),
            }
            self._write_conditioning(assets_dir, conditioning)

            wav_path = os.path.join(assets_dir, "preview.wav")
            self._make_placeholder_audio(wav_path, seconds=3.0)

            timeline = self._smooth_random_timeline(seconds=3.0, fps=30)
            timeline = self.reachy.exaggerate_timeline(timeline, comedian=comedian)

            motion_path = os.path.join(assets_dir, "timeline.json")
            with open(motion_path, "w", encoding="utf-8") as f:
                json.dump([{"t": pt.t, "pose": pt.pose.__dict__} for pt in timeline], f, indent=2)

            takes.append(GeneratedTake(episode_id=episode_id, preview_audio_path=wav_path, timeline=timeline))

        return takes
