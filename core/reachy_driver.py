from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Pose:
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    base: float = 0.0
    antenna_l: float = 0.0
    antenna_r: float = 0.0

@dataclass
class TimelinePoint:
    t: float
    pose: Pose

class ReachyMiniDriver:
    """
    Safe wrapper for Reachy Mini Lite.
    Hardware is off by default; enable via UI.
    """
    LIMITS = {
        "pitch": (-25.0, 25.0),
        "roll": (-25.0, 25.0),
        "yaw": (-45.0, 45.0),
        "base": (0.0, 360.0),
        "antenna": (-30.0, 30.0),
    }

    def __init__(self, enable_hardware: bool = False):
        self.enable_hardware = enable_hardware
        self._robot = None

    def connect(self) -> None:
        if not self.enable_hardware:
            logger.info("Hardware disabled: preview-only mode.")
            return
        try:
            from reachy_mini import ReachyMini  # type: ignore
            self._robot = ReachyMini()
            try:
                self._robot.camera.start_streaming()
            except Exception:
                pass
            logger.info("Connected to Reachy Mini.")
        except Exception as e:
            logger.exception("Failed to connect: %s", e)
            self._robot = None
            self.enable_hardware = False

    def is_connected(self) -> bool:
        return self._robot is not None

    def get_frame(self) -> Optional[np.ndarray]:
        if not self._robot:
            return None
        try:
            return self._robot.camera.get_frame()
        except Exception:
            return None

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, v)))

    def clamp_pose(self, p: Pose) -> Pose:
        p.pitch = self._clamp(p.pitch, *self.LIMITS["pitch"])
        p.roll  = self._clamp(p.roll,  *self.LIMITS["roll"])
        p.yaw   = self._clamp(p.yaw,   *self.LIMITS["yaw"])
        p.base  = float(p.base % 360.0)
        p.antenna_l = self._clamp(p.antenna_l, *self.LIMITS["antenna"])
        p.antenna_r = self._clamp(p.antenna_r, *self.LIMITS["antenna"])
        return p

    def apply_pose(self, p: Pose) -> None:
        p = self.clamp_pose(p)
        if not self._robot:
            return
        # Adjust these calls if your SDK differs.
        try:
            self._robot.head.rotate(pitch=p.pitch, roll=p.roll, yaw=p.yaw)
        except Exception:
            pass
        try:
            self._robot.base.turn_to(angle=p.base)
        except Exception:
            pass
        # Antenna direct control may differ; stub for now.

    def play_timeline(self, timeline: List[TimelinePoint], realtime: bool = True) -> None:
        if not timeline:
            return
        t0 = time.time()
        for pt in timeline:
            self.apply_pose(pt.pose)
            if realtime:
                while time.time() - t0 < pt.t:
                    time.sleep(0.002)

    @staticmethod
    def exaggerate_timeline(timeline: List[TimelinePoint], comedian: float) -> List[TimelinePoint]:
        comedian = float(comedian)
        if comedian <= 0:
            comedian = 1.0
        neutral = Pose()
        out: List[TimelinePoint] = []
        for pt in timeline:
            p = pt.pose
            p2 = Pose(
                pitch=neutral.pitch + comedian * (p.pitch - neutral.pitch),
                roll=neutral.roll + comedian * (p.roll - neutral.roll),
                yaw=neutral.yaw + comedian * (p.yaw - neutral.yaw),
                base=neutral.base + comedian * (p.base - neutral.base),
                antenna_l=neutral.antenna_l + comedian * (p.antenna_l - neutral.antenna_l),
                antenna_r=neutral.antenna_r + comedian * (p.antenna_r - neutral.antenna_r),
            )
            out.append(TimelinePoint(t=pt.t, pose=p2))
        return out
