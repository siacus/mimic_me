"""
Motion extraction from video and motion generation.

Extracts head pose (pitch, yaw, roll) and facial landmarks from video
using MediaPipe, then learns to generate motion from audio features.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeadPose:
    """Head pose at a single timestamp"""
    t: float
    pitch: float  # Up/down nod
    yaw: float    # Left/right turn
    roll: float   # Tilt


@dataclass
class FacialFeatures:
    """Extracted facial features at a single timestamp"""
    t: float
    mouth_open: float      # 0-1, how open the mouth is
    eyebrow_raise: float   # 0-1, eyebrow position
    eye_openness: float    # 0-1, eye openness
    smile: float           # 0-1, smile intensity


@dataclass
class MotionSequence:
    """Complete motion sequence extracted from video"""
    duration: float
    fps: float
    head_poses: List[HeadPose] = field(default_factory=list)
    facial_features: List[FacialFeatures] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "fps": self.fps,
            "head_poses": [
                {"t": p.t, "pitch": p.pitch, "yaw": p.yaw, "roll": p.roll}
                for p in self.head_poses
            ],
            "facial_features": [
                {
                    "t": f.t,
                    "mouth_open": f.mouth_open,
                    "eyebrow_raise": f.eyebrow_raise,
                    "eye_openness": f.eye_openness,
                    "smile": f.smile,
                }
                for f in self.facial_features
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotionSequence":
        seq = cls(
            duration=data["duration"],
            fps=data["fps"],
        )
        seq.head_poses = [
            HeadPose(t=p["t"], pitch=p["pitch"], yaw=p["yaw"], roll=p["roll"])
            for p in data.get("head_poses", [])
        ]
        seq.facial_features = [
            FacialFeatures(
                t=f["t"],
                mouth_open=f["mouth_open"],
                eyebrow_raise=f["eyebrow_raise"],
                eye_openness=f["eye_openness"],
                smile=f["smile"],
            )
            for f in data.get("facial_features", [])
        ]
        return seq


class MotionExtractor:
    """
    Extracts motion from video using MediaPipe.
    
    Outputs head pose and facial features that can be mapped
    to Reachy robot movements.
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        face_landmarker_model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Args:
            use_gpu: Kept for backward compatibility. (MediaPipe Tasks handles device selection internally.)
            face_landmarker_model_path: Optional path to a MediaPipe Tasks face landmarker `.task` model.
                If not provided, the extractor will look for:
                  - env var MEDIAPIPE_FACE_LANDMARKER_MODEL
                  - models/mediapipe/face_landmarker.task (relative to CWD)
                  - <repo_root>/models/mediapipe/face_landmarker.task (relative to this file)
            min_detection_confidence: Minimum confidence for face detection (best-effort mapping).
            min_tracking_confidence: Minimum confidence for tracking (best-effort mapping).
        """
        self.use_gpu = use_gpu
        self.face_landmarker_model_path = face_landmarker_model_path
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)

        # Legacy Solutions backend
        self._face_mesh = None
        self._pose = None

        # Tasks backend
        self._tasks_backend = None  # None | "tasks" | "solutions"
        self._face_landmarker_cls = None
        self._face_landmarker_options = None
        
    def _resolve_face_landmarker_model_path(self) -> Optional[str]:
        """Find a usable `.task` model path for the MediaPipe Tasks FaceLandmarker."""
        # 1) explicit arg
        if self.face_landmarker_model_path:
            return self.face_landmarker_model_path

        # 2) env var
        env_p = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
        if env_p:
            return env_p

        # 3) common repo-relative locations
        candidates = [
            os.path.join("models", "mediapipe", "face_landmarker.task"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mediapipe", "face_landmarker.task"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]  # return the default even if missing, for a useful error

    def _init_mediapipe(self):
        """Lazy initialization of MediaPipe.

        Newer MediaPipe (>=0.10.31) removes `mediapipe.solutions` and requires MediaPipe Tasks.
        We support both backends:
          - Tasks FaceLandmarker (preferred)
          - Legacy Solutions FaceMesh (if available)
        """
        if self._tasks_backend is not None:
            return

        try:
            import mediapipe as mp
        except ImportError:
            logger.error("MediaPipe not installed. Run: pip install mediapipe")
            raise

        self._mp = mp

        # Prefer Tasks if available
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            model_path = self._resolve_face_landmarker_model_path()
            try:
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                # Note: options surface changes across releases; keep the required params minimal.
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.VIDEO,
                )

                # Stash class/options; we create the landmarker per-video via context manager.
                self._face_landmarker_cls = FaceLandmarker
                self._face_landmarker_options = options
                self._tasks_backend = "tasks"
                logger.info("MediaPipe Tasks FaceLandmarker configured")
                return
            except Exception as e:
                # If tasks exists but model path/options fail, fall back to solutions if available.
                logger.warning(f"Failed to configure MediaPipe Tasks FaceLandmarker: {e}")

        # Legacy Solutions (only if present)
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._tasks_backend = "solutions"
            logger.info("MediaPipe Solutions FaceMesh initialized")
            return

        # No usable backend
        self._tasks_backend = "none"
        logger.error(
            "MediaPipe backend unavailable. Newer mediapipe wheels may not include `mediapipe.solutions`. "
            "Install a recent mediapipe and provide a FaceLandmarker `.task` model at models/mediapipe/face_landmarker.task "
            "or via MEDIAPIPE_FACE_LANDMARKER_MODEL."
        )
    
    def _estimate_head_pose(
        self,
        landmarks: Any,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (pitch, yaw, roll) from face landmarks.
        
        Uses a simplified approach based on key facial landmarks.
        """
        # Key landmark indices for pose estimation
        # Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
        indices = [1, 152, 33, 263, 61, 291]
        
        # Get 2D points
        points_2d = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            points_2d.append([lm.x * image_width, lm.y * image_height])
        points_2d = np.array(points_2d, dtype=np.float64)
        
        # 3D model points (generic face model)
        points_3d = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye corner
            [225.0, 170.0, -135.0],    # Right eye corner
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0, -150.0, -125.0],   # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (approximate)
        focal_length = image_width
        center = (image_width / 2, image_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        try:
            import cv2
            success, rotation_vec, translation_vec = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return 0.0, 0.0, 0.0
            
            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            pitch = float(euler_angles[0])
            yaw = float(euler_angles[1])
            roll = float(euler_angles[2])
            
            # Clamp to reasonable values
            pitch = np.clip(pitch, -45, 45)
            yaw = np.clip(yaw, -60, 60)
            roll = np.clip(roll, -30, 30)
            
            return pitch, yaw, roll
            
        except Exception as e:
            logger.debug(f"Head pose estimation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _extract_facial_features(self, landmarks: Any) -> Dict[str, float]:
        """Extract facial features from landmarks"""
        
        def _distance(idx1: int, idx2: int) -> float:
            lm1 = landmarks.landmark[idx1]
            lm2 = landmarks.landmark[idx2]
            return np.sqrt(
                (lm1.x - lm2.x) ** 2 +
                (lm1.y - lm2.y) ** 2 +
                (lm1.z - lm2.z) ** 2
            )
        
        # Mouth openness: distance between upper and lower lip
        # Landmarks: 13 (upper lip), 14 (lower lip)
        mouth_open_dist = _distance(13, 14)
        mouth_width = _distance(61, 291)  # Mouth corners
        mouth_open = min(1.0, mouth_open_dist / (mouth_width * 0.5 + 0.01))
        
        # Eyebrow raise: distance from eyebrow to eye
        # Left eyebrow center (107) to left eye top (159)
        eyebrow_dist = _distance(107, 159)
        eyebrow_raise = min(1.0, max(0.0, (eyebrow_dist - 0.02) * 20))
        
        # Eye openness: distance between upper and lower eyelid
        # Left eye: 159 (top), 145 (bottom)
        eye_dist = _distance(159, 145)
        eye_openness = min(1.0, max(0.0, eye_dist * 30))
        
        # Smile: mouth corner distance and angle
        left_corner = landmarks.landmark[61]
        right_corner = landmarks.landmark[291]
        mouth_center = landmarks.landmark[13]
        
        # Smile pulls corners up relative to center
        corner_height = (left_corner.y + right_corner.y) / 2
        smile = min(1.0, max(0.0, (mouth_center.y - corner_height) * 20))
        
        return {
            "mouth_open": float(mouth_open),
            "eyebrow_raise": float(eyebrow_raise),
            "eye_openness": float(eye_openness),
            "smile": float(smile),
        }
    
    def extract_from_video(
        self,
        video_path: str,
        target_fps: float = 30.0,
    ) -> Optional[MotionSequence]:
        """
        Extract motion sequence from video file.
        
        Args:
            video_path: Path to video file
            target_fps: Target frame rate for extraction
            
        Returns:
            MotionSequence or None if extraction fails
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed. Run: pip install opencv-python")
            return None
        
        self._init_mediapipe()

        if getattr(self, "_tasks_backend", None) in ("none", None):
            logger.error("MotionExtractor: no usable MediaPipe backend configured")
            return None
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / video_fps
        
        # Calculate frame skip for target fps
        frame_skip = max(1, int(video_fps / target_fps))
        
        logger.info(f"Extracting motion from {video_path}")
        logger.info(f"  Duration: {duration:.2f}s, FPS: {video_fps}, Frames: {frame_count}")
        
        sequence = MotionSequence(duration=duration, fps=target_fps)
        
        def _append_neutral(t: float):
            sequence.head_poses.append(HeadPose(t=t, pitch=0, yaw=0, roll=0))
            sequence.facial_features.append(
                FacialFeatures(t=t, mouth_open=0, eyebrow_raise=0.5, eye_openness=0.8, smile=0.3)
            )

        # --- Backend: MediaPipe Solutions (legacy) ---
        if self._tasks_backend == "solutions":
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                t = frame_idx / video_fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame.shape[:2]

                results = self._face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    pitch, yaw, roll = self._estimate_head_pose(landmarks, width, height)
                    sequence.head_poses.append(HeadPose(t=t, pitch=pitch, yaw=yaw, roll=roll))
                    features = self._extract_facial_features(landmarks)
                    sequence.facial_features.append(
                        FacialFeatures(
                            t=t,
                            mouth_open=features["mouth_open"],
                            eyebrow_raise=features["eyebrow_raise"],
                            eye_openness=features["eye_openness"],
                            smile=features["smile"],
                        )
                    )
                else:
                    _append_neutral(t)

                frame_idx += 1

        # --- Backend: MediaPipe Tasks FaceLandmarker (preferred) ---
        elif self._tasks_backend == "tasks":
            model_path = self._resolve_face_landmarker_model_path()
            if not os.path.exists(model_path):
                cap.release()
                logger.error(
                    "FaceLandmarker model not found at %s. Download a FaceLandmarker `.task` model and place it there, "
                    "or set MEDIAPIPE_FACE_LANDMARKER_MODEL.",
                    model_path,
                )
                return None

            # Build landmarker per video using context manager (recommended by MediaPipe docs).
            with self._face_landmarker_cls.create_from_options(self._face_landmarker_options) as landmarker:
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip != 0:
                        frame_idx += 1
                        continue

                    t = frame_idx / video_fps
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width = frame.shape[:2]

                    # MediaPipe Tasks expects an mp.Image
                    mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
                    ts_ms = int(t * 1000)
                    try:
                        result = landmarker.detect_for_video(mp_image, ts_ms)
                    except Exception as e:
                        logger.debug(f"FaceLandmarker detect_for_video failed at t={t:.3f}s: {e}")
                        _append_neutral(t)
                        frame_idx += 1
                        continue

                    if getattr(result, "face_landmarks", None) and len(result.face_landmarks) > 0:
                        # result.face_landmarks[0] is a list of NormalizedLandmark
                        class _LMWrap:
                            def __init__(self, lms):
                                self.landmark = lms

                        landmarks = _LMWrap(result.face_landmarks[0])
                        pitch, yaw, roll = self._estimate_head_pose(landmarks, width, height)
                        sequence.head_poses.append(HeadPose(t=t, pitch=pitch, yaw=yaw, roll=roll))
                        features = self._extract_facial_features(landmarks)
                        sequence.facial_features.append(
                            FacialFeatures(
                                t=t,
                                mouth_open=features["mouth_open"],
                                eyebrow_raise=features["eyebrow_raise"],
                                eye_openness=features["eye_openness"],
                                smile=features["smile"],
                            )
                        )
                    else:
                        _append_neutral(t)

                    frame_idx += 1
        
        cap.release()
        
        logger.info(f"  Extracted {len(sequence.head_poses)} pose frames")
        return sequence
    
    def save_sequence(self, sequence: MotionSequence, output_path: str) -> None:
        """Save motion sequence to JSON file"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sequence.to_dict(), f, indent=2)
        logger.info(f"Saved motion sequence to {output_path}")
    
    def load_sequence(self, input_path: str) -> Optional[MotionSequence]:
        """Load motion sequence from JSON file"""
        if not os.path.exists(input_path):
            return None
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MotionSequence.from_dict(data)


class MotionGenerator:
    """
    Generates Reachy robot motion from audio features.
    
    Two modes:
    1. Transfer mode: Apply extracted motion from video to robot
    2. Synthesis mode: Generate motion from audio features
    """
    
    def __init__(self):
        self._audio_analyzer = None
    
    def _init_audio_analyzer(self):
        """Initialize audio analysis"""
        if self._audio_analyzer is not None:
            return
            
        try:
            import librosa
            self._librosa = librosa
            self._audio_analyzer = True
            logger.info("Audio analyzer (librosa) initialized")
        except ImportError:
            logger.error("librosa not installed. Run: pip install librosa")
            raise
    
    def extract_audio_features(
        self,
        audio_path: str,
        target_fps: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        """
        Extract audio features for motion generation.
        
        Features extracted:
        - RMS energy (overall loudness)
        - Pitch (fundamental frequency)
        - Spectral centroid (brightness)
        - Onset strength (rhythm)
        """
        self._init_audio_analyzer()
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {}
        
        # Load audio
        y, sr = self._librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        # Calculate hop length for target fps
        hop_length = int(sr / target_fps)
        
        # Extract features
        features = {}
        
        # RMS energy
        rms = self._librosa.feature.rms(y=y, hop_length=hop_length)[0]
        features["energy"] = rms / (rms.max() + 1e-6)  # Normalize 0-1
        
        # Pitch tracking
        pitches, magnitudes = self._librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
        pitch_track = []
        for i in range(pitches.shape[1]):
            idx = magnitudes[:, i].argmax()
            pitch_track.append(pitches[idx, i])
        pitch_track = np.array(pitch_track)
        pitch_track = np.where(pitch_track > 0, pitch_track, np.nan)
        # Fill NaN with interpolation
        nans = np.isnan(pitch_track)
        if not np.all(nans):
            pitch_track[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(~nans),
                pitch_track[~nans]
            )
        # Normalize to 0-1 (typical speech range 80-400 Hz)
        pitch_track = np.clip((pitch_track - 80) / 320, 0, 1)
        features["pitch"] = pitch_track
        
        # Spectral centroid (brightness)
        cent = self._librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        features["brightness"] = cent / (cent.max() + 1e-6)
        
        # Onset strength (rhythm/emphasis)
        onset = self._librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        features["onset"] = onset / (onset.max() + 1e-6)
        
        # Time axis
        n_frames = len(features["energy"])
        features["time"] = np.linspace(0, duration, n_frames)
        features["duration"] = duration
        features["fps"] = target_fps
        
        logger.info(f"Extracted audio features: {n_frames} frames over {duration:.2f}s")
        
        return features
    
    def transfer_motion(
        self,
        reference_sequence: MotionSequence,
        target_duration: float,
        comedian: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Transfer motion from reference sequence to target duration.
        
        Resamples and applies comedian (exaggeration) scaling.
        """
        if not reference_sequence.head_poses:
            return self._generate_neutral(target_duration)
        
        # Resample to target duration
        ref_duration = reference_sequence.duration
        scale = target_duration / ref_duration
        
        fps = 30.0
        n_frames = int(target_duration * fps)
        timeline = []
        
        for i in range(n_frames):
            t = i / fps
            ref_t = t / scale  # Time in reference sequence
            
            # Find nearest reference frame
            ref_idx = min(
                len(reference_sequence.head_poses) - 1,
                int(ref_t * reference_sequence.fps)
            )
            
            ref_pose = reference_sequence.head_poses[ref_idx]
            
            # Apply comedian scaling (exaggeration)
            timeline.append({
                "t": t,
                "pose": {
                    "pitch": ref_pose.pitch * comedian,
                    "yaw": ref_pose.yaw * comedian,
                    "roll": ref_pose.roll * comedian,
                    "base": 0.0,
                    "antenna_l": 0.0,
                    "antenna_r": 0.0,
                }
            })
        
        return timeline
    
    def synthesize_motion(
        self,
        audio_features: Dict[str, np.ndarray],
        emotion: str = "neutral",
        comedian: float = 1.0,
        antenna_role: str = "Eyebrows",
    ) -> List[Dict[str, Any]]:
        """
        Synthesize robot motion from audio features.
        
        Maps audio features to robot movements:
        - Energy → head movement amplitude
        - Pitch → head pitch (nod)
        - Brightness → eyebrow/antenna position
        - Onset → punctuation movements
        """
        if not audio_features:
            return self._generate_neutral(3.0)
        
        duration = audio_features.get("duration", 3.0)
        time_axis = audio_features.get("time", np.linspace(0, duration, 90))
        energy = audio_features.get("energy", np.ones_like(time_axis) * 0.5)
        pitch = audio_features.get("pitch", np.ones_like(time_axis) * 0.5)
        brightness = audio_features.get("brightness", np.ones_like(time_axis) * 0.5)
        onset = audio_features.get("onset", np.zeros_like(time_axis))
        
        # Emotion modifiers
        emotion_modifiers = {
            "Happy": {"pitch_bias": 5, "yaw_scale": 1.2, "antenna_base": 15},
            "Excited": {"pitch_bias": 8, "yaw_scale": 1.5, "antenna_base": 20},
            "Angry": {"pitch_bias": -5, "yaw_scale": 0.8, "antenna_base": -10},
            "Sarcastic": {"pitch_bias": 3, "yaw_scale": 1.3, "antenna_base": 5},
            "Whisper": {"pitch_bias": -2, "yaw_scale": 0.5, "antenna_base": 0},
            "Deadpan": {"pitch_bias": 0, "yaw_scale": 0.3, "antenna_base": 0},
        }
        
        # Handle custom emotions
        base_emotion = emotion.split(":")[0] if ":" in emotion else emotion
        mods = emotion_modifiers.get(base_emotion, {"pitch_bias": 0, "yaw_scale": 1.0, "antenna_base": 0})
        
        timeline = []
        
        for i, t in enumerate(time_axis):
            # Base motion from audio features
            e = energy[min(i, len(energy) - 1)]
            p = pitch[min(i, len(pitch) - 1)]
            b = brightness[min(i, len(brightness) - 1)]
            o = onset[min(i, len(onset) - 1)]
            
            # Map to robot movements
            # Pitch: modulated by audio pitch and energy
            head_pitch = (p - 0.5) * 20 * e + mods["pitch_bias"]
            
            # Yaw: subtle side-to-side based on onset
            head_yaw = np.sin(t * 2) * 5 * mods["yaw_scale"] + o * 10
            
            # Roll: very subtle
            head_roll = np.sin(t * 1.5) * 3 * e
            
            # Antenna based on role
            if antenna_role == "Eyebrows":
                # Follow brightness (expressiveness)
                antenna_l = b * 20 + mods["antenna_base"]
                antenna_r = b * 20 + mods["antenna_base"]
            elif antenna_role == "Arms":
                # Follow energy with asymmetry
                antenna_l = e * 25 + mods["antenna_base"]
                antenna_r = -e * 25 + mods["antenna_base"]
            elif antenna_role == "Too much":
                # Exaggerated
                antenna_l = b * 30 + e * 20
                antenna_r = b * 30 - e * 20
            else:  # "Don't use"
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
    
    def _generate_neutral(self, duration: float, fps: float = 30.0) -> List[Dict[str, Any]]:
        """Generate neutral idle motion"""
        n_frames = int(duration * fps)
        timeline = []
        
        for i in range(n_frames):
            t = i / fps
            # Subtle breathing motion
            timeline.append({
                "t": float(t),
                "pose": {
                    "pitch": float(np.sin(t * 0.5) * 2),
                    "yaw": float(np.sin(t * 0.3) * 3),
                    "roll": 0.0,
                    "base": 0.0,
                    "antenna_l": float(np.sin(t * 0.4) * 5),
                    "antenna_r": float(np.sin(t * 0.4) * 5),
                }
            })
        
        return timeline
