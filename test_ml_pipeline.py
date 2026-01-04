#!/usr/bin/env python3
"""
Test script for the ML learning pipeline.

Tests:
1. TTS providers (Edge TTS by default)
2. Motion extraction from video
3. Audio feature extraction
4. Motion generation
5. Sync learning
6. Full candidate generation
"""

import os
import sys
import json
import tempfile
import logging

# Add the project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.logging_utils import setup_logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tts_providers():
    """Test TTS provider initialization and synthesis"""
    print("\n" + "=" * 60)
    print("1. TESTING TTS PROVIDERS")
    print("=" * 60)
    
    from ml.providers import TTSManager, PlaceholderTTSProvider
    
    # Test placeholder (always works)
    print("\n📢 Testing PlaceholderTTS...")
    placeholder = PlaceholderTTSProvider()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        result = placeholder.synthesize(
            text="Hello world, this is a test.",
            output_path=f.name,
            emotion="Happy",
        )
        print(f"   ✓ Placeholder: {result.duration_seconds:.2f}s audio generated")
        os.unlink(f.name)
    
    # Test TTSManager auto-detection
    print("\n📢 Testing TTSManager auto-detection...")
    manager = TTSManager()
    provider = manager.get_provider()
    print(f"   ✓ Auto-detected provider: {provider.name}")
    
    # Test Edge TTS if available
    try:
        print("\n📢 Testing Edge TTS (free, no API key)...")
        manager.set_provider("edge")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            result = manager.synthesize(
                text="Hello! This is a test of the Edge TTS provider.",
                output_path=f.name,
                emotion="Happy",
            )
            print(f"   ✓ Edge TTS: {result.duration_seconds:.2f}s audio generated")
            print(f"   ✓ File: {f.name}")
            # Keep file for inspection
            
    except Exception as e:
        print(f"   ⚠️  Edge TTS failed: {e}")
        print("   Install with: pip install edge-tts")
    
    # Check for API-based providers
    print("\n📢 Checking API-based providers...")
    
    if os.environ.get("OPENAI_API_KEY"):
        print("   ✓ OpenAI API key found")
    else:
        print("   ○ OpenAI API key not set (set OPENAI_API_KEY to enable)")
    
    if os.environ.get("ELEVENLABS_API_KEY"):
        print("   ✓ ElevenLabs API key found")
    else:
        print("   ○ ElevenLabs API key not set (set ELEVENLABS_API_KEY to enable)")
    
    return True


def test_motion_extraction():
    """Test motion extraction from video"""
    print("\n" + "=" * 60)
    print("2. TESTING MOTION EXTRACTION")
    print("=" * 60)
    
    try:
        from ml.motion import MotionExtractor
        
        extractor = MotionExtractor()
        print("   ✓ MotionExtractor initialized")
        
        # We can't test without a real video, but we can verify initialization
        print("   ○ Skipping video extraction (no test video)")
        print("   ℹ️  To test, provide a video path and call:")
        print("      sequence = extractor.extract_from_video('path/to/video.mp4')")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Motion extraction not available: {e}")
        print("   Install with: pip install mediapipe opencv-python")
        return False


def test_audio_features():
    """Test audio feature extraction"""
    print("\n" + "=" * 60)
    print("3. TESTING AUDIO FEATURE EXTRACTION")
    print("=" * 60)
    
    try:
        from ml.motion import MotionGenerator
        import numpy as np
        import soundfile as sf
        
        generator = MotionGenerator()
        print("   ✓ MotionGenerator initialized")
        
        # Create a test audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Generate 3 seconds of audio
            sr = 22050
            duration = 3.0
            t = np.linspace(0, duration, int(sr * duration))
            # Sine wave with varying frequency
            audio = 0.3 * np.sin(2 * np.pi * (200 + 100 * np.sin(2 * np.pi * t / 2)) * t)
            sf.write(f.name, audio, sr)
            
            # Extract features
            features = generator.extract_audio_features(f.name)
            
            print(f"   ✓ Extracted features:")
            print(f"      - Duration: {features.get('duration', 0):.2f}s")
            print(f"      - Energy frames: {len(features.get('energy', []))}")
            print(f"      - Pitch frames: {len(features.get('pitch', []))}")
            print(f"      - Brightness frames: {len(features.get('brightness', []))}")
            
            os.unlink(f.name)
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Audio feature extraction not available: {e}")
        print("   Install with: pip install librosa")
        return False


def test_motion_generation():
    """Test motion generation from audio features"""
    print("\n" + "=" * 60)
    print("4. TESTING MOTION GENERATION")
    print("=" * 60)
    
    try:
        from ml.motion import MotionGenerator
        import numpy as np
        
        generator = MotionGenerator()
        
        # Create fake audio features
        n_frames = 90  # 3 seconds at 30fps
        features = {
            "duration": 3.0,
            "fps": 30.0,
            "time": np.linspace(0, 3.0, n_frames),
            "energy": np.random.rand(n_frames) * 0.5 + 0.5,
            "pitch": np.random.rand(n_frames) * 0.5 + 0.25,
            "brightness": np.random.rand(n_frames) * 0.5 + 0.25,
            "onset": np.random.rand(n_frames) * 0.3,
        }
        
        # Test each emotion
        emotions = ["Happy", "Excited", "Angry", "Sarcastic", "Whisper", "Deadpan"]
        
        for emotion in emotions:
            timeline = generator.synthesize_motion(
                audio_features=features,
                emotion=emotion,
                comedian=1.0,
                antenna_role="Eyebrows",
            )
            print(f"   ✓ {emotion}: {len(timeline)} motion frames")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Motion generation failed: {e}")
        return False


def test_sync_learning():
    """Test sync learning from examples"""
    print("\n" + "=" * 60)
    print("5. TESTING SYNC LEARNING")
    print("=" * 60)
    
    try:
        from ml.sync import SyncLearner
        import numpy as np
        import tempfile
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = SyncLearner(tmpdir)
            print("   ✓ SyncLearner initialized")
            
            # Add some fake examples
            n_frames = 90
            for i in range(5):
                audio_features = {
                    "duration": 3.0,
                    "energy": np.random.rand(n_frames),
                    "pitch": np.random.rand(n_frames),
                    "brightness": np.random.rand(n_frames),
                }
                
                motion = [
                    {
                        "t": j / 30,
                        "pose": {
                            "pitch": float(np.random.randn() * 10),
                            "yaw": float(np.random.randn() * 15),
                            "roll": float(np.random.randn() * 5),
                            "antenna_l": float(np.random.randn() * 10),
                            "antenna_r": float(np.random.randn() * 10),
                        }
                    }
                    for j in range(n_frames)
                ]
                
                learner.add_example(
                    episode_id=f"test_{i}",
                    emotion="Happy",
                    audio_features=audio_features,
                    motion_timeline=motion,
                    antenna_role="Eyebrows",
                    approval_rank=i,
                )
            
            print(f"   ✓ Added 5 training examples")
            
            # Run learning
            stats = learner.learn()
            print(f"   ✓ Learning complete: version {stats.get('version', 0)}")
            
            # Check learned profile
            profile = learner.get_profile("Happy")
            print(f"   ✓ Learned profile for 'Happy':")
            print(f"      - n_examples: {profile.get('n_examples', 0)}")
            print(f"      - energy_to_pitch: {profile.get('energy_to_pitch', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Sync learning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full candidate generation pipeline"""
    print("\n" + "=" * 60)
    print("6. TESTING FULL PIPELINE")
    print("=" * 60)
    
    try:
        from core.storage import StorageManager
        from core.reachy_driver import ReachyMiniDriver
        from learning.candidates import CandidateGenerator
        from ml.config import get_config_manager
        
        # Initialize config
        config_manager = get_config_manager(ROOT)
        config = config_manager.load()
        print(f"   ✓ Config loaded: TTS provider = {config.tts.provider}")
        
        # Initialize components
        storage = StorageManager(root=ROOT)
        reachy = ReachyMiniDriver(enable_hardware=False)
        generator = CandidateGenerator(storage=storage, reachy=reachy, config=config)
        
        print("   ✓ Components initialized")
        
        # Create test profile
        profile_id = storage.create_profile(
            name="MLPipelineTest",
            consent_voice=True,
        )
        print(f"   ✓ Created test profile: {profile_id[:8]}")
        
        # Generate candidates
        print("\n   🎬 Generating 2 candidates...")
        takes = generator.generate_candidates(
            profile_id=profile_id,
            emotion="Happy",
            text="Hello! This is a test of the ML pipeline.",
            voice_mode="synthetic",
            comedian=1.0,
            n_candidates=2,
            source_type="test",
        )
        
        print(f"\n   ✓ Generated {len(takes)} candidates:")
        for i, take in enumerate(takes):
            print(f"      [{i}] Episode: {take.episode_id[:8]}")
            print(f"          Audio: {os.path.exists(take.preview_audio_path) if take.preview_audio_path else False}")
            print(f"          Motion: {len(take.timeline)} frames")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("   MIMIC ME - ML PIPELINE TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results["TTS Providers"] = test_tts_providers()
    results["Motion Extraction"] = test_motion_extraction()
    results["Audio Features"] = test_audio_features()
    results["Motion Generation"] = test_motion_generation()
    results["Sync Learning"] = test_sync_learning()
    results["Full Pipeline"] = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 70)
    print("   TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("   🎉 ALL TESTS PASSED!")
        print("\n   The ML pipeline is ready to use.")
        print("   Run the app with: python app_gradio.py")
    else:
        print("   ⚠️  SOME TESTS FAILED")
        print("\n   Check the error messages above and install missing dependencies.")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
