# Reachy Mimic Me App

**Incremental, Approval-Based Learning for Reachy Mini Robot**

Optimized for **Apple Silicon (M1/M2/M3)** with MPS acceleration.

## Features

- 🎭 **Profiles** with consent-based voice imitation
- 😊 **Moods**: Angry, Happy, Sarcastic, Excited, Whisper, Deadpan + custom
- 🎤 **Pluggable TTS**: Edge (free), OpenAI, ElevenLabs, Coqui XTTS
- 🎬 **Motion extraction** from video via MediaPipe
- 🔄 **Learned sync** between voice and motion
- 👍 **Approval-based learning** from your feedback
- 🍎 **Apple Silicon optimized** - MPS acceleration for ML

---

## Quick Start (Apple Silicon M1/M2/M3)

### Option 1: Mamba (Recommended - Fastest)

```bash
# Install mamba if you haven't
# https://github.com/conda-forge/miniforge

# Create environment
mamba env create -f environment.yml

# Activate
mamba activate mimic_me

# Run
python app_gradio.py
```

### Option 2: Conda

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate mimic_me

# Run
python app_gradio.py
```

### Option 3: Setup Script

```bash
# Make executable and run
chmod +x setup_mac.sh
./setup_mac.sh

# Activate
mamba activate mimic_me  # or conda activate mimic_me

# Run
python app_gradio.py
```

### Option 4: Pure pip (if you don't use conda)

```bash
# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt

# Run
python app_gradio.py
```

---

## TTS Provider Setup

### Edge TTS (Default - Free, No API Key)

Works out of the box! Microsoft's neural voices.

```bash
# Already included in environment.yml
# No configuration needed
```

### OpenAI TTS

```bash
# Install
pip install openai

# Set API key
export OPENAI_API_KEY="sk-your-key-here"

# Edit config.json
# Change: "provider": "openai"
```

### ElevenLabs (Voice Cloning)

```bash
# Install
pip install elevenlabs

# Set API key
export ELEVENLABS_API_KEY="your-key-here"

# Edit config.json
# Change: "provider": "elevenlabs"
```

### Coqui XTTS (Local Voice Cloning - Great on M2!)

```bash
# Install
pip install TTS

# Edit config.json
# Change: "provider": "coqui_xtts"

# First run downloads ~2GB model
# Uses MPS acceleration automatically on Apple Silicon!
```

---

## Configuration

Edit `config.json` or use environment variables:

```bash
# TTS provider
export MIMIC_ME_TTS_PROVIDER=openai

# API keys
export OPENAI_API_KEY=sk-...
export ELEVENLABS_API_KEY=...

# Device (auto-detects M2 MPS)
export MIMIC_ME_DEVICE=auto  # or mps, cuda, cpu
```

### config.json

```json
{
  "tts": {
    "provider": "edge",
    "coqui_device": "auto"
  },
  "motion": {
    "device": "auto",
    "use_gpu": true
  }
}
```

---

## Project Structure

```
mimic_me/
├── app_gradio.py          # Main Gradio UI
├── config.json            # Configuration
├── environment.yml        # Conda/Mamba environment
├── requirements.txt       # Pip requirements
├── setup_mac.sh          # macOS setup script
│
├── core/                  # Core functionality
│   ├── storage.py         # SQLite + filesystem
│   ├── profiles.py        # Profile management
│   ├── reachy_driver.py   # Robot control
│   └── script_parser.py   # Script DSL
│
├── ml/                    # ML pipeline
│   ├── providers.py       # TTS backends
│   ├── motion.py          # Motion extraction/generation
│   ├── sync.py            # Audio-motion sync learning
│   └── config.py          # Configuration
│
├── learning/              # Learning pipeline
│   ├── candidates.py      # Candidate generation
│   ├── approvals.py       # Approval workflow
│   └── training.py        # Incremental training
│
└── data/                  # Generated data
    ├── app.db             # SQLite database
    ├── episodes/          # Generated takes
    └── profiles/          # Per-profile data
```

---

## Learning Pipeline

### 1. Create Profile
- Set name and voice consent
- Consent enables voice cloning

### 2. Record/Upload Training Data
- **Best**: Upload phone video with audio
- Video enables motion extraction via MediaPipe

### 3. Generate Candidates
- TTS synthesis with emotion
- Motion generation from audio
- Multiple candidates to choose from

### 4. Approve Best Take
- Select the best candidate
- Label antenna role
- Feedback trains the model

### 5. Update Model
- Incremental learning
- Per-emotion profiles
- Learns your preferences

---

## Testing

```bash
# Test ML pipeline
python test_ml_pipeline.py

# Test data pipeline
python test_data_pipeline.py
```

---

## Troubleshooting

### "No module named 'mediapipe'"
```bash
pip install mediapipe
```

### "ffmpeg not found"
```bash
# With conda/mamba (recommended)
mamba install ffmpeg

# Or with Homebrew
brew install ffmpeg
```

### "MPS not available"
Make sure you're using Python from conda/mamba, not system Python:
```bash
which python  # Should show conda environment path
```

### Edge TTS hangs
Check your internet connection - Edge TTS requires network access.

### Coqui XTTS slow on first run
First run downloads ~2GB model. Subsequent runs use cached model.

---

## Hardware Notes

- **Without Reachy**: Works in preview-only mode
- **With Reachy Mini**: Enable hardware toggle, install `reachy-mini`
- **M2 Max**: Excellent performance with MPS acceleration

---

## License

MIT License
