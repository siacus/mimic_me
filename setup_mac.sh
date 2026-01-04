#!/bin/bash
# Setup script for Mimic Me on macOS Apple Silicon (M1/M2/M3)
# Works with conda, mamba, or micromamba

set -e

echo "=============================================="
echo "  Mimic Me - Setup for Apple Silicon"
echo "=============================================="

# Detect package manager
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
    echo "✓ Using mamba"
elif command -v micromamba &> /dev/null; then
    PKG_MGR="micromamba"
    echo "✓ Using micromamba"
elif command -v conda &> /dev/null; then
    PKG_MGR="conda"
    echo "✓ Using conda"
else
    echo "❌ No conda/mamba found. Please install one of:"
    echo "   - Miniforge: https://github.com/conda-forge/miniforge"
    echo "   - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment exists
ENV_NAME="mimic_me"
if $PKG_MGR env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        $PKG_MGR env remove -n $ENV_NAME -y
    else
        echo "Activating existing environment..."
        echo "Run: $PKG_MGR activate $ENV_NAME"
        exit 0
    fi
fi

# Create environment
echo ""
echo "Creating conda environment..."
$PKG_MGR env create -f environment.yml -y

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  $PKG_MGR activate $ENV_NAME"
echo ""
echo "To run the app:"
echo "  python app_gradio.py"
echo ""
echo "To test the ML pipeline:"
echo "  python test_ml_pipeline.py"
echo ""
echo "Optional: Install additional TTS providers:"
echo "  pip install openai        # For OpenAI TTS"
echo "  pip install elevenlabs    # For ElevenLabs (voice cloning)"
echo "  pip install TTS           # For Coqui XTTS (local voice cloning)"
echo ""
