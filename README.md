# Reachy Mimic Me App — Incremental, Approval-Based Learning App

Local Gradio app for Reachy Mini Lite to incrementally teach:
- Profiles (per person)
- Standard moods: Angry, Happy, Sarcastic, Excited, Whisper, Deadpan
- Custom one-word moods
- Learned (not rule-based) sync between voice and motion (pluggable stubs included)
- Antenna roles learned via approvals: Eyebrows / Arms / Too much / Don't use
- Synthetic voice fallback if no consent for imitation
- Learning from live recordings and uploaded phone videos

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app_gradio.py
```
or if you use conda:

```bash
conda create -n mimic_me python=3.11 -y
conda activate mimic_me
pip install -r requirements.txt
python app_gradio.py
```
or if you use mamba:
```bash
mamba create -n mimic_me python=3.11 -y
mamba activate mimic_me
pip install -r requirements.txt
python app_gradio.py
```

See `docs/ARCHITECTURE.md` for a full checklist and where each feature lives.
