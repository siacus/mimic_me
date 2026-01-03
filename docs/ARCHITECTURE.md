# Architecture checklist

Everything discussed is represented:

- Local web app with Learning and Simulation tabs (app_gradio.py)
- Standard moods: Angry, Happy, Sarcastic, Excited, Whisper, Deadpan (core/constants.py)
- Custom one-word mood support (core/profiles.py)
- Profiles with consent gating for voice imitation; synthetic fallback (core/profiles.py)
- Incremental learning: teach one mood now, add more later; progress tracking (core/profiles.py)
- Approval mechanism: best-take selection + antenna role label (learning/approvals.py)
- Comedian slider is the only explicit non-learned control (core/reachy_driver.py)
- Upload training videos (phone → mac): store + audio extraction attempt (media/extract.py)
- Scripted simulation with one-word emotions, including Custom:word (core/script_parser.py)
- Reachy Mini Lite driver wrapper with safety clamps; hardware disabled by default (core/reachy_driver.py)
- ML extension points for JEPA + learned sync + learned antenna allocation (learning/candidates.py, learning/training.py)
