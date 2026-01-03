from __future__ import annotations

import os
import json
from typing import List, Optional, Tuple

import gradio as gr
import logging

from core.logging_utils import setup_logging
from core.storage import StorageManager
from core.profiles import ProfileManager, normalize_emotion
from core.constants import EMOTION_CHOICES_UI, ANTENNA_ROLES, VOICE_MODES
from core.state import AppState, CandidateTake
from core.script_parser import parse_script
from core.reachy_driver import ReachyMiniDriver
from learning.candidates import CandidateGenerator
from learning.approvals import ApprovalManager, Approval
from learning.training import Trainer
from media.extract import store_upload, try_extract_audio

setup_logging()
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))

storage = StorageManager(root=ROOT)
profiles = ProfileManager(storage=storage)

# Safe default: hardware disabled
reachy = ReachyMiniDriver(enable_hardware=False)
reachy.connect()

generator = CandidateGenerator(storage=storage, reachy=reachy)
approvals = ApprovalManager(storage=storage, profiles=profiles)
trainer = Trainer(storage=storage)

app_state = AppState()


def _session_id(request: gr.Request) -> str:
    return getattr(request, "session_hash", None) or "default"


def list_profile_choices() -> List[str]:
    ps = profiles.list_profiles()
    return [f"{p.name} [{p.profile_id[:8]}]" for p in ps]


def _profile_id_from_choice(choice: str) -> Optional[str]:
    if not choice:
        return None
    if "[" in choice and choice.endswith("]"):
        return choice.split("[", 1)[1][:-1]
    return None


def _coerce_path(x) -> Optional[str]:
    """
    Normalize Gradio returns to a filesystem path.

    Depending on Gradio version and component, x may be:
    - a string path
    - an object with `.name`
    - a dict like {"name": "..."} or {"path": "..."} or {"video": "..."}
    """
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if hasattr(x, "name") and isinstance(getattr(x, "name"), str):
        return x.name
    if isinstance(x, dict):
        for k in ("name", "path", "video", "file", "data"):
            v = x.get(k)
            if isinstance(v, str):
                return v
        # Sometimes nested dicts
        for v in x.values():
            if isinstance(v, dict):
                p = _coerce_path(v)
                if p:
                    return p
    return None


def _looks_like_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".mp4", ".mov", ".mkv", ".webm"}


def _looks_like_audio(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg"}


def _store_and_extract_from_video(profile_id: str, video_path: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Store the video under uploads/ and extract audio (wav) via ffmpeg if possible.
    Returns (stored_video_path, extracted_wav_path, message).
    """
    up_dir = os.path.join(storage.paths.uploads_dir, profile_id)
    upv = store_upload(video_path, up_dir)
    stored_video = upv.stored_path
    wav_out = os.path.join(up_dir, f"{upv.sha256}.wav")

    ok, msg = try_extract_audio(stored_video, wav_out)
    if ok:
        return stored_video, wav_out, f"Stored video + extracted audio ({msg})"
    return stored_video, None, f"Stored video; audio extraction failed ({msg})"


def _store_audio(profile_id: str, audio_path: str) -> Tuple[Optional[str], str]:
    up_dir = os.path.join(storage.paths.uploads_dir, profile_id)
    upa = store_upload(audio_path, up_dir)
    return upa.stored_path, "Stored audio."


# Mirror ROI setup
def capture_frame_for_roi():
    return reachy.get_frame()


def save_rois(
    profile_choice: str,
    fx: int,
    fy: int,
    fw: int,
    fh: int,
    sx: int,
    sy: int,
    sw: int,
    sh: int,
) -> str:
    pid = _profile_id_from_choice(profile_choice)
    if not pid:
        return "Select a profile first."
    storage.update_profile(
        pid,
        mirror_roi_front=json.dumps([fx, fy, fw, fh]),
        mirror_roi_side=json.dumps([sx, sy, sw, sh]),
    )
    return "Saved mirror ROIs for this profile."


def create_profile(name: str, consent_voice: bool) -> str:
    name = (name or "").strip()
    if not name:
        return "Please enter a profile name."
    pid = profiles.create_profile(name=name, consent_voice=consent_voice)
    return f"Created profile {name} [{pid[:8]}]."


def refresh_profiles_dropdown():
    return gr.update(choices=list_profile_choices())


def generate_candidates(
    request: gr.Request,
    profile_choice: str,
    emotion_choice: str,
    custom_mood: str,
    text: str,
    voice_mode_requested: str,
    n_candidates: int,
    comedian: float,
    source_type: str,
    audio_in,
    video_in,
    upload_file,
):
    """
    Voice reference is derived from VIDEO when available (webcam or uploaded).
    audio_in is only a fallback if the user provides audio-only.
    """
    sid = _session_id(request)
    sess = app_state.get_session(sid)

    pid = _profile_id_from_choice(profile_choice)
    if not pid:
        return "Select a profile.", [], None, None

    emotion = normalize_emotion(emotion_choice, custom_mood)
    text = (text or "").strip() or "(no text provided)"
    voice_mode = profiles.resolve_voice_mode(pid, voice_mode_requested)

    a_path = _coerce_path(audio_in)
    v_path = _coerce_path(video_in)
    u_path = _coerce_path(upload_file)

    reference_video_path: Optional[str] = None
    reference_audio_path: Optional[str] = None
    ingest_notes: List[str] = []

    # Priority: webcam video -> uploaded video -> audio-only fallback
    if v_path and _looks_like_video(v_path):
        stored_v, extracted_wav, msg = _store_and_extract_from_video(pid, v_path)
        reference_video_path = stored_v
        reference_audio_path = extracted_wav
        ingest_notes.append(f"Webcam: {msg}")

    elif u_path and _looks_like_video(u_path):
        stored_v, extracted_wav, msg = _store_and_extract_from_video(pid, u_path)
        reference_video_path = stored_v
        reference_audio_path = extracted_wav
        ingest_notes.append(f"Upload video: {msg}")

    else:
        audio_candidate = None
        if u_path and _looks_like_audio(u_path):
            audio_candidate = u_path
            ingest_notes.append("Upload: audio-only provided.")
        elif a_path and _looks_like_audio(a_path):
            audio_candidate = a_path
            ingest_notes.append("Microphone: audio-only provided.")

        if audio_candidate:
            stored_a, msg = _store_audio(pid, audio_candidate)
            reference_audio_path = stored_a
            ingest_notes.append(msg)
        else:
            ingest_notes.append("No usable video/audio provided. Candidates generated without a reference sample.")

    takes = generator.generate_candidates(
        profile_id=pid,
        emotion=emotion,
        text=text,
        voice_mode=voice_mode,
        comedian=comedian,
        n_candidates=n_candidates,
        source_type=source_type,
        reference_video_path=reference_video_path,
        reference_audio_path=reference_audio_path,
    )

    sess.current_profile_id = pid
    sess.candidates = []
    for i, t in enumerate(takes):
        sess.candidates.append(
            CandidateTake(
                candidate_id=str(i),
                episode_id=t.episode_id,
                emotion=emotion,
                text=text,
                voice_mode=voice_mode,
                comedian=comedian,
                preview_audio_path=t.preview_audio_path,
            )
        )

    table = [
        {"idx": i, "episode_id": c.episode_id[:8], "emotion": c.emotion, "voice": c.voice_mode}
        for i, c in enumerate(sess.candidates)
    ]
    audio_preview = sess.candidates[0].preview_audio_path if sess.candidates else None

    status = f"Generated {len(sess.candidates)} candidates for {emotion} (voice={voice_mode}). " + " | ".join(ingest_notes)
    if voice_mode_requested == "imitate":
        prof = storage.get_profile(pid)
        if prof and not bool(prof["consent_voice"]):
            status += " Voice imitation not allowed (no consent), used synthetic instead."

    # Return stored path if we have one; gr.Video can display a filepath in most versions
    video_preview = reference_video_path if reference_video_path else None
    return status, table, audio_preview, video_preview


def approve_take(request: gr.Request, best_idx: int, antenna_role: str, notes: str) -> str:
    sid = _session_id(request)
    sess = app_state.get_session(sid)
    if not sess.candidates:
        return "No candidates to approve. Generate candidates first."

    try:
        best_idx = int(best_idx)
    except Exception:
        return "Pick a valid best candidate index."

    if best_idx < 0 or best_idx >= len(sess.candidates):
        return "Best index out of range."

    episode_ids = [c.episode_id for c in sess.candidates]
    chosen = sess.candidates[best_idx].episode_id
    ranking = [best_idx] + [i for i in range(len(sess.candidates)) if i != best_idx]

    ap = Approval(episode_id=chosen, ranking=ranking, antenna_role=antenna_role, notes=notes or "")
    approvals.approve(profile_id=sess.current_profile_id, episode_ids=episode_ids, approval=ap)
    return f"Approved candidate {best_idx} (episode {chosen[:8]}), antenna={antenna_role}."


def update_profile_model(profile_choice: str) -> str:
    pid = _profile_id_from_choice(profile_choice)
    if not pid:
        return "Select a profile."
    info = trainer.update_profile_models(profile_id=pid)
    return json.dumps(info, indent=2)


def mood_progress(profile_choice: str):
    pid = _profile_id_from_choice(profile_choice)
    if not pid:
        return []
    return profiles.mood_progress_table(pid)


def run_simulation(profile_choice: str, script_text: str, voice_mode_requested: str, comedian: float, run_mode: str):
    pid = _profile_id_from_choice(profile_choice)
    if not pid:
        return "Select a profile.", None

    voice_mode = profiles.resolve_voice_mode(pid, voice_mode_requested)
    script = parse_script(script_text or "")
    blocks = script.blocks or []
    if not blocks:
        return "Script is empty.", None

    outputs = []
    for b in blocks:
        takes = generator.generate_candidates(
            profile_id=pid,
            emotion=b.emotion,
            text=b.text,
            voice_mode=voice_mode,
            comedian=comedian,
            n_candidates=1,
            source_type="simulation",
            reference_video_path=None,
            reference_audio_path=None,
        )
        t0 = takes[0]
        outputs.append(t0.preview_audio_path)

        if run_mode == "Perform on Reachy":
            timeline_path = os.path.join(storage.paths.episodes_dir, t0.episode_id, "timeline.json")
            try:
                with open(timeline_path, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                from core.reachy_driver import TimelinePoint, Pose

                timeline = [TimelinePoint(t=float(x["t"]), pose=Pose(**x["pose"])) for x in arr]
                reachy.play_timeline(timeline, realtime=True)
            except Exception:
                logger.exception("Failed to execute timeline")

    return f"Ran {len(outputs)} block(s) in {run_mode}.", (outputs[0] if outputs else None)


def toggle_hardware(enable: bool) -> str:
    reachy.enable_hardware = bool(enable)
    reachy.connect()
    return "Connected" if reachy.is_connected() else "Preview-only (not connected or disabled)"


def build_ui():
    with gr.Blocks(title="Reachy Mimic (Mini Lite)") as demo:
        gr.Markdown(
            "# Reachy Mimic (Mini Lite)\n"
            "Incremental, approval-based learning with profiles, moods, and synced voice+motion.\n\n"
            "**Your voice reference is sourced from VIDEO** when available (webcam or uploaded)."
        )

        with gr.Row():
            hw_enable = gr.Checkbox(value=False, label="Enable hardware (actuate Reachy)")
            hw_status = gr.Textbox(value="Preview-only mode", label="Hardware status", interactive=False)
        hw_enable.change(toggle_hardware, hw_enable, hw_status)

        with gr.Tabs():
            with gr.Tab("Learning"):
                gr.Markdown(
                    "## Learning\n"
                    "Record or upload a short clip. The app extracts audio from the video (ffmpeg recommended).\n"
                    "Then generate candidates, approve the best, and incrementally update the profile."
                )

                with gr.Accordion("Create profile", open=False):
                    new_name = gr.Textbox(label="Profile name")
                    consent = gr.Checkbox(value=False, label="I have consent to imitate this person's voice")
                    create_btn = gr.Button("Create")
                    create_out = gr.Textbox(label="Result", interactive=False)
                    create_btn.click(lambda n, c: create_profile(n, c), [new_name, consent], create_out)

                profile = gr.Dropdown(choices=list_profile_choices(), label="Select profile")
                refresh = gr.Button("Refresh profiles")
                refresh.click(refresh_profiles_dropdown, None, profile)

                with gr.Row():
                    emotion = gr.Dropdown(choices=EMOTION_CHOICES_UI, value="Happy", label="Mood/Emotion")
                    custom = gr.Textbox(label="Custom one-word mood (used only if Mood=Custom)")
                    voice_mode = gr.Dropdown(choices=VOICE_MODES, value="auto", label="Voice mode (auto enforces consent)")

                text = gr.Textbox(lines=2, label="Text to say (optional)")

                with gr.Row():
                    n_candidates = gr.Slider(1, 4, value=2, step=1, label="# Candidates")
                    comedian = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Comedian slider (exaggeration)")
                    source_type = gr.Dropdown(choices=["live", "upload"], value="live", label="Learning source type")

                gr.Markdown("### Record live (recommended)")
                with gr.Row():
                    audio_in = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="(Optional) Mic audio fallback (use only if no video)",
                    )
                    # IMPORTANT: DO NOT use type=... on gr.Video for older Gradio versions
                    video_in = gr.Video(
                        sources=["webcam"],
                        label="Webcam video (drives voice learning)",
                    )

                gr.Markdown("### Upload phone video (recommended)")
                upload = gr.File(file_types=[".mp4", ".mov", ".m4a", ".wav", ".mp3"], label="Upload file")

                gen_btn = gr.Button("Generate candidates")
                status = gr.Textbox(label="Status", interactive=False)

                candidates_table = gr.Dataframe(
                    headers=["idx", "episode_id", "emotion", "voice"],
                    datatype=["number", "str", "str", "str"],
                    row_count=5,
                    col_count=(4, "fixed"),
                    interactive=False,
                    label="Candidates",
                )

                with gr.Row():
                    candidate_audio = gr.Audio(type="filepath", label="Preview audio (first candidate)")
                    reference_video_preview = gr.Video(label="Reference video (playback)")

                gen_btn.click(
                    generate_candidates,
                    inputs=[
                        profile,
                        emotion,
                        custom,
                        text,
                        voice_mode,
                        n_candidates,
                        comedian,
                        source_type,
                        audio_in,
                        video_in,
                        upload,
                    ],
                    outputs=[status, candidates_table, candidate_audio, reference_video_preview],
                )

                gr.Markdown("### Approval")
                best_idx = gr.Number(value=0, label="Best candidate idx (0-based)")
                antenna_role = gr.Dropdown(choices=ANTENNA_ROLES, value="Eyebrows", label="Antenna role label")
                notes = gr.Textbox(lines=2, label="Notes (optional)")
                approve_btn = gr.Button("Approve best take")
                approve_out = gr.Textbox(label="Approval result", interactive=False)
                approve_btn.click(approve_take, inputs=[best_idx, antenna_role, notes], outputs=approve_out)

                gr.Markdown("### Incremental training")
                update_btn = gr.Button("Update profile model")
                update_out = gr.Textbox(lines=12, label="Update output", interactive=False)
                update_btn.click(update_profile_model, inputs=[profile], outputs=update_out)

                gr.Markdown("### Progress")
                progress = gr.Dataframe(headers=["mood", "approved"], datatype=["str", "number"], interactive=False)
                profile.change(mood_progress, inputs=profile, outputs=progress)
                approve_btn.click(mood_progress, inputs=profile, outputs=progress)

                with gr.Accordion("Mirror Setup (optional, for self-view learning)", open=False):
                    gr.Markdown("Capture a frame and set ROIs for front and side mirror reflections (x,y,w,h).")
                    cap_btn = gr.Button("Capture current Reachy frame")
                    frame_img = gr.Image(type="numpy", label="Captured frame")
                    cap_btn.click(lambda: capture_frame_for_roi(), None, frame_img)

                    with gr.Row():
                        fx = gr.Number(value=0, label="front x")
                        fy = gr.Number(value=0, label="front y")
                        fw = gr.Number(value=200, label="front w")
                        fh = gr.Number(value=200, label="front h")
                    with gr.Row():
                        sx = gr.Number(value=0, label="side x")
                        sy = gr.Number(value=0, label="side y")
                        sw = gr.Number(value=200, label="side w")
                        sh = gr.Number(value=200, label="side h")

                    save_btn = gr.Button("Save ROIs to profile")
                    save_out = gr.Textbox(label="ROI save result", interactive=False)
                    save_btn.click(save_rois, inputs=[profile, fx, fy, fw, fh, sx, sy, sw, sh], outputs=save_out)

            with gr.Tab("Simulation"):
                gr.Markdown("## Simulation\nSelect a profile, write a script, and preview or perform on Reachy.")

                profile2 = gr.Dropdown(choices=list_profile_choices(), label="Select profile")
                refresh2 = gr.Button("Refresh profiles")
                refresh2.click(refresh_profiles_dropdown, None, profile2)

                voice2 = gr.Dropdown(choices=VOICE_MODES, value="auto", label="Voice mode (auto enforces consent)")
                comedian2 = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Comedian slider")
                run_mode = gr.Radio(choices=["Preview only", "Perform on Reachy"], value="Preview only", label="Run mode")

                script = gr.Textbox(
                    lines=12,
                    label="Script",
                    value=(
                        'PROFILE: Me\nVOICE: auto\nCOMEDIAN: 1.0\n\n'
                        'SAY(Emotion=Deadpan):\n'
                        '"Of course I am calm. I am always calm."\n'
                    ),
                )

                run_btn = gr.Button("Run")
                sim_status = gr.Textbox(label="Status", interactive=False)
                sim_audio = gr.Audio(type="filepath", label="Preview audio (first block)")
                run_btn.click(run_simulation, inputs=[profile2, script, voice2, comedian2, run_mode], outputs=[sim_status, sim_audio])

        gr.Markdown(
            "Notes:\n"
            "- Install ffmpeg for best video→audio extraction: `brew install ffmpeg`\n"
            "- The ML is still stubbed, but reference video/audio is now stored and passed into the generator.\n"
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
