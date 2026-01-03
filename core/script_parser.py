from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import re

from .profiles import normalize_custom_mood

@dataclass
class SayBlock:
    emotion: str
    text: str

@dataclass
class Script:
    profile: Optional[str] = None
    voice: str = "auto"
    comedian: float = 1.0
    blocks: List[SayBlock] = None

HEADER_RE = re.compile(r"^\s*(PROFILE|VOICE|COMEDIAN)\s*:\s*(.+?)\s*$", re.IGNORECASE)
SAY_START_RE = re.compile(r"^\s*SAY\s*\((.*?)\)\s*:\s*$", re.IGNORECASE)
QUOTE_RE = re.compile(r'^\s*"(.*)"\s*$')

def _parse_args(arg_str: str) -> Dict[str, str]:
    args = {}
    parts = [p.strip() for p in arg_str.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            args[k.strip().lower()] = v.strip()
    return args

def _normalize_emotion_token(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "Custom:neutral"
    if raw.lower().startswith("custom:"):
        w = normalize_custom_mood(raw.split(":", 1)[1])
        return f"Custom:{w}"
    return raw

def parse_script(text: str) -> Script:
    lines = (text or "").splitlines()
    script = Script(profile=None, voice="auto", comedian=1.0, blocks=[])

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if not line.strip():
            i += 1
            continue

        m = HEADER_RE.match(line)
        if m:
            key = m.group(1).lower()
            val = m.group(2).strip()
            if key == "profile":
                script.profile = val
            elif key == "voice":
                script.voice = val.lower()
            elif key == "comedian":
                try:
                    script.comedian = float(val)
                except ValueError:
                    script.comedian = 1.0
            i += 1
            continue

        m = SAY_START_RE.match(line)
        if m:
            args = _parse_args(m.group(1))
            emotion = _normalize_emotion_token(args.get("emotion", ""))
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                break
            q = QUOTE_RE.match(lines[j].strip())
            say_text = q.group(1) if q else lines[j].strip()
            script.blocks.append(SayBlock(emotion=emotion, text=say_text))
            i = j + 1
            continue

        script.blocks.append(SayBlock(emotion="Custom:neutral", text=line.strip()))
        i += 1

    return script
