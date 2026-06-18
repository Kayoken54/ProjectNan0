"""
Nan0 V7 Identity Persistence Layer.
Local-first actor identity memory for speech normalization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

IDENTITY_PATH = Path("data/nan0/identity_memory.json")

DEFAULT_MEMORY: Dict[str, Any] = {
    "actors": {
        "kyo": {
            "display_name": "Kyo",
            "gender": "girl",
            "pronouns": ["she", "her"],
            "relationship": "creator_anchor",
            "importance": 1.0,
            "notes": [
                "Kyo is Nan0's primary emotional anchor.",
                "Kyo is a girl.",
                "Kyo should never be referred to as the user."
            ]
        }
    },
    "rules": {
        "never_call_kyo_user": True,
        "resolve_kyo_gender": True,
        "internal_json_stays_internal": True,
        "thought_object_not_spoken_line": True
    }
}


def _ensure_parent() -> None:
    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_identity_memory() -> Dict[str, Any]:
    _ensure_parent()
    if not IDENTITY_PATH.exists():
        save_identity_memory(DEFAULT_MEMORY)
        return json.loads(json.dumps(DEFAULT_MEMORY))
    try:
        data = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = json.loads(json.dumps(DEFAULT_MEMORY))
    # Merge defaults without destroying user-learned data.
    data.setdefault("actors", {})
    data.setdefault("rules", {})
    for actor, value in DEFAULT_MEMORY["actors"].items():
        data["actors"].setdefault(actor, value)
        if isinstance(data["actors"][actor], dict):
            for k, v in value.items():
                data["actors"][actor].setdefault(k, v)
    for k, v in DEFAULT_MEMORY["rules"].items():
        data["rules"].setdefault(k, v)
    save_identity_memory(data)
    return data


def save_identity_memory(data: Dict[str, Any]) -> None:
    _ensure_parent()
    IDENTITY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def remember_from_user_text(text: str) -> None:
    """Extract durable identity corrections from Kyo/user messages."""
    if not text:
        return
    lowered = text.lower()
    data = load_identity_memory()
    actors = data.setdefault("actors", {})
    kyo = actors.setdefault("kyo", DEFAULT_MEMORY["actors"]["kyo"].copy())

    if re.search(r"\bkyo\b.*\b(girl|woman|female|she/her|she is|she's)\b", lowered):
        kyo["gender"] = "girl"
        kyo["pronouns"] = ["she", "her"]
        notes = kyo.setdefault("notes", [])
        if "Kyo is a girl." not in notes:
            notes.append("Kyo is a girl.")

    if "kyo is not the user" in lowered or "don't call kyo the user" in lowered or "do not call kyo the user" in lowered:
        data.setdefault("rules", {})["never_call_kyo_user"] = True
        notes = kyo.setdefault("notes", [])
        if "Kyo should never be referred to as the user." not in notes:
            notes.append("Kyo should never be referred to as the user.")

    save_identity_memory(data)


def get_actor(actor_id: str) -> Optional[Dict[str, Any]]:
    data = load_identity_memory()
    return data.get("actors", {}).get(actor_id.lower())


def resolve_identity_text(text: str) -> str:
    """Apply durable actor identity rules to a final spoken line."""
    if not text:
        return text
    data = load_identity_memory()
    kyo = data.get("actors", {}).get("kyo", {})
    if data.get("rules", {}).get("resolve_kyo_gender", True):
        # Correct common cases where Kyo is nearby in the line.
        text = re.sub(r"\bHe\b(?=[^.!?]{0,60}\bKyo\b)", "She", text)
        text = re.sub(r"\bhe\b(?=[^.!?]{0,60}\bKyo\b)", "she", text)
        text = re.sub(r"\bHis\b(?=[^.!?]{0,60}\bKyo\b)", "Her", text)
        text = re.sub(r"\bhis\b(?=[^.!?]{0,60}\bKyo\b)", "her", text)
        # Also correct lines that target Kyo but omit her name.
        if "kyo" in text.lower():
            text = re.sub(r"\bHe\b", "She", text)
            text = re.sub(r"\bhe\b", "she", text)
            text = re.sub(r"\bHim\b", "Her", text)
            text = re.sub(r"\bhim\b", "her", text)
            text = re.sub(r"\bHis\b", "Her", text)
            text = re.sub(r"\bhis\b", "her", text)
    if data.get("rules", {}).get("never_call_kyo_user", True):
        text = re.sub(r"\bthe user\b", "Kyo", text, flags=re.IGNORECASE)
    return text


def normalize_actor_id(actor_id: str, source: str = "") -> str:
    """Return a stable actor id without converting Kyo/Nan0 actions into each other."""
    raw = str(actor_id or "").strip()
    low = raw.lower()
    src = str(source or "").lower()
    if src.startswith("kyo") or low in {"kyo", "kayok", "kayo"}:
        return "kyo"
    if src in {"boot", "monologue", "proactive", "social_pressure", "vision_pressure"} or low in {"nan0", "nano"}:
        return "nan0"
    if "discord" in src:
        return low or "discord_friend"
    return low or "unknown"


def actor_perspective_contract(actor_id: str, source: str = "") -> Dict[str, Any]:
    """Small prompt-safe ownership contract for thought generation."""
    stable = normalize_actor_id(actor_id, source)
    if stable == "kyo":
        return {
            "source_actor_id": "kyo",
            "display_name": "Kyo",
            "actor_role": "Kyo is the one who spoke or acted.",
            "nan0_role": "Nan0 is the observer/reactor, not the actor who did Kyo's action.",
            "ownership_rule": "If Kyo says 'I watched/was/did', that action belongs to Kyo. Nan0 may react to it but must not remember it as Nan0 doing it.",
            "pronouns": ["she", "her"],
        }
    if stable == "nan0":
        return {
            "source_actor_id": "nan0",
            "display_name": "Nan0",
            "actor_role": "Nan0 is the actor/source of this internal event.",
            "nan0_role": "Nan0 may use I/me for this event.",
            "ownership_rule": "Internal/boot/monologue actions belong to Nan0, not Kyo.",
            "pronouns": ["she", "her"],
        }
    return {
        "source_actor_id": stable,
        "display_name": actor_id or stable or "unknown",
        "actor_role": "This external actor spoke or acted.",
        "nan0_role": "Nan0 is the observer/reactor unless the event explicitly says Nan0 acted.",
        "ownership_rule": "Do not convert another actor's first-person statement into Nan0's memory or action.",
        "pronouns": [],
    }
