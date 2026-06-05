from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional


STATE_PATH = Path("data/nan0/output_normalizer_state.json")
LOG_PATH = Path("data/logs/nan0_output_guard.jsonl")


BANNED_OUTPUT_FRAGMENTS = [
    "pixels are moving",
    "screen is thrashing",
    "judging the physics",
    "disaster engine",
    "motion detected",
    "activity detected",
    "monitor 3",
    "monitor three",
    "private_text",
    "thought_text",
    "thought packet",
    "source_thought_id",
    "inner thought",
    "as an ai",
    "how can i help",
    "happy to help",
]


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_log(record: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_thought_id(speech_packet: Dict[str, Any]) -> bool:
    """
    Sacred Architecture V2 invariant.

    Every speech packet must originate from a thought.

    Required:
        speech_packet["thought_id"]

    If missing:
        block speech immediately
        log SPEECH_BLOCKED_NO_THOUGHT
    """

    if not isinstance(speech_packet, dict):
        _append_log(
            {
                "timestamp": time.time(),
                "event": "SPEECH_BLOCKED_NO_THOUGHT",
                "reason": "packet_not_dict",
            }
        )
        return False

    thought_id = speech_packet.get("thought_id")

    if thought_id is None:
        _append_log(
            {
                "timestamp": time.time(),
                "event": "SPEECH_BLOCKED_NO_THOUGHT",
                "reason": "thought_id_missing",
                "packet": str(speech_packet)[:500],
            }
        )
        return False

    if not str(thought_id).strip():
        _append_log(
            {
                "timestamp": time.time(),
                "event": "SPEECH_BLOCKED_NO_THOUGHT",
                "reason": "thought_id_blank",
                "packet": str(speech_packet)[:500],
            }
        )
        return False

    return True


def normalize_llm_output(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()

    if not text:
        return ""

    text = re.sub(r"^(assistant|system|nan0)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)

    if text.startswith("{") or text.startswith("["):
        return ""

    low = text.lower()

    for fragment in BANNED_OUTPUT_FRAGMENTS:
        if fragment in low:
            return ""

    return text.strip()


def build_speech_packet(
    thought_id: str,
    line_text: str,
    mood: str,
    target_actor_id: str = "unknown",
    voice_enabled: bool = True,
    display_enabled: bool = True,
) -> Dict[str, Any]:

    return {
        "thought_id": thought_id,
        "line_text": normalize_llm_output(line_text),
        "mood": mood,
        "target_actor_id": target_actor_id,
        "voice_enabled": voice_enabled,
        "display_enabled": display_enabled,
        "avatar_state": mood,
    }


def normalize_speech_packet(packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(packet, dict):
        return None

    if not validate_thought_id(packet):
        return None

    text = normalize_llm_output(packet.get("line_text", ""))

    if not text:
        _append_log(
            {
                "timestamp": time.time(),
                "event": "SPEECH_BLOCKED_EMPTY_AFTER_NORMALIZATION",
                "thought_id": packet.get("thought_id"),
            }
        )
        return None

    normalized = dict(packet)
    normalized["line_text"] = text

    return normalized


def record_output(
    speech_packet: Dict[str, Any],
    destination: str = "tts",
) -> bool:

    if not validate_thought_id(speech_packet):
        return False

    packet = normalize_speech_packet(speech_packet)

    if packet is None:
        return False

    state = _load_json(
        STATE_PATH,
        {
            "last_output_at": 0,
            "last_thought_id": None,
            "last_line": "",
            "outputs": 0,
        },
    )

    state["last_output_at"] = time.time()
    state["last_thought_id"] = packet["thought_id"]
    state["last_line"] = packet["line_text"]
    state["outputs"] = int(state.get("outputs", 0)) + 1

    _save_json(STATE_PATH, state)

    _append_log(
        {
            "timestamp": time.time(),
            "event": "SPEECH_APPROVED",
            "destination": destination,
            "thought_id": packet["thought_id"],
            "line_text": packet["line_text"][:250],
            "mood": packet.get("mood"),
        }
    )

    return True