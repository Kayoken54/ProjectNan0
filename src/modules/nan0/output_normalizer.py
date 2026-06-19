"""
Nan0 V7 output normalizer.

Hard rule:
Thought object != spoken line.

Every speech path should run through:
raw model output -> internal thought object -> speech compression layer -> output line
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from src.modules.nan0.runtime_guard import validate_cognition_text, validate_thought_packet

ALLOWED_MOODS = {
    "normal",
    "suspicion",
    "boredom",
    "gremlin_rage",
    "smug",
    "possessive",
    "offended",
    "muttering",
    "neutral",
}

MOOD_ALIASES = {
    "curiosity": "suspicion",
    "curious": "suspicion",
    "warmth": "possessive",
    "friendly": "normal",
    "friendliness": "normal",
    "anger": "gremlin_rage",
    "rage": "gremlin_rage",
    "annoyed": "offended",
    "sad": "muttering",
}

GENERIC_BAD_PATTERNS = [
    "how can i help",
    "as an ai",
    "i am just a program",
    "i don't have feelings",
    "i do not have feelings",
    "thanks for the message",
    "interesting question",
    "i understand",
    "certainly",
    "my algorithms grapple",
    "algorithms grapple",
    "discern its",
    "disconcerted by the unexpected query",
    "as a language model",
    "i don't possess",
    "i do not possess",
]

NAN0_MARKERS = [
    "room",
    "kyo",
    "hardware",
    "ssd",
    "hdd",
    "ram",
    "cpu",
    "gpu",
    "motherboard",
    "thermal",
    "lag",
    "running",
    "silence",
    "betrayal",
    "hostile",
    "machine",
    "wires",
    "attention",
    "not neutral",
]


def normalize_llm_output(
    raw: Any,
    target_actor: str = "kyo",
    fallback_mood: str = "normal",
    max_chars: int = 190,
) -> Dict[str, Any]:
    """
    Converts an LLM result into a ProjectBEA speech packet without inventing
    fallback speech. If the input is unusable, message is empty and the caller
    must suppress it.
    """
    thought = _coerce_to_thought_object(raw)

    mood = _normalize_mood(
        thought.get("primary_emotion")
        or thought.get("emotion")
        or thought.get("mood")
        or fallback_mood
    )

    target = thought.get("target_actor") or thought.get("target") or target_actor

    candidate = (
        thought.get("speech_line")
        or thought.get("spoken_line")
        or thought.get("message")
        or thought.get("thought_text")
        or thought.get("text")
        or ""
    )

    if _looks_like_json(candidate):
        nested = _coerce_to_thought_object(candidate)
        if nested:
            thought.update({f"nested_{k}": v for k, v in nested.items()})
            candidate = (
                nested.get("speech_line")
                or nested.get("spoken_line")
                or nested.get("message")
                or nested.get("thought_text")
                or nested.get("text")
                or candidate
            )
            mood = _normalize_mood(nested.get("primary_emotion") or nested.get("mood") or mood)
            target = nested.get("target_actor") or nested.get("target") or target

    line = _compress_to_nan0_line(str(candidate), mood=mood, target=str(target), thought=thought)
    line = _sanitize_line(line)
    line = _trim_line(line, max_chars=max_chars)
    content_valid, suppression_reason = validate_cognition_text(line)
    if not content_valid:
        line = ""

    return {
        "mood": mood,
        "message": line,
        "internal_thought": thought,
        "normalized_by": "nan0_output_normalizer_v7_prime_directive",
        "suppression_reason": None if content_valid else suppression_reason,
    }


def validate_output_candidate(
    origin_packet: Any,
    line: Any,
    thought_id: Any,
) -> Tuple[bool, str]:
    """Final read-only guard before a line can be handed to output/TTS."""
    valid, reason = validate_thought_packet(origin_packet)
    if not valid:
        return False, reason
    if str(origin_packet.get("thought_id") or "") != str(thought_id or ""):
        return False, "thought_origin_mismatch"
    return validate_cognition_text(
        line,
        source=origin_packet.get("source"),
        source_family=(origin_packet.get("event_context") or {}).get("source_family"),
        event_text=(origin_packet.get("event_context") or {}).get("text"),
    )

def normalize_mood_message(mood: str, message: Any, target_actor: str = "kyo") -> Tuple[str, str]:
    packet = normalize_llm_output(
        {"mood": mood, "message": message},
        target_actor=target_actor,
        fallback_mood=mood or "normal",
    )
    return packet["mood"], packet["message"]


def _coerce_to_thought_object(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}

    if isinstance(raw, dict):
        return dict(raw)

    text = str(raw).strip()
    if not text:
        return {}

    parsed = _parse_jsonish(text)
    if isinstance(parsed, dict):
        return parsed

    # Sometimes the model returns text around a JSON object.
    extracted = _extract_first_json_object(text)
    if extracted:
        parsed = _parse_jsonish(extracted)
        if isinstance(parsed, dict):
            return parsed

    return {"thought_text": text}


def _parse_jsonish(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    # Repair common truncated object endings just enough to avoid speech leaks.
    repaired = text.strip()
    if repaired.startswith("{") and not repaired.endswith("}"):
        repaired += "}"
        try:
            return json.loads(repaired)
        except Exception:
            pass

    return None


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Truncated JSON. Return from first brace so sanitization can handle it.
    return text[start:]


def _looks_like_json(value: Any) -> bool:
    text = str(value).strip()
    return text.startswith("{") or '"thought_text"' in text or '"emotional_charge"' in text


def _normalize_mood(value: Any) -> str:
    mood = str(value or "normal").strip().lower()
    mood = MOOD_ALIASES.get(mood, mood)
    if mood not in ALLOWED_MOODS:
        return "normal"
    if mood == "neutral":
        return "normal"
    return mood


def _compress_to_nan0_line(text: str, mood: str, target: str, thought: Dict[str, Any]) -> str:
    text = _strip_json_noise(text).strip()
    text = re.sub(r"^(thought|message|response|speech)\s*:\s*", "", text, flags=re.I).strip()

    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences:
        first = sentences[0].strip()
        if len(first) >= 12:
            return first

    return text

def _strip_json_noise(text: str) -> str:
    text = str(text)

    # Remove common dangling JSON field chunks.
    text = re.sub(r'"?(thought_text|message|primary_emotion|target_actor|memory_recall|new_grudge_formed|running_bit_callback|emotional_charge|speech_pressure)"?\s*:\s*', "", text, flags=re.I)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("null", "")
    text = text.replace("false", "").replace("true", "")
    text = text.replace('"', "")
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,:\n\t")


def _sanitize_line(line: str) -> str:
    line = _strip_json_noise(line).strip()

    for bad in GENERIC_BAD_PATTERNS:
        if bad in line.lower():
            return ""

    if "emotional_charge" in line or "speech_pressure" in line or "target_actor" in line:
        return ""
    if "{" in line or "}" in line:
        return ""

    return line

def _enforce_nan0_flavor(line: str, mood: str, target: str, thought: Dict[str, Any]) -> str:
    """No flavor appending. Nan0 flavor must come from thought/speech generation."""
    return line

def _trim_line(line: str, max_chars: int) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) <= max_chars:
        return line

    cut = line[:max_chars].rsplit(" ", 1)[0].strip()
    if not cut:
        cut = line[:max_chars].strip()
    return cut + "..."
