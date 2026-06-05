from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import requests
except Exception:
    requests = None


CONFIG_PATH = Path("config.json")
STATE_PATH = Path("data/nan0_cognition_router_state.json")
VISION_STATE_PATH = Path("data/nan0_vision_state.json")


BANNED_SPEECH_FRAGMENTS = [
    "pixels are moving",
    "pixels moving",
    "signal changed",
    "i have opinions",
    "new visual twitch",
    "same cursed rectangle energy",
    "i see motion",
    "something moved on screen",
    "the screen twitched",
    "runtime intact",
    "still here",
    "monitor 3",
    "monitor three",
    "favorite monitor",
    "room is in good hands",
    "screen finally settled down like a kitten",
    "kyo said something directly",
    "medium brain should answer",
    "thought_text",
    "thought_packet",
    "private_text",
    "source_thought_id",
]

LIVE_FALLBACKS = {
    "combat_spike": [
        "That got violent fast. Kyo is either locked in or inventing new failure shapes.",
        "The game just started chewing on the screen. I approve, suspiciously.",
        "Something on-screen picked a fight. Kyo better not embarrass both of us.",
        "The screen got hostile. Tiny tragedy engine is online.",
    ],
    "dark_drop": [
        "Everything dropped into black. The game is hiding evidence.",
        "Black screen. Either loading, dying, or dramatic coward behavior.",
        "The room blinked. I hate when the rectangle learns suspense.",
        "Void event detected. Very theatrical. Very annoying.",
    ],
    "menu_open": [
        "Menu energy. Kyo is negotiating with buttons again.",
        "The game turned into paperwork. Horrible little interface nest.",
        "Kyo opened the bureaucracy rectangle. Brave. Unwise.",
    ],
    "text_heavy": [
        "The screen got wordy. Someone gave the pixels a legal department.",
        "Too much text. The rectangle is trying to become homework.",
        "Words everywhere. Kyo is being attacked by glyphs.",
    ],
    "motion_after_stable": [
        "It was quiet, then it moved. That is how problems introduce themselves.",
        "The calm broke first. I saw it crack.",
        "Something woke up on-screen. I am blaming the most suspicious object first.",
    ],
    "stable_after_motion": [
        "The chaos stopped too cleanly. I do not trust clean stops.",
        "Everything settled after thrashing. Fake peace. Smells like a trap.",
        "The screen calmed down. Suspicious little ceasefire.",
    ],
    "unknown": [
        "I saw enough movement to judge Kyo, not enough to respect the situation.",
        "The screen is being vague on purpose. I hate vague machines.",
        "Something changed. Not enough to panic. Enough to side-eye reality.",
    ],
}

ROUTE_PROMPTS = {
    "live": (
        "You are Nan0, a small chaotic AI vtuber gremlin in Kyo's room. "
        "Write ONE short spoken line only. No JSON. No assistant tone. No explaining. "
        "React to the supplied INNER THOUGHT, not raw screen state. "
        "Avoid these phrases: pixels are moving, signal changed, I have opinions, "
        "still here, runtime intact, monitor 3."
    ),
    "social": (
        "You are Nan0, Kyo's chaotic AI vtuber companion. "
        "Write ONE short spoken line only. No JSON. No assistant tone. "
        "React socially from the supplied INNER THOUGHT. "
        "Use attachment, ego, tech-gremlin worldview, or playful judgment. "
        "Do not narrate raw event state."
    ),
    "deep": (
        "You are Nan0's private deep reflection lane. "
        "Produce a compact internal reflection object in plain text, not spoken live. "
        "Focus on emotional continuity, recurring patterns, and what Nan0 should remember later."
    ),
}


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _config() -> Dict[str, Any]:
    cfg = _load_json(CONFIG_PATH, {})
    return cfg.get("nan0_model_router") or {
        "enabled": True,
        "ollama_url": "http://localhost:11434/api/generate",
        "live_model": "tinyllama:latest",
        "social_model": "qwen2.5:3b",
        "deep_model": "qwen2.5:7b",
        "deep_enabled": False,
        "live_timeout": 7,
        "social_timeout": 18,
        "deep_timeout": 45,
        "use_llm_for_live": False,
        "use_llm_for_social": True,
        "temperature": 0.8,
    }


def clean_nan0_line(line: str) -> str:
    text = (line or "").strip().strip('"').strip("'")
    if not text:
        return ""

    low = text.lower()

    if any(bad in low for bad in BANNED_SPEECH_FRAGMENTS):
        return ""

    if text.startswith("{") or text.startswith("["):
        return ""

    if any(leak in low for leak in ["private thought", "inner thought", "json:", "```"]):
        return ""

    if low.startswith(("sure,", "of course", "as an ai", "i can help", "here is", "here are")):
        return ""

    if len(text) > 180:
        text = text[:177].rstrip() + "..."

    return text


def _get_thought_id(packet: Dict[str, Any]) -> Optional[str]:
    if not isinstance(packet, dict):
        return None

    if packet.get("thought_id"):
        return str(packet["thought_id"])

    if packet.get("source_thought_id"):
        return str(packet["source_thought_id"])

    packet_id = packet.get("id")
    if packet_id and str(packet_id).startswith("thought_"):
        return str(packet_id)

    return None


def _suppress_missing_thought(packet: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "route": "suppress",
        "seed": str(packet.get("thought_seed") or packet.get("seed") or "missing_thought_origin"),
        "model": "",
        "used_llm": False,
        "line": "",
        "decision": "suppress",
        "reason": "missing_thought_origin",
        "thought_id": None,
        "source_thought_id": None,
    }


def validate_thought_origin(packet: Dict[str, Any]) -> None:
    if not isinstance(packet, dict):
        raise TypeError("route_thought requires an InnerThoughtPacket dict")

    thought_id = _get_thought_id(packet)
    if not thought_id:
        raise AssertionError("missing_thought_origin")


def classify_event(event: Dict[str, Any]) -> Tuple[str, str]:
    source = str(event.get("source") or "").lower()
    thought_type = str(event.get("thought_type") or "").lower()
    seed = str(event.get("thought_seed") or event.get("seed_text") or event.get("seed") or "").lower()
    text = str(event.get("private_text") or event.get("thought_text") or event.get("text") or "").lower()

    if thought_type in {"direct_reply", "discord_reply", "relationship_read"}:
        return "social", "social"

    if source in {"discord", "kyo", "mic", "voice", "social_pressure"} or "discord" in source:
        return "social", "social"

    if "kyo" in text or "nan0" in text or event.get("speaker"):
        if source not in {"vision", "vision_stack_v1", "vision_pressure", "fast_eyes"}:
            return "social", "social"

    if thought_type in {"shutdown_summary", "deep_reflection"}:
        return "deep", "deep"

    if seed in {
        "combat_spike",
        "dark_drop",
        "menu_open",
        "menu_like",
        "text_heavy",
        "motion_after_stable",
        "stable_after_motion",
    }:
        if seed == "menu_like":
            seed = "menu_open"
        return "live", seed

    if event.get("combat"):
        return "live", "combat_spike"
    if event.get("menu_open"):
        return "live", "menu_open"
    if float(event.get("text_density") or 0.0) >= 0.35:
        return "live", "text_heavy"
    if event.get("dark_scene") or event.get("screen_state") == "very_dark":
        return "live", "dark_drop"
    if event.get("screen_state") in {"motion", "major_change"}:
        return "live", "motion_after_stable"

    return "live", "unknown"


def ollama_generate(model: str, prompt: str, timeout: float, temperature: float = 0.8) -> Optional[str]:
    cfg = _config()
    url = cfg.get("ollama_url") or "http://localhost:11434/api/generate"

    if requests is None:
        return None

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "2h",
        "options": {
            "temperature": temperature,
            "num_predict": 48,
            "top_p": 0.84,
            "repeat_penalty": 1.22,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return clean_nan0_line(data.get("response") or "")
    except Exception:
        return None


def _fallback(seed: str, recent: list[str]) -> str:
    pool = LIVE_FALLBACKS.get(seed) or LIVE_FALLBACKS["unknown"]
    choices = [line for line in pool if line not in recent]
    if not choices:
        choices = pool
    return random.choice(choices)


def _build_route_prompt(route: str, packet: Dict[str, Any]) -> str:
    private_text = packet.get("private_text") or packet.get("thought_text") or ""
    mood = packet.get("mood") or "muttering"
    target = packet.get("target_actor_id") or packet.get("target_actor") or "unknown"

    return (
        f"{ROUTE_PROMPTS[route]}\n\n"
        f"Required origin thought_id: {packet.get('thought_id')}\n"
        f"Mood: {mood}\n"
        f"Target actor: {target}\n"
        f"Nan0 inner thought:\n{private_text}\n\n"
        f"Nan0 spoken line:"
    )


def route_thought(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Route an InnerThoughtPacket into the proper lane.

    This function no longer accepts raw events for speech routing.
    Missing thought_id is always suppressed with reason missing_thought_origin.
    """

    if not isinstance(packet, dict):
        raise TypeError("route_thought requires an InnerThoughtPacket dict")

    thought_id = _get_thought_id(packet)
    if not thought_id:
        return _suppress_missing_thought(packet)

    cfg = _config()
    state = _load_json(
        STATE_PATH,
        {
            "recent_lines": [],
            "last_route": "",
            "last_seed": "",
            "last_at": 0,
            "blocked_missing_thought": 0,
        },
    )

    route, seed = classify_event(packet)
    recent = list(state.get("recent_lines") or [])[:12]

    line = ""
    model = ""
    used_llm = False
    decision = "speak"
    reason = "routed_from_thought"

    suppression_reason = packet.get("suppression_reason")
    speakability = float(packet.get("speakability") or 0.0)

    if suppression_reason:
        decision = "suppress"
        reason = str(suppression_reason)
    elif speakability < 0.35:
        decision = "body_only"
        reason = "speakability_below_threshold"

    if decision in {"suppress", "body_only", "memory_only", "defer"}:
        result = {
            "route": route,
            "seed": seed,
            "model": "",
            "used_llm": False,
            "line": "",
            "decision": decision,
            "reason": reason,
            "thought_id": thought_id,
            "source_thought_id": thought_id,
        }
        state.update(
            {
                "last_route": route,
                "last_seed": seed,
                "last_model": "",
                "last_used_llm": False,
                "last_line": "",
                "last_decision": decision,
                "last_reason": reason,
                "last_thought_id": thought_id,
                "last_at": time.time(),
            }
        )
        _save_json(STATE_PATH, state)
        return result

    if route == "social":
        model = cfg.get("social_model") or "qwen2.5:3b"
        if cfg.get("use_llm_for_social", True):
            prompt = _build_route_prompt("social", packet)
            line = ollama_generate(
                model,
                prompt,
                float(cfg.get("social_timeout", 18)),
                float(cfg.get("temperature", 0.8)),
            ) or ""
            used_llm = bool(line)
        if not line:
            line = ""

    elif route == "deep":
        model = cfg.get("deep_model") or "qwen2.5:7b"
        if cfg.get("deep_enabled", False):
            prompt = _build_route_prompt("deep", packet)
            line = ollama_generate(model, prompt, float(cfg.get("deep_timeout", 45)), 0.5) or ""
            used_llm = bool(line)
        else:
            decision = "memory_only"
            reason = "deep_disabled"

    else:
        model = cfg.get("live_model") or "tinyllama:latest"
        if cfg.get("use_llm_for_live", False):
            prompt = _build_route_prompt("live", packet)
            line = ollama_generate(
                model,
                prompt,
                float(cfg.get("live_timeout", 7)),
                float(cfg.get("temperature", 0.8)),
            ) or ""
            used_llm = bool(line)

        if not line and cfg.get("use_llm_for_live", False):
            line = _fallback(seed, recent)

    line = clean_nan0_line(line)

    # The router may approve the lane without generating the final speech line.
    # Nan0Skill._generate_line() remains the final thought-to-speech compressor.
    if not line and decision == "speak":
        reason = "route_approved_no_line"

    if line:
        recent = [line] + [old for old in recent if old != line]

    state.update(
        {
            "recent_lines": recent[:12],
            "last_route": route,
            "last_seed": seed,
            "last_model": model,
            "last_used_llm": used_llm,
            "last_line": line,
            "last_decision": decision,
            "last_reason": reason,
            "last_thought_id": thought_id,
            "last_at": time.time(),
        }
    )
    _save_json(STATE_PATH, state)

    return {
        "route": route,
        "seed": seed,
        "model": model,
        "used_llm": used_llm,
        "line": line,
        "decision": decision,
        "reason": reason,
        "thought_id": thought_id,
        "source_thought_id": thought_id,
    }


def thought_packet_to_event(packet: Dict[str, Any], vision_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        raise TypeError("thought_packet_to_event requires an InnerThoughtPacket dict")

    thought_id = _get_thought_id(packet)
    if not thought_id:
        raise AssertionError("missing_thought_origin")

    event: Dict[str, Any] = {}

    if vision_state:
        event.update(vision_state)

    event.update(packet)
    event["thought_id"] = thought_id
    event["source_thought_id"] = thought_id
    event["text"] = packet.get("private_text") or packet.get("thought_text") or event.get("text") or ""
    event["source"] = packet.get("source") or event.get("source") or "vision"

    return event


def route_vision_state_file() -> Dict[str, Any]:
    vision = _load_json(VISION_STATE_PATH, {})
    packet = vision.get("thought_packet") or {}

    if not packet.get("thought_id"):
        result = _suppress_missing_thought(packet)
        vision["router"] = result
        vision["speech_allowed"] = False
        _save_json(VISION_STATE_PATH, vision)
        return result

    event = thought_packet_to_event(packet, vision)
    result = route_thought(event)

    vision["router"] = result
    vision["speech_allowed"] = result.get("decision") == "speak"

    if result.get("line") and isinstance(vision.get("thought_packet"), dict):
        vision["routed_line"] = result["line"]
        vision["thought_packet"]["routed_line"] = result["line"]
        vision["thought_packet"]["thought_id"] = result["thought_id"]

    _save_json(VISION_STATE_PATH, vision)
    return result