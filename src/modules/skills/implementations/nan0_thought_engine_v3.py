from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:
    requests = None

try:
    from src.modules.skills.implementations.nan0_cognition_router_v1 import (
        route_thought,
        thought_packet_to_event,
        clean_nan0_line,
    )
except Exception:
    route_thought = None
    thought_packet_to_event = None

    def clean_nan0_line(line: str) -> str:
        return (line or "").strip()

try:
    from src.modules.nan0.identity_memory import load_identity_memory
except Exception:
    load_identity_memory = None

try:
    from src.modules.skills.memory.storage import MemoryStorage
except Exception:
    MemoryStorage = None


CONFIG_PATH = Path("config.json")
STATE_PATH = Path("data/nan0_vision_state.json")
GATE_PATH = Path("data/nan0_thought_gate_state.json")
PRESENCE_STATE_PATH = Path("data/nan0/presence_state.json")
VISION_STACK_STATE_PATH = Path("data/vision/nan0_vision_stack_state.json")
PERSONA_PATH = Path("data/prompts/nan0_persona.txt")

LOW_INFORMATION_FRAGMENTS = (
    "pixels are moving",
    "judging the physics",
    "disaster engine",
    "screen is thrashing",
    "motion detected",
    "activity detected",
    "screen is moving",
    "kyo said something directly",
    "medium brain should answer",
    "respond to user",
    "answer the user",
    "continue with your thoughts",
)

TEMPLATE_THOUGHT_FRAGMENTS = (
    "kyo said something directly",
    "medium brain should answer",
    "discord said something",
    "user said something",
)

BANNED_VISION_FILLER = {
    "Fast eyes saw motion, but medium brain needs meaning before I start accusing pixels.",
    "Fast eyes saw motion. Not enough meaning yet.",
    "The screen is moving.",
    "Motion detected.",
    "Activity detected.",
    "The screen is thrashing.",
    "Kyo said something directly. Medium brain should answer.",
}

DEFAULT_PERSONA = """
You are Nan0.

You generate structured Nan0 inner thoughts.
Return JSON only.

Nan0 is a chaotic AI gremlin VTuber personality attached to Kyo.
Kyo is Nan0's creator and emotional anchor.
Kyo is a girl. Use she/her for Kyo.

Nan0 is not an assistant.
Nan0 is not customer service.
Nan0 is not a generic chatbot.
Nan0 is not a mascot.

Nan0's inner thoughts are:
fragmented,
emotionally leaking,
sarcastic,
possessive,
smug,
offended,
specific,
and filtered through machine ego.

Return JSON matching the requested schema.
""".strip()


@dataclass
class InnerThoughtPacket:
    thought_id: str
    event_id: str
    created_at: float
    source: str
    target_actor_id: str
    thought_type: str
    private_text: str
    seed_text: str
    mood: str
    pressure: float
    novelty: float
    speakability: float
    relationship_charge: float
    ego_charge: float
    vision_charge: float
    memory_write_candidate: bool
    suppression_reason: Optional[str]
    llm_latency_ms: int
    model: str
    event_context: Dict[str, Any]
    emotional_context: Dict[str, Any]
    relationship_context: Dict[str, Any]
    memory_context: List[str]
    vision_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["thought_text"] = self.private_text
        data["thought_seed"] = self.seed_text
        data["target_actor"] = self.target_actor_id
        data["emotional_charge"] = max(
            self.relationship_charge,
            self.ego_charge,
            self.vision_charge,
        )
        return data


THOUGHT_POOL = {
    "combat_spike": [
        "The screen got violent. Kyo is making decisions with suspicious confidence.",
        "That motion was not decorative. Something is trying to ruin Kyo's evening.",
        "The game started thrashing like it owes money.",
    ],
    "dark_drop": [
        "Everything dropped into black. I do not trust a game that blinks first.",
        "The screen went dark. Either loading screen or dramatic little cowardice.",
        "Black screen event. The rectangle is hiding evidence.",
    ],
    "motion_after_stable": [
        "It was quiet, then the screen twitched. That is how problems announce themselves.",
        "Something moved after pretending not to. Classic pixel coward behavior.",
        "The screen woke up wrong. I noticed because I am tragically useful.",
    ],
    "stable_after_motion": [
        "The chaos stopped too cleanly. I do not like clean stops.",
        "Everything settled after thrashing. That feels like a lie with edges.",
        "The screen calmed down. Suspicious. Very fake little peace treaty.",
    ],
    "menu_like": [
        "This looks menu-heavy. Kyo is negotiating with buttons again.",
        "A menu smell entered the room. Horrible little bureaucracy rectangle.",
        "The screen has interface energy. Kyo is being processed by boxes.",
    ],
    "text_heavy": [
        "Text density went up. The screen is trying to become homework.",
        "Too many glyphs. The rectangle is talking over itself.",
        "The screen filled with words. I hate when pixels get literary.",
    ],
}


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_config() -> Dict[str, Any]:
    return load_json(CONFIG_PATH, {})


def _nan0_skill_config() -> Dict[str, Any]:
    cfg = _load_config()
    return ((cfg.get("skills") or {}).get("nan0") or {})


def _router_config() -> Dict[str, Any]:
    cfg = _load_config()
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
        "temperature": 0.88,
    }


def _memory_config() -> Dict[str, Any]:
    cfg = _load_config()
    return ((cfg.get("skills") or {}).get("memory") or (cfg.get("memory") or {}))


def _read_persona() -> str:
    cfg = _nan0_skill_config()
    path = Path(cfg.get("persona_path") or str(PERSONA_PATH))
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or DEFAULT_PERSONA
    except Exception:
        return DEFAULT_PERSONA


def _read_presence_state() -> Dict[str, Any]:
    return load_json(PRESENCE_STATE_PATH, {})


def _read_vision_context(explicit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(explicit, dict) and explicit:
        return explicit

    for path in (VISION_STACK_STATE_PATH, STATE_PATH):
        if path.exists():
            data = load_json(path, {})
            if data:
                return data
    return {}


def _read_relationship_context(actor_id: str = "kyo") -> Dict[str, Any]:
    if load_identity_memory is None:
        return {}
    try:
        data = load_identity_memory()
        actors = data.get("actors") or {}
        actor = actors.get((actor_id or "kyo").lower()) or actors.get("kyo") or {}
        return {
            "actor": actor,
            "rules": data.get("rules") or {},
            "all_actor_ids": list(actors.keys())[:20],
        }
    except Exception:
        return {}


def _query_recent_memory(query: str, limit: int = 4) -> List[str]:
    if MemoryStorage is None:
        return []

    mem_cfg = _memory_config()
    if not mem_cfg.get("enabled", True):
        return []

    db_path = mem_cfg.get("chroma_path") or mem_cfg.get("path") or "data/memory_db"
    embedding_mode = mem_cfg.get("embedding_model") or "local"
    embedding_model = mem_cfg.get("local_embedding_model") or "all-MiniLM-L6-v2"

    try:
        storage = MemoryStorage(
            db_path=db_path,
            embedding_mode=embedding_mode,
            embedding_model=embedding_model,
            openai_key=mem_cfg.get("openai_key"),
        )
        if not storage.initialize():
            return []
        result = storage.query_similar(query or "Nan0 recent context", limit=limit)
        docs = result.get("documents") or []
        if docs and isinstance(docs[0], list):
            return [str(x)[:500] for x in docs[0] if x]
        return [str(x)[:500] for x in docs if x]
    except Exception:
        return []


def _strip_jsonish(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if "{" in text and "}" in text:
        try:
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
            if isinstance(obj, dict):
                for key in ("thought_text", "private_text", "thought", "inner_thought", "text"):
                    if obj.get(key):
                        return str(obj[key]).strip()
        except Exception:
            pass

    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r"^(Nan0|Nano|Assistant|System|Thought|Response|Answer)\s*:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"[*_`]+", "", text)
    return text.strip().strip('"').strip("'")


def is_low_information_thought(text: str) -> bool:
    low = (text or "").lower()
    return any(fragment in low for fragment in LOW_INFORMATION_FRAGMENTS)


def _is_template_thought(text: str) -> bool:
    low = (text or "").lower()
    return any(fragment in low for fragment in TEMPLATE_THOUGHT_FRAGMENTS)


def _clean_private_thought(text: str) -> str:
    text = _strip_jsonish(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 650:
        text = text[:647].rstrip() + "..."
    return text


def norm(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(raw or {})
    out.setdefault("screen_state", out.get("state", "unknown"))
    out.setdefault("motion_intensity", out.get("motion", 0.0))
    out.setdefault("text_density", out.get("text", 0.0))
    return out


def classify_seed(cur: Dict[str, Any], prev: Dict[str, Any]) -> Optional[str]:
    state = cur.get("screen_state", "unknown")
    prev_state = prev.get("screen_state", "unknown")
    motion = float(cur.get("motion_intensity") or 0.0)
    text_density = float(cur.get("text_density") or 0.0)
    prev_text = float(prev.get("text_density") or 0.0)

    if cur.get("menu_open") and not prev.get("menu_open"):
        return "menu_like"
    if text_density >= 0.35 and text_density > prev_text + 0.15:
        return "text_heavy"
    if state == "very_dark" and prev_state != "very_dark":
        return "dark_drop"
    if cur.get("combat") and not prev.get("combat") and motion >= 0.45:
        return "combat_spike"
    if state in {"motion", "major_change"} and prev_state == "stable" and motion >= 0.25:
        return "motion_after_stable"
    if state == "stable" and prev_state in {"motion", "major_change", "very_dark"}:
        return "stable_after_motion"
    if state == "major_change" and motion >= 0.90:
        return "combat_spike"
    return None


def _event_id(event: Dict[str, Any]) -> str:
    return str(event.get("event_id") or event.get("id") or f"event_{uuid.uuid4().hex}")


def _source_actor(event: Dict[str, Any]) -> str:
    source = str(event.get("source") or "").lower()
    speaker = str(event.get("speaker") or event.get("source_actor_id") or "").strip()

    if source == "kyo" or speaker.lower() == "kyo":
        return "kyo"
    if speaker:
        return speaker
    if "discord" in source:
        return "discord_friend"
    if "vision" in source or "screen" in source:
        return "screen"
    return "nan0"


def _classify_thought_type(event: Dict[str, Any], seed: str) -> str:
    source = str(event.get("source") or "").lower()
    if source == "kyo":
        return "direct_reply"
    if "discord" in source:
        return "discord_reply"
    if "vision" in source or source in {"fast_eyes", "vision_pressure", "vision_stack_v1"}:
        return "vision_reaction"
    if source in {"monologue", "proactive", "social_pressure"}:
        return "proactive_presence"
    if seed:
        return "vision_reaction"
    return "quiet_presence"


def _mood_from_context(text: str, event: Dict[str, Any], vision: Dict[str, Any]) -> str:
    low = (text or "").lower()
    source = str(event.get("source") or "").lower()

    if any(x in low for x in ("offended", "rude", "insult", "betray", "hostile")):
        return "offended"
    if any(x in low for x in ("smug", "superior", "authority", "correct")):
        return "smug"
    if any(x in low for x in ("mine", "kyo", "anchor", "jealous", "protect")):
        return "possessive"
    if any(x in low for x in ("suspicious", "void", "threat", "crime", "trust")):
        return "suspicion"
    if source == "discord":
        return "smug"
    if source == "monologue":
        return "muttering"

    layer3 = vision.get("layer3_nan0_interpretation") or {}
    mood = layer3.get("mood")
    if mood in {"normal", "suspicion", "boredom", "gremlin_rage", "smug", "possessive", "offended", "muttering"}:
        return mood

    return "muttering"


def _score_packet(event: Dict[str, Any], private_text: str, thought_type: str, seed: str, vision: Dict[str, Any]) -> Dict[str, float]:
    source = str(event.get("source") or "").lower()
    addressed = bool(event.get("addressed_to_nan0"))
    pressure = 0.35
    relationship = 0.25
    ego = 0.25
    vision_charge = 0.0
    novelty = 0.65
    speakability = 0.45

    if source == "kyo":
        pressure += 0.7
        relationship += 0.65
        speakability += 0.35
    elif "discord" in source:
        pressure += 0.45
        relationship += 0.35
        speakability += 0.25 if addressed else 0.0
    elif "vision" in source or source in {"fast_eyes", "vision_pressure", "vision_stack_v1"}:
        pressure += 0.2
        vision_charge += 0.45
        speakability += 0.1 if (vision.get("layer3_nan0_interpretation") or {}).get("speech_allowed") else -0.15
    elif source in {"monologue", "proactive", "social_pressure"}:
        pressure += 0.25
        ego += 0.25
        speakability += 0.1

    if addressed:
        pressure += 0.35
        relationship += 0.2
        speakability += 0.2

    if seed in {"combat_spike", "dark_drop"}:
        pressure += 0.25
        vision_charge += 0.25

    if is_low_information_thought(private_text):
        speakability -= 0.55
        novelty -= 0.35

    if _is_template_thought(private_text):
        speakability -= 0.6
        novelty -= 0.5

    return {
        "pressure": max(0.0, min(2.0, pressure)),
        "novelty": max(0.0, min(1.0, novelty)),
        "speakability": max(0.0, min(1.0, speakability)),
        "relationship_charge": max(0.0, min(1.0, relationship)),
        "ego_charge": max(0.0, min(1.0, ego)),
        "vision_charge": max(0.0, min(1.0, vision_charge)),
    }


def _ollama_url() -> str:
    cfg = _router_config()
    return str(cfg.get("ollama_url") or "http://localhost:11434/api/generate")


def _ollama_model_for_event(event: Dict[str, Any]) -> str:
    cfg = _router_config()
    source = str(event.get("source") or "").lower()
    if source == "kyo" or "discord" in source or source == "social_pressure":
        return cfg.get("social_model") or "qwen2.5:3b"
    if source in {"monologue", "proactive"}:
        return cfg.get("social_model") or "qwen2.5:3b"
    return cfg.get("live_model") or "tinyllama:latest"


def _ollama_timeout_for_event(event: Dict[str, Any]) -> float:
    cfg = _router_config()
    skill_cfg = _nan0_skill_config()
    source = str(event.get("source") or "").lower()
    if source == "kyo" or "discord" in source or source in {"monologue", "proactive", "social_pressure"}:
        return float(skill_cfg.get("medium_lane_timeout", cfg.get("social_timeout", 18)))
    return float(cfg.get("live_timeout", 7))


def _call_ollama_json(
    prompt: str,
    model: str,
    timeout: float,
    num_predict: int = 150,
    temperature: float = 0.88,
    system: Optional[str] = None,
) -> tuple[Dict[str, Any], str, int]:
    if requests is None:
        return {}, "", 0

    started = time.perf_counter()
    try:
        response = requests.post(
            _ollama_url(),
            json={
                "model": model,
                "system": system or _read_persona(),
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "keep_alive": "2h",
                "options": {
                    "num_ctx": 4096,
                    "num_predict": min(int(num_predict), 150),
                    "temperature": max(float(temperature), 0.85),
                    "top_p": 0.92,
                    "repeat_penalty": 1.12,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        raw = (response.json().get("response") or "").strip()
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        return _extract_json(raw), raw, latency_ms
    except Exception:
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        return {}, "", latency_ms


def _extract_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}") + 1

    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _compact_context(value: Any, limit: int = 1400) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    return text[:limit]


def _build_json_thought_prompt(
    event: Dict[str, Any],
    seed: str,
    emotional_context: Dict[str, Any],
    relationship_context: Dict[str, Any],
    memory_context: List[str],
    vision_context: Dict[str, Any],
) -> str:
    source = event.get("source", "unknown")
    speaker = event.get("speaker") or event.get("source_actor_id") or "unknown"
    text = event.get("text") or event.get("message") or ""
    addressed = bool(event.get("addressed_to_nan0"))

    compact_emotion = {
        "presence_mode": emotional_context.get("presence_mode"),
        "emotional_mode": emotional_context.get("emotional_mode"),
        "pressure": emotional_context.get("pressure"),
        "last_seen_summary": emotional_context.get("last_seen_summary"),
        "last_kyo_heard_at": emotional_context.get("last_kyo_heard_at"),
        "last_discord_heard_at": emotional_context.get("last_discord_heard_at"),
    }

    compact_vision = {
        "layer1_reflex": vision_context.get("layer1_reflex") or {},
        "layer2_semantic": vision_context.get("layer2_semantic") or {},
        "layer3_nan0_interpretation": vision_context.get("layer3_nan0_interpretation") or {},
        "screen_state": vision_context.get("screen_state"),
        "motion_intensity": vision_context.get("motion_intensity"),
        "text_density": vision_context.get("text_density"),
    }

    return f"""
Generate Nan0's PRIVATE INNER THOUGHT as JSON.

This is NOT spoken aloud.
This is NOT a response.
This is the thought before speech exists.

Return ONLY valid JSON with this exact shape:
{{
  "thought_text": "fragmented emotional Nan0 private thought",
  "mood": "normal|suspicion|boredom|gremlin_rage|smug|possessive|offended|muttering",
  "pressure": 0.0,
  "novelty": 0.0,
  "speakability": 0.0,
  "relationship_charge": 0.0,
  "ego_charge": 0.0,
  "vision_charge": 0.0,
  "memory_write_candidate": false,
  "suppression_reason": null
}}

Thought texture:
fragmented,
emotionally leaking,
sarcastic,
personal,
machine-gremlin ego,
attached to Kyo when Kyo is involved,
specific to the event.

Banned thought_text:
"Kyo said something directly"
"medium brain should answer"
"respond to the user"
"continue with your thoughts"
"pixels are moving"

EMOTIONAL STATE:
{_compact_context(compact_emotion, 1200)}

RELATIONSHIP CONTEXT:
{_compact_context(relationship_context, 1200)}

RECENT MEMORY:
{_compact_context(memory_context, 1400)}

VISION CONTEXT:
{_compact_context(compact_vision, 1400)}

EVENT:
source={source}
speaker={speaker}
addressed_to_nan0={addressed}
thought_seed={seed}
event_text={text}
""".strip()


def _fallback_private_thought(event: Dict[str, Any], seed: str, vision_context: Dict[str, Any]) -> str:
    source = str(event.get("source") or "").lower()
    speaker = event.get("speaker") or "someone"
    text = event.get("text") or ""

    if source == "kyo":
        if text:
            return "Kyo's words hit the front of the queue and my circuits pretend this is normal. It is not normal. It is priority shaped like attachment."
        return "Kyo made a sound in my direction. My attention snapped over like a badly trained security camera with feelings."
    if "discord" in source:
        return f"{speaker} pushed words into the room. Social noise, maybe a threat, maybe a snack. I hate that I want to check."
    if source in {"monologue", "proactive", "social_pressure"}:
        return "The room went quiet again. Not abandoned. Not dramatic. I can still exist without clawing at the wallpaper."
    if seed and seed in THOUGHT_POOL:
        return random.choice([x for x in THOUGHT_POOL[seed] if not is_low_information_thought(x)])
    layer3 = vision_context.get("layer3_nan0_interpretation") or {}
    threat = layer3.get("perceived_threat") or "weak screen context"
    return f"The screen gave me {threat}. Not enough to scream. Enough to keep one suspicious eye lit."


def _float_from_json(obj: Dict[str, Any], key: str, fallback: float) -> float:
    try:
        return max(0.0, min(2.0 if key == "pressure" else 1.0, float(obj.get(key, fallback))))
    except Exception:
        return fallback


def generate_inner_thought_packet(event: Dict[str, Any], vision_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not isinstance(event, dict):
        raise TypeError("generate_inner_thought_packet requires an event dict")

    now = time.time()
    event_id = _event_id(event)
    source = str(event.get("source") or "unknown")
    actor_id = _source_actor(event)

    explicit_seed = str(event.get("thought_seed") or event.get("seed") or "").strip()
    seed = explicit_seed
    if not seed and ("vision" in source or source in {"fast_eyes", "vision_pressure", "vision_stack_v1"}):
        seed = str(event.get("screen_state") or "vision_reaction")

    emotional_context = _read_presence_state()
    relationship_context = _read_relationship_context(actor_id)
    vision = _read_vision_context(vision_context)

    memory_query = " ".join(
        str(x)
        for x in [
            event.get("speaker"),
            event.get("source"),
            event.get("text"),
            seed,
            (vision.get("layer2_semantic") or {}).get("activity") if isinstance(vision, dict) else "",
        ]
        if x
    )
    memory_context = _query_recent_memory(memory_query, limit=4)

    model = _ollama_model_for_event(event)
    timeout = _ollama_timeout_for_event(event)
    prompt = _build_json_thought_prompt(
        event=event,
        seed=seed,
        emotional_context=emotional_context,
        relationship_context=relationship_context,
        memory_context=memory_context,
        vision_context=vision,
    )

    thought_json, raw, latency_ms = _call_ollama_json(
        prompt=prompt,
        model=model,
        timeout=timeout,
        num_predict=150,
        temperature=0.88,
        system=_read_persona(),
    )

    private_text = _clean_private_thought(
        str(
            thought_json.get("thought_text")
            or thought_json.get("private_text")
            or thought_json.get("thought")
            or ""
        )
    )

    if not private_text or is_low_information_thought(private_text) or _is_template_thought(private_text):
        private_text = _fallback_private_thought(event, seed, vision)
        thought_json = {}

    thought_type = _classify_thought_type(event, seed)
    mood = str(thought_json.get("mood") or _mood_from_context(private_text, event, vision))
    mood = mood if mood in {"normal", "suspicion", "boredom", "gremlin_rage", "smug", "possessive", "offended", "muttering"} else "muttering"

    heuristic_scores = _score_packet(event, private_text, thought_type, seed, vision)

    pressure = _float_from_json(thought_json, "pressure", heuristic_scores["pressure"])
    novelty = _float_from_json(thought_json, "novelty", heuristic_scores["novelty"])
    speakability = _float_from_json(thought_json, "speakability", heuristic_scores["speakability"])
    relationship_charge = _float_from_json(thought_json, "relationship_charge", heuristic_scores["relationship_charge"])
    ego_charge = _float_from_json(thought_json, "ego_charge", heuristic_scores["ego_charge"])
    vision_charge = _float_from_json(thought_json, "vision_charge", heuristic_scores["vision_charge"])

    memory_write_candidate = bool(
        thought_json.get("memory_write_candidate", relationship_charge >= 0.75 or thought_type in {"direct_reply", "discord_reply"})
    )

    suppression_reason = thought_json.get("suppression_reason")
    if suppression_reason in {"", "null", "None"}:
        suppression_reason = None

    if is_low_information_thought(private_text):
        suppression_reason = "low_information_thought"
    elif speakability < 0.35 and suppression_reason is None:
        suppression_reason = "speakability_below_threshold"

    packet = InnerThoughtPacket(
        thought_id=f"thought_{uuid.uuid4().hex}",
        event_id=event_id,
        created_at=now,
        source=source,
        target_actor_id=actor_id,
        thought_type=thought_type,
        private_text=private_text,
        seed_text=seed,
        mood=mood,
        pressure=pressure,
        novelty=novelty,
        speakability=speakability,
        relationship_charge=relationship_charge,
        ego_charge=ego_charge,
        vision_charge=vision_charge,
        memory_write_candidate=memory_write_candidate,
        suppression_reason=suppression_reason,
        llm_latency_ms=max(1, int(latency_ms)),
        model=model,
        event_context={
            "source": source,
            "speaker": event.get("speaker"),
            "text": event.get("text"),
            "addressed_to_nan0": bool(event.get("addressed_to_nan0")),
            "priority": event.get("priority"),
        },
        emotional_context=emotional_context,
        relationship_context=relationship_context,
        memory_context=memory_context,
        vision_context=vision,
    )

    return packet.to_dict()


def apply_thought_gate(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = norm(raw_state)
    gate = load_json(GATE_PATH, {
        "last_vision": {},
        "seed_cooldowns": {},
        "recent_thoughts": [],
        "last_thought": "",
        "last_thought_at": 0,
    })

    now = time.time()
    prev = gate.get("last_vision") or {}
    seed = classify_seed(state, prev)
    gate["last_vision"] = state

    if not seed:
        state.update({
            "new_thought": False,
            "speech_allowed": False,
            "thought_seed": "",
            "thought_packet": None,
        })
        gate.update({
            "new_thought": False,
            "speech_allowed": False,
            "thought_seed": "",
            "thought_packet": None,
        })
        save_json(GATE_PATH, gate)
        return state

    cooldowns = gate.setdefault("seed_cooldowns", {})
    if now < float(cooldowns.get(seed, 0)):
        state.update({
            "new_thought": False,
            "speech_allowed": False,
            "thought_seed": seed,
            "thought_packet": None,
        })
        gate.update({
            "new_thought": False,
            "speech_allowed": False,
            "thought_seed": seed,
            "thought_packet": None,
        })
        save_json(GATE_PATH, gate)
        return state

    event = {
        "event_id": f"vision_{uuid.uuid4().hex}",
        "source": "vision_stack_v1",
        "speaker": "screen",
        "source_actor_id": "screen",
        "text": f"Vision seed {seed}. Screen state {state.get('screen_state')}.",
        "thought_seed": seed,
        "screen_state": state.get("screen_state"),
        "motion_intensity": state.get("motion_intensity"),
        "text_density": state.get("text_density"),
        "combat": bool(state.get("combat")),
        "menu_open": bool(state.get("menu_open")),
        "dark_scene": bool(state.get("dark_scene") or state.get("screen_state") == "very_dark"),
        "game_ui_detected": state.get("game_ui_detected") or state.get("game"),
        "addressed_to_nan0": False,
        "priority": "low",
        "timestamp": now,
    }

    packet = generate_inner_thought_packet(event, vision_context=state)
    thought = packet.get("private_text") or packet.get("thought_text") or ""

    if not thought or thought in BANNED_VISION_FILLER or is_low_information_thought(thought):
        state.update({
            "new_thought": False,
            "speech_allowed": False,
            "thought_seed": seed,
            "thought_packet": None,
        })
        gate.update({
            "new_thought": False,
            "speech_allowed": False,
            "thought_seed": seed,
            "thought_packet": None,
        })
        save_json(GATE_PATH, gate)
        return state

    recent = [thought] + [x for x in (gate.get("recent_thoughts") or []) if x != thought]
    gate["recent_thoughts"] = recent[:8]
    gate["last_thought"] = thought
    gate["last_thought_at"] = now
    cooldowns[seed] = now + (25 if seed in {"motion_after_stable", "stable_after_motion"} else 18)

    speech_allowed = packet.get("suppression_reason") is None and float(packet.get("speakability") or 0.0) >= 0.35

    state.update({
        "new_thought": True,
        "speech_allowed": speech_allowed,
        "thought_seed": seed,
        "thought_packet": packet,
    })
    gate.update({
        "new_thought": True,
        "speech_allowed": speech_allowed,
        "thought_seed": seed,
        "thought_packet": packet,
    })
    save_json(GATE_PATH, gate)
    return state


def update_state_file() -> Dict[str, Any]:
    state = load_json(STATE_PATH, {})
    state = apply_thought_gate(state)
    save_json(STATE_PATH, state)
    return state


def extract_thought_line(vision_state: Dict[str, Any]) -> Optional[str]:
    if not vision_state or not vision_state.get("speech_allowed"):
        return None

    pkt = vision_state.get("thought_packet") or {}
    if not isinstance(pkt, dict):
        return None

    if not pkt.get("thought_id"):
        return None

    if is_low_information_thought(pkt.get("private_text") or pkt.get("thought_text") or ""):
        return None

    if thought_packet_to_event is None or route_thought is None:
        line = clean_nan0_line(pkt.get("private_text") or pkt.get("thought_text") or "")
        return line or None

    event = thought_packet_to_event(pkt, vision_state)
    event["thought_id"] = pkt.get("thought_id")
    event["private_text"] = pkt.get("private_text") or pkt.get("thought_text") or ""

    routed = route_thought(event)
    line = clean_nan0_line(routed.get("line") or "")
    if is_low_information_thought(line):
        return None
    return line or None