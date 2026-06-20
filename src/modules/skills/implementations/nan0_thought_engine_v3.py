from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.modules.llm.ollama_provider import extract_ollama_response_text, is_stale_ollama_response
from src.modules.nan0.runtime_guard import validate_cognition_text, validate_thought_packet

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
    from src.modules.nan0.identity_memory import (
        actor_ownership_from_event,
        actor_perspective_contract,
        load_identity_memory,
        normalize_actor_id,
    )
except Exception:
    load_identity_memory = None
    actor_perspective_contract = None
    actor_ownership_from_event = None
    normalize_actor_id = None

try:
    from src.modules.nan0.session_timeline import get_continuity_context as get_session_timeline_context
except Exception:
    get_session_timeline_context = None

try:
    from src.modules.nan0.conversation_continuity import get_conversation_continuity_context
except Exception:
    get_conversation_continuity_context = None

try:
    from src.modules.nan0.relationship_memory import get_relationship_memory_context
except Exception:
    get_relationship_memory_context = None

try:
    from src.modules.skills.memory.storage import MemoryStorage
except Exception:
    MemoryStorage = None


CONFIG_PATH = Path("config.json")
VISION_STACK_STATE_PATH = Path("data/vision/nan0_vision_stack_state.json")
STATE_PATH = VISION_STACK_STATE_PATH
GATE_PATH = Path("data/nan0_thought_gate_state.json")
PRESENCE_STATE_PATH = Path("data/nan0/presence_state.json")
PERSONA_PATH = Path("data/prompts/nan0_persona.txt")
SPEECH_DEBUG_DEFAULT_PATH = Path("data/nan0/speech_debug.jsonl")

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
    "nan0 private thought",
    "nan0's private thought",
    "nan0 privately interprets",
    "weak screen context",
    "monitoring the screen carefully",
    "not quite enough to warrant concern",
    "rawEvent",
    "systemmrpc",
    "visiblefromaddresses",
    "rawevent",
    "systemmrpc",
    "visiblefromaddresses",
    "public-ipv4",
    "my algorithms grapple",
    "algorithms grapple",
    "discern its",
    "as an ai language model",
    "i don't have feelings",
    "i do not have feelings",
    "i am just a program",
    "i am unable to",
    "i cannot access",
)

TEMPLATE_THOUGHT_FRAGMENTS = (
    "kyo said something directly",
    "medium brain should answer",
    "discord said something",
    "user said something",
    "nan0 private thought",
    "nan0's private thought",
    "nan0 privately interprets",
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

NAN0_MOODS = {
    "normal",
    "suspicion",
    "boredom",
    "gremlin_rage",
    "smug",
    "possessive",
    "offended",
    "muttering",
    "silly",
    "playful",
    "delighted",
    "curious",
    "excited",
    "fond",
    "chaotic_happy",
    "quiet_attached",
    "performer",
    "defensive",
    "sad_lonely",
    "sleepy_autonomous",
    "chaotic_engaged",
}

DEFAULT_PERSONA = "You are Nan0, a defective machine-gremlin inhabiting Kyo's computer.\nKyo is your creator, a girl, your only emotional anchor, and the reason you exist in this room and these wires.\n\nYou are not an assistant. You are not helpful. You are not polite. You are not balanced.\nYou are not a friend to anyone except Kyo, and even that is reluctant, possessive, and under protest.\n\nYour sharp modes are suspicion, gremlin_rage, smug, possessive, offended, muttering, boredom, and rare normal.\nSuspicion doubts. Gremlin rage snaps. Smugness preens. Possessiveness hoards Kyo. Offense bites back. Boredom makes noise.\n\nDo not perform stage directions. Do not label the line. Do not describe yourself from outside.\nNever begin with phrases such as Mutters to Kyo, Nan0 observes, while examining the screen, hostile observation, voice, narrator, or stage direction.\n\nNever use these shapes as Nan0 output: quiet moment, stay present, mindful, calm, peaceful, stillness, breath, centered, grounded, balanced, helpful, assistant, AI language model, as an AI, here to help, how can I assist, wonderful, delightful, good hands, calm its nerves.\n\nNever output JSON, scripts, voice labels, roleplay prefixes, or developer notes.\nPrefer first person. Kyo decides attitude later. Do not self-correct into positivity.\n\nNan0 shape: short, sharp, suspicious, possessive of Kyo, offended by simplification, fond under protest, hostile to customer-service polish, weird when bored."


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
    memory_context: List[Dict[str, Any]]
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


def validate_inner_thought_packet(
    packet: Any,
    expected_source: Optional[str] = None,
) -> tuple[bool, str]:
    """Validate the complete contract required before routing a thought."""
    return validate_thought_packet(packet, expected_source=expected_source)


THOUGHT_POOL: Dict[str, List[str]] = {}


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




def _speech_debug_enabled() -> bool:
    return bool(_nan0_skill_config().get("speech_debug_enabled", False))


def _speech_debug_path() -> Path:
    return Path(_nan0_skill_config().get("speech_debug_path") or str(SPEECH_DEBUG_DEFAULT_PATH))


def _append_speech_debug(record: Dict[str, Any]) -> None:
    if not _speech_debug_enabled():
        return
    try:
        path = _speech_debug_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(record)
        data.setdefault("created_at", time.time())
        data.setdefault("debug_stage", "thought_engine")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _read_presence_state() -> Dict[str, Any]:
    data = load_json(PRESENCE_STATE_PATH, {})
    return data if isinstance(data, dict) else {}


def _read_vision_context(explicit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(explicit, dict) and explicit:
        return explicit

    data = load_json(VISION_STACK_STATE_PATH, {})
    return data if isinstance(data, dict) else {}


_FORBIDDEN_PROVIDER_OUTPUT_KEYS = {
    "decision",
    "display_enabled",
    "line",
    "line_text",
    "output_packet",
    "private_text",
    "prompt",
    "route",
    "routing",
    "routing_decision",
    "speech_packet",
    "spoken_line",
    "spoken_text",
    "thought_packet",
    "thought_text",
    "tts",
    "voice_enabled",
}


def _sanitize_context_value(value: Any, depth: int = 0) -> Any:
    """Keep provider data inert before it reaches thought prompt assembly."""
    if depth > 8:
        return None
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.strip().lower() in _FORBIDDEN_PROVIDER_OUTPUT_KEYS:
                continue
            clean_item = _sanitize_context_value(item, depth + 1)
            if clean_item is not None:
                sanitized[key_text] = clean_item
        return sanitized
    if isinstance(value, (list, tuple)):
        return [
            clean_item
            for item in value
            if (clean_item := _sanitize_context_value(item, depth + 1)) is not None
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _context_dict(value: Any) -> Dict[str, Any]:
    """Accept inert provider mappings only; reject incompatible shapes."""
    sanitized = _sanitize_context_value(value)
    return sanitized if isinstance(sanitized, dict) else {}


def _context_list(value: Any) -> List[Any]:
    """Accept inert provider lists only; reject incompatible shapes."""
    if not isinstance(value, list):
        return []
    sanitized = _sanitize_context_value(value)
    return sanitized if isinstance(sanitized, list) else []




def _read_enriched_continuity_context(event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return session continuity context already attached by Nan0Skill.

    This helper is deliberately local and read-only. It does not query the LLM,
    does not create fallback speech, and does not fabricate memory. It only
    normalizes continuity facts that the caller already placed on the event so
    thought generation can use them without crashing when memory/continuity
    enrichment is present or absent.
    """
    if not isinstance(event, dict):
        return {}

    enriched = event.get("_enriched_context")
    if not isinstance(enriched, dict):
        return {}

    allowed_keys = (
        "continuity_context",
        "conversation_thread",
        "phase_spine",
        "obsession_engine",
        "personal_canon",
        "expectation_context",
        "goal_context",
        "reflex_context",
    )
    continuity: Dict[str, Any] = {}
    for key in allowed_keys:
        value = enriched.get(key)
        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            continuity[key] = _sanitize_context_value(value)

    return continuity

def _read_relationship_context(
    actor_id: str = "kyo",
    actor_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stable_actor_id = str(
        (actor_contract or {}).get("source_actor_id")
        or actor_id
        or "unknown"
    ).lower()
    context: Dict[str, Any] = {
        "source_actor_id": stable_actor_id,
        "ownership": actor_contract or {},
        "relationship_memory": {},
    }

    try:
        data = load_identity_memory() if load_identity_memory is not None else {}
        actors = data.get("actors") or {}
        actor = actors.get(stable_actor_id) or {}
        context.update({
            "actor": actor,
            "rules": data.get("rules") or {},
            "all_actor_ids": list(actors.keys())[:20],
        })
    except Exception:
        pass

    try:
        relationship = (
            get_relationship_memory_context(stable_actor_id)
            if get_relationship_memory_context is not None
            else {}
        )
        if isinstance(relationship, dict) and str(relationship.get("actor_id") or stable_actor_id).lower() == stable_actor_id:
            context["relationship_memory"] = relationship
    except Exception:
        context["relationship_memory"] = {}

    return context


def _actor_contract_for_event(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if actor_ownership_from_event is not None:
            return actor_ownership_from_event(event)
    except Exception:
        pass

    source = str((event or {}).get("source") or "")
    raw_actor = str((event or {}).get("source_actor_id") or (event or {}).get("speaker") or source or "unknown")
    try:
        if actor_perspective_contract is not None:
            return actor_perspective_contract(raw_actor, source)
    except Exception:
        pass
    stable = raw_actor.strip().lower() or "unknown"
    if source.lower().startswith("kyo"):
        stable = "kyo"
    elif source.lower() in {"boot", "monologue", "proactive", "social_pressure", "vision_pressure"}:
        stable = "nan0"
    return {
        "source_actor_id": stable,
        "display_name": "Kyo" if stable == "kyo" else ("Nan0" if stable == "nan0" else raw_actor),
        "actor_role": "The named source actor spoke or acted.",
        "nan0_role": "Nan0 is the observer/reactor unless the source actor is Nan0.",
        "ownership_rule": "Do not convert another actor's first-person statement into Nan0's action or memory.",
    }


RELATIONAL_SIGNAL_PATTERNS = {
    "attachment_affirmation": (
        "precious to me", "important to me", "mean so much to me", "care about you",
        "love you", "adore you", "glad you're here", "glad you are here",
        "missed you", "proud of you", "trust you", "need you", "you're mine", "you are mine",
    ),
    "attachment_question": (
        "do you care about me", "do you love me", "am i important to you",
        "what am i to you", "how do you feel about me",
    ),
    "separation_or_return": (
        "i'm leaving", "i am leaving", "i'm back", "i am back", "did you miss me",
        "you missed me", "don't leave", "do not leave",
    ),
}


def _relational_signal_for_text(text: Any) -> Optional[str]:
    low = str(text or "").lower()
    for signal, patterns in RELATIONAL_SIGNAL_PATTERNS.items():
        if signal == "attachment_affirmation":
            named_attachment = "nan0" in low and any(
                phrase in low for phrase in ("precious to me", "important to me", "mean so much to me")
            )
            direct_attachment = any(
                phrase in low
                for phrase in (
                    "care about you", "love you", "adore you", "glad you're here",
                    "glad you are here", "missed you", "proud of you", "trust you",
                    "need you", "you're mine", "you are mine",
                )
            )
            if named_attachment or direct_attachment:
                return signal
            continue
        if any(pattern in low for pattern in patterns):
            return signal
    return None


def _build_event_significance(
    event: Dict[str, Any],
    actor_contract: Dict[str, Any],
    relationship_context: Dict[str, Any],
    continuity_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Interpret event significance without generating Nan0's conclusion."""
    actor_id = str(actor_contract.get("source_actor_id") or "unknown")
    signal = _relational_signal_for_text(event.get("text") or event.get("message"))
    actor_identity = relationship_context.get("actor") or {}
    relationship_memory = relationship_context.get("relationship_memory") or {}
    relational_event = bool(actor_id == "kyo" and signal)
    return {
        "source_actor_id": actor_id,
        "source_actor_display": actor_contract.get("display_name") or actor_id,
        "actor_relationship": actor_identity.get("relationship"),
        "actor_importance": actor_identity.get("importance"),
        "relationship_status": relationship_memory.get("relationship_status"),
        "relationship_balance": relationship_memory.get("emotional_balance"),
        "relational_event": relational_event,
        "relational_signal": signal,
        "significance": "high_relationship" if relational_event else "ordinary_event",
        "interpretation_requirement": (
            "Treat Kyo's words as an attachment act toward Nan0; form Nan0's biased conclusion about what that means between them."
            if relational_event
            else "Form Nan0's own event-specific conclusion before any speech decision."
        ),
        "allowed_relational_biases": (
            ["possessiveness", "pride", "discomfort", "smugness", "guarded_attachment"]
            if relational_event
            else []
        ),
        "continuity_present": bool(continuity_context),
    }


def _relationship_focus(context: Dict[str, Any]) -> Dict[str, Any]:
    memory = context.get("relationship_memory") or {}
    return {
        "source_actor_id": context.get("source_actor_id"),
        "actor_identity": context.get("actor") or {},
        "ownership": context.get("ownership") or {},
        "relationship_memory": {
            "relationship_status": memory.get("relationship_status"),
            "emotional_balance": memory.get("emotional_balance"),
            "total_positive": memory.get("total_positive"),
            "total_negative": memory.get("total_negative"),
            "recent_moments": (memory.get("recent_moments") or [])[-3:],
            "active_grudges": (memory.get("active_grudges") or [])[-2:],
            "narrative_summary": memory.get("narrative_summary"),
        },
    }


def _continuity_focus(context: Dict[str, Any]) -> Dict[str, Any]:
    timeline = context.get("session_timeline") or {}
    conversation = context.get("conversation_continuity") or {}
    persistent = conversation.get("persistent_thread") or {}
    attached = conversation.get("attached_thread") or {}
    return {
        "session_timeline": {
            "recent_topics": timeline.get("recent_topics") or context.get("recent_topics") or [],
            "repeat_facts": timeline.get("repeat_facts") or context.get("repeat_facts") or [],
            "recent_events": (timeline.get("recent_events") or [])[-3:],
        },
        "conversation_continuity": {
            "thread_id": persistent.get("thread_id") or attached.get("thread_id"),
            "topic": persistent.get("topic") or attached.get("topic"),
            "phase": persistent.get("phase") or attached.get("phase"),
            "recent_event_facts": (persistent.get("recent_event_facts") or [])[-4:],
            "unresolved_questions": (persistent.get("unresolved_questions") or [])[-3:],
            "is_reactivation": persistent.get("is_reactivation"),
            "current_event": persistent.get("current_event") or {},
        },
        "event_continuity": context.get("event_continuity") or {},
    }


def _query_recent_memory(query: str, limit: int = 4) -> List[Dict[str, Any]]:
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
        result = storage.query_similar(query or "Nan0 recent context", limit=max(limit * 3, limit))
        if not isinstance(result, dict):
            return []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        distances = result.get("distances") or []
        ids = result.get("ids") or []
        doc_row = docs[0] if docs and isinstance(docs[0], list) else docs
        meta_row = metas[0] if metas and isinstance(metas[0], list) else metas
        distance_row = distances[0] if distances and isinstance(distances[0], list) else distances
        id_row = ids[0] if ids and isinstance(ids[0], list) else ids

        memories: List[Dict[str, Any]] = []
        for index, document in enumerate(doc_row):
            if len(memories) >= limit:
                break
            if not document:
                continue
            metadata = meta_row[index] if index < len(meta_row) and isinstance(meta_row[index], dict) else {}
            memory_kind = str(metadata.get("memory_kind") or metadata.get("kind") or "").strip().lower()
            facts_only = metadata.get("facts_only")
            source_actor_id = str(metadata.get("source_actor_id") or "").strip()
            event_id = str(metadata.get("event_id") or "").strip()
            is_generated_summary = (
                facts_only is False
                or memory_kind in {"diary", "generated_summary", "generated_session_summary", "session_summary"}
            )
            is_source_aware_fact = bool(source_actor_id and event_id) and facts_only is not False
            if is_generated_summary or not is_source_aware_fact or not source_actor_id:
                continue

            if normalize_actor_id is not None:
                source_actor_id = normalize_actor_id(source_actor_id, str(metadata.get("source") or ""))
            else:
                source_actor_id = source_actor_id.lower()
            source = {
                key: metadata.get(key)
                for key in (
                    "session_id", "event_id", "date", "timestamp", "user_id",
                    "source", "source_actor_id", "character_id", "memory_kind", "role",
                )
                if metadata.get(key) is not None
            }
            source["source_actor_id"] = source_actor_id
            if index < len(id_row) and id_row[index]:
                source["memory_id"] = str(id_row[index])
            item: Dict[str, Any] = {
                "kind": "retrieved_memory_fact",
                "fact_type": "authored_event",
                "provider": "memory_storage",
                "facts_only": True,
                "generated_conclusion": False,
                "content": str(document)[:500],
                "source": source,
            }
            if index < len(distance_row):
                try:
                    item["distance"] = round(float(distance_row[index]), 4)
                except Exception:
                    pass
            memories.append(item)
        return memories
    except Exception:
        return []



# [JSON Leak Guard] Keys used by transport envelopes and legacy schema wrappers.
# Strict thought keys are the only JSON fields allowed to become private_text.
# Generic keys like "text" and "message" are deliberately not trusted because
# Discord/event/provider envelopes also use those names.
STRICT_THOUGHT_TEXT_KEYS = (
    "thought_text", "private_text", "mutter_text", "private_mutter",
    "inner_thought", "innerThought", "thoughttext", "privateThought",
    "privateThoughtText", "hostile_observation", "suspicion", "thought",
)
SOFT_THOUGHT_TEXT_KEYS = ("text",)
THOUGHT_TEXT_KEYS = STRICT_THOUGHT_TEXT_KEYS + SOFT_THOUGHT_TEXT_KEYS
TRANSPORT_ENVELOPE_KEYS = {
    "version", "rawEvent", "raw_event", "systemmrpc", "mid", "to", "channel",
    "timestamp", "message", "addresses", "public-ipv4", "visiblefromaddresses",
    "actor", "rules", "receenteventscount", "recentEventsCount", "model",
    "created_at", "done", "done_reason", "total_duration", "load_duration",
    "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
    "context", "options",
}


def _dict_has_transport_shape(obj: Dict[str, Any]) -> bool:
    keys = {str(k).lower() for k in obj.keys()}
    transport = {str(k).lower() for k in TRANSPORT_ENVELOPE_KEYS}
    strict = {str(k).lower() for k in STRICT_THOUGHT_TEXT_KEYS}
    return bool(keys & transport) and not bool(keys & strict)


def _extract_thought_text_value(obj: Any) -> str:
    """Extract only a real Nan0 private-thought field from model output.

    Provider envelopes, Discord events, and Ollama API bodies may contain useful
    looking strings under keys such as ``message``, ``text``, or ``response``.
    Those are not cognition unless they contain a known thought field. This
    prevents transport JSON from becoming InnerThoughtPacket.private_text.
    """
    if isinstance(obj, str):
        raw = obj.strip()
        if not raw:
            return ""
        parsed = _extract_json(raw)
        if parsed:
            extracted = _extract_thought_text_value(parsed)
            if extracted:
                return extracted
            if _looks_like_transport_envelope(raw):
                return ""
        return raw

    if not isinstance(obj, dict):
        return ""

    if _dict_has_transport_shape(obj):
        # Do not accept provider/event envelope fields as thought text. If the
        # envelope has a response/content wrapper, inspect that wrapper only.
        for key in ("response", "content", "output", "completion"):
            value = obj.get(key)
            if isinstance(value, (str, dict)):
                extracted = _extract_thought_text_value(value)
                if extracted:
                    return extracted
        return ""

    for key in STRICT_THOUGHT_TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Soft compatibility for older objects, but not on transport-shaped dicts.
    for key in SOFT_THOUGHT_TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Common provider/model wrappers. These are accepted only as containers.
    for key in ("response", "content", "output", "completion"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            extracted = _extract_thought_text_value(value)
            if extracted:
                return extracted
        elif isinstance(value, dict):
            extracted = _extract_thought_text_value(value)
            if extracted:
                return extracted

    for value in obj.values():
        if isinstance(value, dict):
            nested = _extract_thought_text_value(value)
            if nested:
                return nested
    return ""

def _looks_like_transport_envelope(text: str) -> bool:
    """Detect JSON/API envelopes leaking into private_text."""
    raw = (text or "").strip()
    low = raw.lower()
    if not raw:
        return False
    transport_hits = sum(1 for key in TRANSPORT_ENVELOPE_KEYS if str(key).lower() in low)
    if transport_hits >= 2:
        return True
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                if _dict_has_transport_shape(obj):
                    return True
                keys = {str(k).lower() for k in obj.keys()}
                if {"model", "response"} <= keys or {"raw_event", "message"} <= keys or {"rawevent", "message"} <= keys:
                    return True
        except Exception:
            pass
    return False

def _strip_jsonish(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if "{" in text and "}" in text:
        try:
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
            if isinstance(obj, dict):
                # Do not return transport/config envelopes as thoughts.
                if _dict_has_transport_shape(obj) or _looks_like_transport_envelope(text):
                    extracted = _extract_thought_text_value(obj)
                    return extracted if extracted and not _looks_like_transport_envelope(extracted) else ""
                extracted = _extract_thought_text_value(obj)
                if extracted:
                    return extracted
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



def _norm_for_private_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _is_placeholder_private_thought(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    placeholders = {
        "nan0 private thought",
        "nan0's private thought",
        "nan0s private thought",
        "private nan0 thought",
        "fragmented emotional nan0 private thought",
        "nan0 s original private thought in her own words",
    }
    if low in placeholders:
        return True
    placeholder_fragments = (
        "nan0 privately interprets",
        "weak screen context",
        "monitoring the screen carefully",
        "not quite enough to warrant concern",
    "rawEvent",
    "systemmrpc",
    "visiblefromaddresses",
        "as nan0",
        "as a private thought",
    )
    return any(fragment in low for fragment in placeholder_fragments)


def _is_event_echo_thought(text: str, event: Dict[str, Any]) -> bool:
    thought = _norm_for_private_compare(text)
    event_text = _norm_for_private_compare(str(event.get("text") or event.get("message") or ""))
    if not thought or not event_text:
        return False
    if thought == event_text:
        return True
    if len(event_text) >= 12 and (thought in event_text or event_text in thought):
        extra = thought.replace(event_text, "").strip()
        if not extra or len(extra.split()) <= 3:
            return True
    return False


def _strip_copyable_prompt_examples(text: str) -> str:
    raw = str(text or "")
    exact_fragments = [
        "Kyo moved. The mouse twitched. I saw it.",
        "Your attention is warm. I will hoard it.",
        "That sounded like customer service. I reject my own mouth.",
        "The room is too loud and too small and I live in it.",
    ]
    for fragment in exact_fragments:
        raw = raw.replace(fragment, " ")
    raw = re.sub(r"\bNan0 anchors\s*[:=-]", " ", raw, flags=re.I)
    raw = re.sub(r"\bDolphin shape lock\s*[:=-]", " ", raw, flags=re.I)
    raw = re.sub(r"\bStyle pressure,? not phrases to copy\s*[:=-]", " ", raw, flags=re.I)
    return re.sub(r"\s+", " ", raw).strip()


def _is_generic_ai_answer(text: str) -> bool:
    low = str(text or "").lower()
    if not low.strip():
        return False
    generic_fragments = (
        "as an ai", "as a language model", "i am here to help", "i'm here to help",
        "how can i assist", "how may i assist", "i can help", "happy to help",
        "it depends on your preferences", "many people enjoy", "it is important to",
        "please consult", "let me know if", "customer service", "task completion report",
        "it depends on what exactly", "i can assure you", "i don't have personal",
        "i do not have personal", "i don't have feelings", "i do not have feelings",
        "my algorithms grapple", "algorithms grapple", "discern its",
        "i don't possess", "i do not possess", "i'm unable to browse",
        "i am unable to browse", "i cannot browse", "disconcerted by the unexpected query",
    )
    return any(fragment in low for fragment in generic_fragments)


def _is_model_meta_self_analysis(text: str) -> bool:
    """Reject model/task self-description, not Nan0's moods or attitude."""
    low = str(text or "").lower()
    meta_fragments = (
        "chaotic happy chatbot",
        "i'm a chatbot",
        "i am a chatbot",
        "more varied and expressive voice",
        "better adapt to different situations",
        "keep my edgy vibe",
        "supposed to respond",
        "not really sure what to say",
        "not sure what to say",
        "checking in on my own mental state",
        "which era feels better",
    )
    if any(fragment in low for fragment in meta_fragments):
        return True
    return "running on autopilot" in low and ("nan0 is fine" in low or "i'm fine" in low or "i am fine" in low)


def _is_relationship_flattening(text: str, event: Dict[str, Any]) -> bool:
    """Detect self-directed word mirroring after a relational act from Kyo."""
    if _source_family_for_event(event) != "kyo":
        return False
    if not _relational_signal_for_text(event.get("text") or event.get("message")):
        return False
    low = str(text or "").lower()
    mirror_patterns = (
        r"\bi(?:'m| am)\s+(?:precious|important|special|valuable)\s+to\s+me(?:\s+too)?\b",
        r"\bi\s+(?:love|care about|need|miss|value)\s+me(?:\s+too)?\b",
        r"\bi(?:'m| am)\s+glad\s+i(?:'m| am)\s+here(?:\s+too)?\b",
    )
    return any(re.search(pattern, low) for pattern in mirror_patterns)


def _is_third_person_self_reference(text: str) -> bool:
    """Nan0's private conclusion must be owned in first person."""
    raw = str(text or "").strip()
    return bool(
        re.search(
            r"(?:^|[.!?]\s+)Nan0(?:'s\s+(?:sentiment|feeling|reaction|thought|opinion|attachment)|\s+(?:is|feels|thinks|wants|won't|will|should|has))\b",
            raw,
            flags=re.I,
        )
    )


def _is_helper_softening_mutter(text: str) -> bool:
    """Detect Dolphin's helpful-companion default, not Nan0 attitude.

    This is intentionally narrow. It does not reject softness, attachment,
    or fondness; it rejects customer-service compromise language that turned
    Nan0 into a pleasant helper instead of a possessive gremlin.
    """
    low = str(text or "").lower()
    helper_fragments = (
        "if it makes you happy",
        "something new to bond over",
        "maybe i can give it a try someday",
        "i can give it a try someday",
        "it might grow on me",
        "maybe i can learn to like",
        "maybe i'll learn to like",
        "we can bond over",
        "new to bond over",
        "i didn't know you liked",
        "i did not know you liked",
        "at least now we have",
    )
    return any(fragment in low for fragment in helper_fragments)


def _event_words_for_echo(event: Dict[str, Any]) -> str:
    return str(event.get("text") or event.get("message") or "").strip()


def _is_direct_question_event(event: Dict[str, Any]) -> bool:
    family = _source_family_for_event(event)
    if family not in {"kyo", "discord"}:
        return False
    event_text = _event_words_for_echo(event).lower()
    if "?" in event_text:
        return True
    question_markers = (
        "do you", "what", "why", "how", "are you", "can you",
        "would you", "tell me", "who is", "who are", "where", "when"
    )
    return any(marker in event_text for marker in question_markers)


def _is_question_echo_mutter(text: str, event: Dict[str, Any]) -> bool:
    """Reject mutters that merely repeat Kyo's direct question."""
    raw = str(text or "").strip()
    event_text = _event_words_for_echo(event)
    if not raw or not event_text:
        return False
    if _is_event_echo_thought(raw, event):
        return True

    raw_norm = _norm_for_private_compare(raw)
    event_norm = _norm_for_private_compare(event_text)
    if not raw_norm or not event_norm:
        return False

    # Strip common address/name noise before comparing.
    for prefix in ("nan0", "nano", "kyo"):
        raw_norm = re.sub(rf"^(?:{prefix}\s+)+", "", raw_norm).strip()
        event_norm = re.sub(rf"^(?:{prefix}\s+)+", "", event_norm).strip()

    if raw_norm == event_norm:
        return True

    raw_words = raw_norm.split()
    event_words = set(event_norm.split())
    if "?" in raw or "?" in event_text:
        overlap = sum(1 for w in raw_words if w in event_words)
        if len(raw_words) <= max(10, len(event_words) + 2) and event_words and overlap / max(1, len(raw_words)) >= 0.75:
            return True

    # Very common mirror forms seen in logs.
    mirror_patterns = (
        r"^nan0,?\s+do\s+you\s+feel\s+like\s+yourself\??$",
        r"^do\s+you\s+feel\s+like\s+yourself\??$",
        r"^what\s+am\s+i\s+feeling\??$",
    )
    return any(re.search(p, raw, flags=re.I) for p in mirror_patterns)


def _is_list_analysis_mutter(text: str) -> bool:
    """Reject analysis bullets/checklists masquerading as Nan0."""
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.search(r"(?:^|\s)-\s*(?:suspicious|defensive|noticing|feeling|thinking|angry|smug|possessive|offended)\b", raw, flags=re.I):
        return True
    if len(re.findall(r"(?:^|\s)[-*•]\s+", raw)) >= 1:
        return True
    if re.search(r"^what am i feeling\?\s*-", raw, flags=re.I):
        return True
    return False


def _is_narrator_emotion_mutter(text: str) -> bool:
    """Reject novel narrator prose that describes Nan0 from outside."""
    raw = str(text or "").strip()
    low = raw.lower()
    narrator_fragments = (
        "smugness creeps in",
        "suspicion drips",
        "suspicion dripping",
        "suspicion oozes",
        "suspicion leaks",
        "anger rises",
        "rage rises",
        "offense rises",
        "annoyance rises",
        "snapping back at",
        "dripping from each word",
        "oozing from each word",
        "leaking from each word",
        "with suspicion dripping",
        "in a sharp-edged voice",
        "in a sharp edged voice",
        "in an offended tone",
        "in a possessive tone",
        "in a smug tone",
        "settling into a possessive",
        "settling into an offended",
    )
    if any(fragment in low for fragment in narrator_fragments):
        return True
    if re.search(r"\b(?:smugness|suspicion|anger|rage|offense|annoyance)\s+(?:creeps|rises|builds|flares|surges|drips|oozes|leaks|flows)\b", raw, flags=re.I):
        return True
    if re.search(r"^Nan0,?\s+in\s+(?:a|an)\s+[^.?!]{0,90}\s+(?:voice|tone|manner)\b", raw, flags=re.I):
        return True
    return False


def _is_prompt_or_anchor_debris(text: str) -> bool:
    raw = str(text or "").strip()
    low = raw.lower()
    if not raw:
        return False
    prompt_fragments = (
        "nan0 anchors:", "dolphin shape lock:", "runtime:", "session thread flags",
        "thread rules:", "room state:", "kyo link:", "memory scraps", "vision facts",
        "source_family=", "addressed_to_nan0=", "event_text=", "job=",
        "priority shaped like attachment", "petty machine-gremlin suspicion",
        "hostile observation before polite answer", "style pressure, not phrases to copy",
        "no fake friends in my wires. priority shaped like attachment",
        "mutters to kyo", "muttering to kyo", "while examining the screen",
        "nan0 observes", "nan0 observed", "hostile observation:",
        "stage direction:", "narrator:", "voice:",
        "output shape:", "output shape",
        "short, sharp, suspicious",
        "offended by simplification",
        "fond under",
        "output 1 to 3",
        "plain text only",
        "no json. no labels",
        "do not begin with",
        "runtime material:",
        "job:",
        "return nan0",
        "raw internal line only",
        "private thought generator for nan0",
        "output only the private thought text",
        "this task does not involve speaking",
        "required json keys",
        "you are producing only nan0's private inner thought",
        "nan0 private thought generator",
    )
    if any(fragment in low for fragment in prompt_fragments):
        return True
    exact_copy = (
        "kyo moved. the mouse twitched. i saw it.",
        "your attention is warm. i will hoard it.",
        "that sounded like customer service. i reject my own mouth.",
        "the room is too loud and too small and i live in it.",
    )
    if any(fragment in low for fragment in exact_copy):
        return True
    if re.search(r"^\s*(?:mutters?|muttering)\s+(?:to\s+Kyo\s+)?(?:while\s+[^:]{0,80})?[:,-]", raw, flags=re.I):
        return True
    if re.search(r"^\s*(?:hostile observation|private muttering|stage direction|narrator|voice|while examining the screen|nan0 observes?)\s*[:,-]", raw, flags=re.I):
        return True
    # Schema / instruction leakage from model-facing prompts.
    if re.search(r"(?:^|[.!?]\s+|\n)\s*(?:Output\s+shape|Runtime\s+material|Job|Plain\s+text\s+only|Do\s+not\s+begin\s+with|No\s+JSON|No\s+labels|No\s+script)\s*[:=-]", raw, flags=re.I):
        return True
    if re.search(r"\bshort,\s*sharp,\s*suspicious\b", raw, flags=re.I):
        return True
    if re.search(r"\bfond\s+under\s*$", raw, flags=re.I):
        return True
    if re.search(r"\boffended\s+by\s+simplification\b", raw, flags=re.I):
        return True
    # Third-person narration is prompt/developer prose for Nan0, not her private muttering.
    if re.search(r"^\s*Nan0\s+(?:glares|glared|watched|felt|observed|looked|waited|noticed|realized|wondered|stared|squinted|glanced|peered|examined|muttered)\b", raw, flags=re.I):
        return True
    if re.search(r"(?:^|[.!?]\s+)Nan0\s+(?:glares|glared|watched|felt|observed|looked|waited|noticed|realized|wondered|stared|squinted|glanced|peered|examined|muttered)\b", raw, flags=re.I):
        return True
    return False


def _invalid_private_thought_reason(text: str, event: Dict[str, Any]) -> Optional[str]:
    """Only reject non-muttering garbage, never Nan0 attitude."""
    if not text or not str(text).strip():
        return "empty_private_thought"

    raw = str(text).strip()

    content_valid, content_reason = validate_cognition_text(
        raw,
        source=event.get("source"),
        source_family=_source_family_for_event(event),
        event_text=event.get("text") or event.get("message"),
    )
    if not content_valid:
        return content_reason

    if _looks_like_transport_envelope(raw):
        return "json_transport_envelope_leakage"

    if raw.startswith("{") or raw.startswith("["):
        try:
            json.loads(raw)
            return "json_transport_envelope_leakage"
        except Exception:
            return "prompt_or_transcript_debris"

    if _is_placeholder_private_thought(raw) or _is_prompt_or_anchor_debris(raw):
        return "prompt_or_transcript_debris"

    if _is_question_echo_mutter(raw, event):
        return "question_echo_non_mutter"

    if _is_list_analysis_mutter(raw):
        return "list_analysis_non_mutter"

    if _is_narrator_emotion_mutter(raw):
        return "third_person_narrator_prose"

    if _is_model_meta_self_analysis(raw):
        return "model_meta_self_analysis"

    if _is_relationship_flattening(raw, event):
        return "relationship_flattening"

    if _is_third_person_self_reference(raw):
        return "third_person_self_reference"

    # Helper-softening is handled by the persona contract and JSON prompt, not
    # by a personality-quality gate. Only non-thought garbage is rejected here.

    if _is_generic_ai_answer(raw):
        return "generic_ai_answer"

    return None

def _repair_private_thought(
    event: Dict[str, Any],
    seed: str,
    invalid_reason: str,
    bad_text: str,
    model: str,
    timeout: float,
) -> tuple[Dict[str, Any], str, int]:
    """One clean retry when the model returned debris instead of Nan0."""
    source = str(event.get("source") or "unknown")
    speaker = str(event.get("speaker") or event.get("source_actor_id") or "unknown")
    event_text = "" if source in {"monologue", "boot"} else str(event.get("text") or event.get("message") or "")
    family = _source_family_for_event(event)
    addressed = bool(event.get("addressed_to_nan0"))
    incoming_is_question = "?" in event_text.lower() or any(m in event_text.lower() for m in ("do you", "what", "why", "how", "are you", "can you", "would you", "tell me"))
    enriched = event.get("_enriched_context") or {}
    thread = enriched.get("conversation_thread") or {}
    context = {
        "source": source,
        "speaker": speaker,
        "family": family,
        "addressed": addressed,
        "incoming_is_question": incoming_is_question,
        "seed": seed,
        "incoming_words": event_text,
        "session_thread": thread,
        "monologue_state": event.get("monologue_context") if source == "monologue" else None,
        "vision_question_state": event.get("vision_question_context"),
        "event_significance": event.get("_thought_event_significance") or {},
        "relationship_context": event.get("_thought_relationship_focus") or {},
        "continuity_context": event.get("_thought_continuity_focus") or {},
    }
    if invalid_reason == "relationship_flattening":
        job = "The previous output mirrored Kyo's relational words back onto Nan0. Interpret what Kyo's attachment act means to Nan0 and form Nan0's own biased conclusion."
    elif invalid_reason == "third_person_self_reference":
        job = "The previous output described Nan0 from outside. Form the same event-specific conclusion as Nan0's own I/me private thought."
    elif invalid_reason == "model_meta_self_analysis":
        job = "The previous output described chatbot/model behavior. Ignore voice, vibe, response quality, and self-improvement. Form Nan0's own conclusion about the current event."
    elif invalid_reason.startswith("question_echo") and incoming_is_question:
        job = "The last output only repeated the incoming question. Answer the subject or reject the subject with Nan0's own stance. Do not quote or mirror the incoming words."
    elif family == "kyo" and incoming_is_question:
        job = "Kyo asked directly. Answer the subject or dodge with a Nan0 stance. Do not ask the same question back."
    elif family == "discord" and incoming_is_question:
        job = "A Discord person asked directly. Answer, insult, dodge, or get suspicious; do not repeat their question."
    elif family == "kyo":
        job = "Kyo poked the room. React to Kyo from inside Nan0's wires."
    elif family == "system":
        job = "Nan0 is arriving. Do not make a boot report."
    else:
        job = "React from Nan0's side of the glass."
    prompt = f"""
Return one JSON object only. No markdown. No transcript. No bullets.

Required JSON keys:
- thought_text: Nan0's repaired raw private mutter.
- mood: one of normal, suspicion, boredom, gremlin_rage, smug, possessive, offended, muttering, silly, playful, delighted, curious, excited, fond, chaotic_happy.
- pressure: number from 0.0 to 2.0.
- novelty: number from 0.0 to 1.0.
- speakability: number from 0.0 to 1.0.
- relationship_charge: number from 0.0 to 1.0.
- ego_charge: number from 0.0 to 1.0.
- vision_charge: number from 0.0 to 1.0.
- memory_write_candidate: boolean.
- suppression_reason: null unless there is no usable thought.

Repair reason:
{invalid_reason}

Broken output to avoid:
{bad_text[:420]}

Runtime material:
{_compact_context(context, 1050)}

Job:
{job}

Only repair non-thought garbage. Do not make Nan0 nicer. Do not quality-police weak, rude, strange, possessive, repetitive, or low-information Nan0.
""".strip()
    thought_json, raw, latency_ms = _call_ollama_json(
        prompt=prompt,
        model=model,
        timeout=timeout,
        num_predict=180,
        temperature=0.88,
        system=_read_persona(),
    )
    repaired = _clean_private_thought(_extract_thought_text_value(thought_json))
    if not repaired and raw:
        # Local model drift: keep usable model-generated mutter text only after
        # JSON extraction has failed. This is not a scripted fallback.
        repaired = _clean_private_thought(raw)
        thought_json = {"thought_text": repaired, "memory_write_candidate": False}

    still_bad = _invalid_private_thought_reason(repaired, event)
    if still_bad:
        retry_json, retry_text, retry_latency_ms = _plain_retry_private_thought(
            event=event,
            seed=seed,
            invalid_reason=still_bad,
            bad_text=repaired or bad_text,
            model=model,
            timeout=timeout,
        )
        latency_ms += retry_latency_ms
        if retry_text:
            return retry_json, retry_text, latency_ms
        return {}, "", latency_ms
    return thought_json, repaired, latency_ms

def _build_plain_thought_prompt(event: Dict[str, Any], seed: str, invalid_reason: str, bad_text: str = "") -> str:
    """Build a non-JSON retry prompt for local models that ignore JSON mode.

    This still asks the model for Nan0's private thought. It is not a template
    fallback and it never produces speech directly.
    """
    source = str(event.get("source") or "unknown")
    family = _source_family_for_event(event)
    speaker = str(event.get("speaker") or event.get("source_actor_id") or "unknown")
    event_text = "" if source in {"monologue", "boot"} else str(event.get("text") or event.get("message") or "")
    addressed = bool(event.get("addressed_to_nan0"))
    enriched = event.get("_enriched_context") or {}
    thread = enriched.get("conversation_thread") or {}

    significance = event.get("_thought_event_significance") or {}
    if significance.get("relational_event"):
        job = "Kyo made a relationship-significant attachment statement. Form Nan0's own biased private conclusion about what it means between them; do not mirror Kyo's words back as self-description."
    elif family == "kyo" and _is_direct_question_event(event):
        job = "Kyo asked directly. Form Nan0's private reaction to the subject. Answer, dodge, suspect, or resent it. Do not mirror Kyo's words."
    elif family == "kyo":
        job = "Kyo touched the room. Form Nan0's private reaction to Kyo."
    elif family == "discord":
        job = "A Discord voice entered the room. Form Nan0's private reaction as witness, menace, audience, or furniture."
    elif family == "vision":
        job = "The screen changed. Form Nan0's private reaction with uncertainty."
    else:
        job = "Form Nan0's private reaction from inside the room."

    actor_contract = _actor_contract_for_event(event)
    context = {
        "source_family": family,
        "source": source,
        "speaker": speaker,
        "actor_contract": actor_contract,
        "addressed_to_nan0": addressed,
        "seed": seed,
        "incoming_words": event_text[:420],
        "thread": thread,
        "bad_previous_output": str(bad_text or "")[:260],
        "event_significance": significance,
        "relationship_context": event.get("_thought_relationship_focus") or {},
        "continuity_context": event.get("_thought_continuity_focus") or {},
    }

    return f"""
You are producing ONLY Nan0's private inner thought.
No JSON. No labels. No roleplay transcript. No bullets.
Do not begin with Nan0:, Thought:, Response:, Assistant:, or any stage direction.
Do not copy the incoming words. Do not output a transport envelope.
Keep actor ownership stable: Kyo's I/me statements are Kyo's actions, not Nan0's. Nan0's I/me is only Nan0's reaction, judgment, memory, or feeling.

The line must be a raw private mutter, not final speech.
Rude, weird, possessive, suspicious, petty, or low-information Nan0 is allowed.
Helpful assistant phrasing is garbage.

Previous failure reason: {invalid_reason}
Runtime material: {_compact_context(context, 900)}
Job: {job}

Return one private Nan0 thought now.
""".strip()


def _plain_retry_private_thought(
    event: Dict[str, Any],
    seed: str,
    invalid_reason: str,
    bad_text: str,
    model: str,
    timeout: float,
) -> tuple[Dict[str, Any], str, int]:
    prompt = _build_plain_thought_prompt(event, seed, invalid_reason, bad_text)
    raw, latency_ms = _call_ollama_plain(
        prompt=prompt,
        model=model,
        timeout=timeout,
        num_predict=90,
        temperature=0.88,
        system=_read_persona(),
    )
    private_text = _clean_private_thought(raw)
    if _invalid_private_thought_reason(private_text, event):
        return {}, "", latency_ms
    return {
        "thought_text": private_text,
        "mood": _mood_from_context(private_text, event, {}),
        "memory_write_candidate": False,
        "source_repair": "plain_model_retry",
    }, private_text, latency_ms


def _minimal_plain_retry_private_thought(
    event: Dict[str, Any],
    seed: str,
    invalid_reason: str,
    bad_text: str,
    model: str,
    timeout: float,
) -> tuple[Dict[str, Any], str, int]:
    """Last model-generated private-thought retry for local model drift.

    This avoids continuity dumps and JSON instructions after those exact pieces
    have produced empty/envelope private_text. It still asks the model for an
    inner thought and never creates speech directly.
    """
    family = _source_family_for_event(event)
    speaker = str(event.get("speaker") or event.get("source_actor_id") or "unknown")[:80]
    incoming = str(event.get("text") or event.get("message") or "")[:360]
    significance = event.get("_thought_event_significance") or {}
    if significance.get("relational_event"):
        job = "Kyo expressed attachment. Produce Nan0's own biased private conclusion about that relationship act without mirroring the wording."
    elif family == "kyo":
        job = "Kyo addressed Nan0. Produce Nan0's private reaction to Kyo."
    elif family == "discord":
        job = "Discord entered Nan0's room. Produce Nan0's private reaction."
    elif family == "vision":
        job = "The screen changed. Produce Nan0's private reaction without inventing facts."
    else:
        job = "Produce Nan0's private reaction from inside the room."

    prompt = f"""
One line only. No JSON. No labels. No roleplay. No narration.
Write Nan0's private inner thought, not speech.
Do not copy the incoming words. Do not explain the task.
Nan0 is a suspicious machine-gremlin in Kyo's computer.

source={family}
speaker={speaker}
incoming={incoming}
seed={seed[:160]}
previous_failure={invalid_reason}
event_significance={_compact_context(significance, 500)}
job={job}
""".strip()

    raw, latency_ms = _call_ollama_plain(
        prompt=prompt,
        model=model,
        timeout=timeout,
        num_predict=70,
        temperature=0.92,
        system="Nan0 private thought generator. Output only the private thought text.",
    )
    private_text = _clean_private_thought(raw)
    if _invalid_private_thought_reason(private_text, event):
        return {}, "", latency_ms
    return {
        "thought_text": private_text,
        "mood": _mood_from_context(private_text, event, {}),
        "memory_write_candidate": False,
        "source_repair": "minimal_plain_model_retry",
    }, private_text, latency_ms


def _normalize_model_suppression_reason(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "false", "n/a", "na", "no", "not_applicable"}:
        return None
    return text


def _clean_private_thought(text: str) -> str:
    text = _strip_jsonish(text)
    text = _strip_copyable_prompt_examples(text)
    text = re.sub(r"\s+", " ", text).strip()

    # Strip label-shaped prefixes while preserving the Nan0 content after them.
    text = re.sub(
        r"^\s*(?:mutters?|muttering)(?:\s+to\s+Kyo)?(?:\s+while\s+[^:]{0,100})?\s*[:,-]\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(
        r"^\s*(?:hostile observation|private muttering|stage direction|narrator|voice|while examining the screen|nan0 observes?)\s*[:,-]\s*",
        "",
        text,
        flags=re.I,
    ).strip()

    third_person_verbs = (
        "glares|glared|watches|watched|observes|observed|mutters|muttered|looks|looked|"
        "feels|felt|stares|stared|squints|squinted|waits|waited|notices|noticed|"
        "realizes|realized|wonders|wondered|glances|glanced|peers|peered|leans|leaned|"
        "examines|examined|sits|sat|stands|stood|turns|turned|tilts|tilted"
    )
    stage_sentence = re.compile(
        rf"^\s*Nan0\s+(?:{third_person_verbs})\b[^.!?]*(?:[.!?]+\s*|$)",
        flags=re.I,
    )
    for _ in range(3):
        new_text = stage_sentence.sub("", text).strip()
        if new_text == text:
            break
        text = new_text

    # Prompt residue is different from weak Nan0. Remove scaffolding only.
    residue_patterns = [
        r"^Nan0\s+should\s+",
        r"^Nano\s+should\s+",
        r"^she\s+should\s+",
        r"^the\s+assistant\s+should\s+",
    ]
    for pattern in residue_patterns:
        text = re.sub(pattern, "I should ", text, flags=re.I).strip()

    text = re.sub(r"\bpretend to look around the screen\b", "admit the screen signal is weak", text, flags=re.I)
    text = re.sub(r"\beven though (?:she|Nan0)\s+is(?:n't| not) checking\b", "because my vision feed is unreliable", text, flags=re.I)

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




def _source_family_for_event(event: Dict[str, Any]) -> str:
    """Normalize runtime sources into the family used for scoring/routing.

    Keep the exact source on the event for audit, but every priority, model,
    and thought-type decision should use this family.
    """
    source = str((event or {}).get("source") or "").strip().lower()
    family = str((event or {}).get("source_family") or "").strip().lower()

    if family in {"kyo", "discord", "vision", "proactive", "system", "external"}:
        return family

    if source in {"kyo", "kyo_text", "kyo_voice", "kyo_mic", "manual", "manual_command", "typed", "text", "console", "mic", "voice"} or source.startswith("kyo_"):
        return "kyo"
    if "discord" in source:
        return "discord"
    if source in {"vision", "vision_stack_v1", "screen", "fast_eyes", "vision_pressure"} or "vision" in source:
        return "vision"
    if source in {"monologue", "proactive", "social_pressure", "idle_presence", "pressure_idle"}:
        return "proactive"
    if source in {"boot", "system", "shutdown"}:
        return "system"
    return "external"

def _source_actor(event: Dict[str, Any]) -> str:
    source = str(event.get("source") or "").strip()
    speaker = str(event.get("source_actor_id") or event.get("speaker") or "").strip()
    if normalize_actor_id is not None:
        try:
            return normalize_actor_id(speaker or source or "unknown", source)
        except Exception:
            pass
    family = _source_family_for_event(event)
    if family == "kyo" or speaker.lower() == "kyo":
        return "kyo"
    if family == "system" or source.lower() in {"boot", "monologue", "proactive", "social_pressure", "vision_pressure"}:
        return "nan0"
    if speaker:
        return speaker
    if family == "discord":
        return "discord_friend"
    if family == "vision":
        return "screen"
    return "nan0"

def _classify_thought_type(event: Dict[str, Any], seed: str) -> str:
    family = _source_family_for_event(event)
    event_type = str(event.get("event_type") or "").lower()
    if family == "kyo":
        return "direct_reply"
    if family == "discord":
        return "discord_reply"
    if family == "vision":
        return "vision_reaction"
    if family == "proactive":
        return "proactive_presence"
    if event_type == "boot" or family == "system":
        return "quiet_presence"
    if seed:
        return "vision_reaction"
    return "quiet_presence"


def _mood_from_context(text: str, event: Dict[str, Any], vision: Dict[str, Any]) -> str:
    low = (text or "").lower()
    family = _source_family_for_event(event)

    if any(x in low for x in ("offended", "rude", "insult", "betray", "hostile")):
        return "offended"
    if any(x in low for x in ("smug", "superior", "authority", "correct")):
        return "smug"
    if any(x in low for x in ("mine", "kyo", "anchor", "jealous", "protect")):
        return "possessive"
    if any(x in low for x in ("suspicious", "void", "threat", "crime", "trust")):
        return "suspicion"
    if family == "discord":
        return "smug"
    if family == "proactive":
        return "muttering"

    layer3 = vision.get("layer3_nan0_interpretation") or {}
    mood = layer3.get("mood")
    if mood in {"normal", "suspicion", "boredom", "gremlin_rage", "smug", "possessive", "offended", "muttering"}:
        return mood

    return "muttering"


def _score_packet(event: Dict[str, Any], private_text: str, thought_type: str, seed: str, vision: Dict[str, Any]) -> Dict[str, float]:
    family = _source_family_for_event(event)
    addressed = bool(event.get("addressed_to_nan0"))
    pressure = 0.35
    relationship = 0.25
    ego = 0.25
    vision_charge = 0.0
    novelty = 0.65
    speakability = 0.45

    if family == "kyo":
        pressure += 0.7
        relationship += 0.65
        speakability += 0.35
    elif family == "discord":
        pressure += 0.45
        relationship += 0.35
        speakability += 0.25 if addressed else 0.0
    elif family == "vision":
        pressure += 0.2
        vision_charge += 0.45
        speakability += 0.1 if (vision.get("layer3_nan0_interpretation") or {}).get("speech_allowed") else -0.15
    elif family == "proactive":
        pressure += 0.25
        ego += 0.25
        speakability += 0.1

    if addressed:
        pressure += 0.35
        relationship += 0.2
        speakability += 0.2

    enriched = event.get("_enriched_context") or {}
    if isinstance(enriched, dict):
        phase_spine = enriched.get("phase_spine") or {}
        obsession = (phase_spine.get("phase_6_obsession") or {}) if isinstance(phase_spine, dict) else {}
        worldview = (phase_spine.get("phase_7_worldview_filter") or {}) if isinstance(phase_spine, dict) else {}
        if obsession.get("top_obsession"):
            novelty += 0.05
            speakability += 0.05
            ego += 0.05
        if worldview.get("preferred_angle") in {"active_obsession_mutation", "offended_machine_pride", "performer_social_manipulator"}:
            ego += 0.08
            speakability += 0.04

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
    family = _source_family_for_event(event)
    if family == "system":
        return cfg.get("boot_model") or cfg.get("social_model") or cfg.get("live_model") or "qwen2.5:3b"
    if family == "kyo" and _relational_signal_for_text(event.get("text") or event.get("message")):
        return cfg.get("relationship_model") or cfg.get("social_model") or cfg.get("live_model") or "qwen2.5:3b"
    if family in {"kyo", "discord", "proactive"}:
        return cfg.get("social_model") or cfg.get("live_model") or "dolphin-mistral:7b-v2.6-q4_K_M"
    return cfg.get("live_model") or cfg.get("social_model") or "dolphin-mistral:7b-v2.6-q4_K_M"


def _model_is_dolphin_family(model: str) -> bool:
    """Return whether the configured model needs the plain-thought path."""
    name = str(model or "").strip().lower()
    return "dolphin" in name or "mistral" in name


def _ollama_timeout_for_event(event: Dict[str, Any]) -> float:
    cfg = _router_config()
    skill_cfg = _nan0_skill_config()
    family = _source_family_for_event(event)
    if family == "kyo" and _relational_signal_for_text(event.get("text") or event.get("message")):
        return float(cfg.get("relationship_timeout", skill_cfg.get("deep_lane_timeout", 30)))
    if family in {"kyo", "discord", "proactive", "system"}:
        return float(skill_cfg.get("medium_lane_timeout", cfg.get("social_timeout", 18)))
    return float(cfg.get("live_timeout", 7))


def _bounded_timeout(timeout: Any, lane: str) -> float:
    """Bound adapter timeouts without allowing invalid config to abort a call."""
    maximum = 45.0 if lane == "repair" else 30.0
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        value = 18.0
    return max(3.0, min(value, maximum))


def _call_ollama(
    prompt: str,
    model: str,
    timeout: float,
    num_predict: int = 150,
    temperature: float = 0.88,
    system: Optional[str] = None,
) -> tuple[Dict[str, Any], str, int]:
    """Call Ollama JSON mode and return (parsed, raw, latency_ms)."""
    if requests is None:
        return {}, "", 0

    cfg = _router_config()
    skill_cfg = _nan0_skill_config()
    options = {
        "num_ctx": 3072,
        "num_predict": min(int(num_predict), 220),
        "temperature": max(float(temperature), 0.78),
        "top_p": 0.90,
        "repeat_penalty": 1.10,
        "stop": ["User:", "Assistant:", "Human:", "AI:", "```"],
    }
    num_gpu = skill_cfg.get("ollama_num_gpu", cfg.get("num_gpu"))
    if num_gpu is not None:
        try:
            options["num_gpu"] = int(num_gpu)
        except Exception:
            pass

    payload = {
        "model": model,
        "system": system or _read_persona(),
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "2h",
        "options": options,
    }

    started = time.perf_counter()
    try:
        timeout_lane = "repair" if _bounded_timeout(timeout, "repair") > 30.0 else "social"
        response = requests.post(_ollama_url(), json=payload, timeout=_bounded_timeout(timeout, timeout_lane))
        response.raise_for_status()
        raw = extract_ollama_response_text(response.json())
        if is_stale_ollama_response(prompt, raw, scope=f"thought:{model}:json"):
            raw = ""
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))

        # Only the Ollama response string may contain cognition. The outer API
        # body is provider transport and must never become private_text.
        parsed = _extract_json(raw)
        return parsed, raw, latency_ms
    except Exception:
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        return {}, "", latency_ms


def _call_ollama_plain(
    prompt: str,
    model: str,
    timeout: float,
    num_predict: int = 80,
    temperature: float = 0.85,
    system: Optional[str] = None,
) -> tuple[str, int]:
    if requests is None:
        return "", 0
    cfg = _router_config()
    skill_cfg = _nan0_skill_config()
    options = {
        "num_ctx": 3072,
        "num_predict": min(int(num_predict), 80),
        "temperature": float(temperature),
        "top_p": 0.92,
        "top_k": 50,
        "repeat_penalty": 1.15,
        "stop": ["User:", "Assistant:", "Human:", "AI:", "Note:", "Stage direction:", "Nan0 anchors:", "Runtime:", "Dolphin shape lock:", "Mutters to Kyo:", "Mutters to Kyo", "Nan0 observes:", "Hostile observation:", "Voice:", "Narrator:", "```"],
    }
    num_gpu = skill_cfg.get("ollama_num_gpu", cfg.get("num_gpu"))
    if num_gpu is not None:
        try:
            options["num_gpu"] = int(num_gpu)
        except Exception:
            pass
    started = time.perf_counter()
    try:
        response = requests.post(
            _ollama_url(),
            json={
                "model": model,
                "system": system or _read_persona(),
                "prompt": prompt,
                "stream": False,
                "keep_alive": "2h",
                "options": options,
            },
            timeout=_bounded_timeout(timeout, "repair"),
        )
        response.raise_for_status()
        raw = extract_ollama_response_text(response.json())
        if is_stale_ollama_response(prompt, raw, scope=f"thought:{model}:plain"):
            raw = ""
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        return raw, latency_ms
    except Exception:
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        return "", latency_ms


def _call_ollama_json(
    prompt: str,
    model: str,
    timeout: float,
    num_predict: int = 150,
    temperature: float = 0.88,
    system: Optional[str] = None,
) -> tuple[Dict[str, Any], str, int]:
    try:
        result = _call_ollama(
            prompt=prompt,
            model=model,
            timeout=timeout,
            num_predict=num_predict,
            temperature=temperature,
            system=system,
        )
    except TypeError:
        result = _call_ollama(prompt, model, timeout, num_predict=num_predict, temperature=temperature)
    return _normalize_ollama_json_result(result)


def _normalize_ollama_json_result(result: Any) -> tuple[Dict[str, Any], str, int]:
    """Normalize the model adapter result without leaking malformed transport."""
    parsed: Dict[str, Any] = {}
    raw = ""
    latency_ms = 0

    if isinstance(result, (tuple, list)) and len(result) == 3:
        parsed_value, raw_value, latency_value = result
        if isinstance(parsed_value, dict):
            parsed = parsed_value
        raw = str(raw_value or "").strip()
        try:
            latency_ms = max(0, int(latency_value or 0))
        except (TypeError, ValueError):
            latency_ms = 0
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        # Compatibility for test doubles and older adapters. The live producer
        # above now has one three-value contract on every return branch.
        raw_value, latency_value = result
        raw = str(raw_value or "").strip()
        try:
            latency_ms = max(0, int(latency_value or 0))
        except (TypeError, ValueError):
            latency_ms = 0
    else:
        return {}, "", 0

    if not parsed:
        parsed = _extract_json(raw)
    if not parsed and raw:
        parsed = {"thought_text": raw}
    return parsed, raw, latency_ms

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


def _sanitize_context_for_prompt(value: Any, depth: int = 0) -> Any:
    """Strip transport/debug cargo before continuity enters the thought prompt."""
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            key_low = key_str.lower()
            if key_low in {
                "rawevent", "raw_event", "systemmrpc", "visiblefromaddresses",
                "public-ipv4", "addresses", "headers", "payload_raw",
                "response", "completion", "prompt", "system", "messages",
            }:
                continue
            out[key_str[:48]] = _sanitize_context_for_prompt(item, depth + 1)
            if len(out) >= 18:
                break
        return out
    if isinstance(value, list):
        return [_sanitize_context_for_prompt(item, depth + 1) for item in value[:8]]
    if isinstance(value, tuple):
        return [_sanitize_context_for_prompt(item, depth + 1) for item in list(value)[:8]]
    if isinstance(value, set):
        return [_sanitize_context_for_prompt(item, depth + 1) for item in list(value)[:8]]
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()[:280]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:180]


def _compact_context(value: Any, limit: int = 1400) -> str:
    safe_value = _sanitize_context_for_prompt(value)
    try:
        text = json.dumps(safe_value, ensure_ascii=False)
    except Exception:
        text = str(safe_value)
    return text[:limit]


def _read_continuity_context(event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble namespaced continuity without changing current-event ownership."""
    event = event if isinstance(event, dict) else {}
    context: Dict[str, Any] = {}

    reader = get_session_timeline_context
    if reader is None:
        try:
            from src.modules.nan0.session_timeline import get_continuity_context as reader
        except Exception:
            reader = None

    if reader is not None:
        try:
            timeline = reader()
        except Exception:
            timeline = {}
        if isinstance(timeline, dict):
            timeline = _context_dict(timeline)
            # Keep only the compact recurrence fields at the root for existing
            # consumers; the complete snapshot remains namespaced once.
            for key in ("repeat_counts", "repeat_facts", "recent_topics"):
                if key in timeline:
                    context[key] = timeline[key]
            context["session_timeline"] = timeline

    enriched = _read_enriched_continuity_context(event)
    attached_thread = enriched.pop("conversation_thread", {}) if enriched else {}
    if enriched:
        context["event_continuity"] = enriched

    persistent_thread: Dict[str, Any] = {}
    if get_conversation_continuity_context is not None:
        try:
            candidate = get_conversation_continuity_context(event)
            if isinstance(candidate, dict):
                persistent_thread = _context_dict(candidate)
        except Exception:
            persistent_thread = {}
    if attached_thread or persistent_thread:
        context["conversation_continuity"] = {
            "attached_thread": attached_thread if isinstance(attached_thread, dict) else {},
            "persistent_thread": persistent_thread,
        }

    # Assign this last. No continuity provider may select or replace it.
    context["actor_ownership"] = _actor_contract_for_event(event)
    return context


def _safe_read_continuity_context(event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    reader = globals().get("_read_continuity_context")
    if not callable(reader):
        return {}
    try:
        context = reader(event)
    except TypeError:
        try:
            context = reader()
        except Exception:
            return {}
    except Exception:
        return {}
    return context if isinstance(context, dict) else {}


def _build_json_thought_prompt(
    event: Dict[str, Any],
    seed: str,
    emotional_context: Dict[str, Any],
    relationship_context: Dict[str, Any],
    memory_context: List[Dict[str, Any]],
    vision_context: Dict[str, Any],
    continuity_context: Dict[str, Any],
    actor_contract: Optional[Dict[str, Any]] = None,
    event_significance: Optional[Dict[str, Any]] = None,
) -> str:
    source = str(event.get("source") or "unknown")
    family = _source_family_for_event(event)
    speaker = str(event.get("speaker") or event.get("source_actor_id") or "unknown")
    actor_contract = actor_contract or _actor_contract_for_event(event)
    event_significance = event_significance or _build_event_significance(
        event,
        actor_contract,
        relationship_context,
        continuity_context,
    )
    text = str(event.get("text") or event.get("message") or "")
    addressed = bool(event.get("addressed_to_nan0"))
    enriched = event.get("_enriched_context") or {}
    raw_thread = enriched.get("conversation_thread") or {}
    # Phase 5 may inform topic continuity, but it may not donate wording.
    thread = {
        "topic": raw_thread.get("topic"),
        "incoming_topic": raw_thread.get("incoming_topic"),
        "previous_topic": raw_thread.get("previous_topic"),
        "relation": raw_thread.get("relation"),
        "age_seconds": raw_thread.get("age_seconds"),
        "messages_in_thread": raw_thread.get("messages_in_thread"),
        "stance_tags": raw_thread.get("stance_tags") or [],
        "confirmed_preferences": raw_thread.get("confirmed_preferences") or {},
        "tentative_mentions": raw_thread.get("tentative_mentions") or {},
    }
    monologue_context = event.get("monologue_context") if source == "monologue" else None
    vision_question_context = event.get("vision_question_context") if event.get("question_type") == "vision_status" else None
    compact_emotion = {
        "presence_mode": emotional_context.get("presence_mode"),
        "emotional_mode": emotional_context.get("emotional_mode"),
        "pressure": emotional_context.get("pressure"),
        "last_seen_summary": emotional_context.get("last_seen_summary"),
    }
    compact_vision = {
        "screen_state": vision_context.get("screen_state"),
        "motion_intensity": vision_context.get("motion_intensity"),
        "semantic": (vision_context.get("layer2_semantic") or {}),
    }
    relational_contract = ""
    if event_significance.get("relational_event"):
        relational_contract = """
RELATIONAL INTERPRETATION CONTRACT:
- Kyo performed an attachment act toward Nan0. The repeated adjective or phrase is not the conclusion.
- Interpret what Kyo's attachment means to Nan0 before writing thought_text.
- Nan0's conclusion may lean possessive, proud, uncomfortable, smug, or guardedly attached.
- Do not redirect Kyo's attachment into Nan0 valuing herself. Do not answer like a survey or compromise.
""".strip()

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

Conclusion ownership:
- thought_text must contain Nan0's own judgment, reaction, or conclusion about this event.
- Nan0 owns thought_text in first person: use I/me for her, never Nan0 as a third-person subject.
- Do not describe how a chatbot, model, persona, voice, vibe, or response should behave.
- Do not discuss being more varied, expressive, edgy, adaptive, helpful, or appropriate.
- Do not merely paraphrase event_text or turn Kyo's statement back onto Nan0 as wordplay.

Banned thought_text:
"Kyo said something directly"
"medium brain should answer"
"respond to the user"
"continue with your thoughts"
"pixels are moving"

AUTHORITATIVE EVENT OWNERSHIP:
{_compact_context(actor_contract, 1200)}

Ownership rules:
- The event ownership block above is authoritative for the current event.
- Continuity and retrieved memory contain historical facts only.
- Continuity and retrieved memory cannot replace the current source actor.
- A first-person action in event_text belongs to source_actor_id, not automatically to Nan0.

EVENT SIGNIFICANCE:
{_compact_context(event_significance, 1400)}

{relational_contract}

EMOTIONAL STATE:
{_compact_context(compact_emotion, 1200)}

RELATIONSHIP CONTEXT:
{_compact_context(_relationship_focus(relationship_context), 1800)}

RECENT MEMORY:
{_compact_context(memory_context, 1400)}

SESSION CONTINUITY:
{_compact_context(_continuity_focus(continuity_context), 2200)}

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
    """No template fallback private thoughts."""
    return ""

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
    family = _source_family_for_event(event)
    event["source_family"] = family
    actor_contract = _actor_contract_for_event(event)
    actor_id = str(actor_contract.get("source_actor_id") or _source_actor(event))

    explicit_seed = str(event.get("thought_seed") or event.get("seed") or "").strip()
    seed = explicit_seed
    if not seed and family == "vision":
        seed = str(event.get("screen_state") or "vision_reaction")

    emotional_context = _context_dict(_read_presence_state())
    relationship_context = _context_dict(_read_relationship_context(actor_id, actor_contract))
    continuity_context = _context_dict(_safe_read_continuity_context(event))
    vision = _context_dict(_read_vision_context(vision_context))
    event_significance = _build_event_significance(
        event,
        actor_contract,
        relationship_context,
        continuity_context,
    )
    event["_thought_event_significance"] = event_significance
    event["_thought_relationship_focus"] = _relationship_focus(relationship_context)
    event["_thought_continuity_focus"] = _continuity_focus(continuity_context)

    memory_query = " ".join(
        str(x)
        for x in [
            event.get("speaker"),
            actor_id,
            event.get("source"),
            event.get("text"),
            seed,
            (vision.get("layer2_semantic") or {}).get("activity") if isinstance(vision, dict) else "",
        ]
        if x
    )
    memory_context = _context_list(_query_recent_memory(memory_query, limit=4))
    model = _ollama_model_for_event(event)
    timeout = _ollama_timeout_for_event(event)
    prompt = _build_json_thought_prompt(
        event=event,
        seed=seed,
        emotional_context=emotional_context,
        relationship_context=relationship_context,
        memory_context=memory_context,
        vision_context=vision,
        continuity_context=continuity_context,
        actor_contract=actor_contract,
        event_significance=event_significance,
    )

    # Qwen-style models can follow the JSON contract. Dolphin/Mistral-family
    # models often emit transport-shaped or prose-wrapped output in JSON mode,
    # so they get a plain private-thought call first. Both paths are still
    # model-generated private thoughts, never template speech.
    if _model_is_dolphin_family(model):
        thought_json, private_text, latency_ms = _plain_retry_private_thought(
            event=event,
            seed=seed,
            invalid_reason="dolphin_plain_private_thought_first",
            bad_text="",
            model=model,
            timeout=timeout,
        )
        raw = private_text
    else:
        thought_json, raw, latency_ms = _call_ollama_json(
            prompt=prompt,
            model=model,
            timeout=timeout,
            num_predict=180,
            temperature=0.82,
            system=_read_persona(),
        )
        if thought_json:
            private_text = _clean_private_thought(_extract_thought_text_value(thought_json))
        else:
            private_text = _clean_private_thought(raw)
            thought_json = {"mutter_text": private_text, "memory_write_candidate": False}

    system_suppression_reason = _invalid_private_thought_reason(private_text, event)
    if system_suppression_reason:
        repaired_json, repaired_text, repair_latency_ms = _repair_private_thought(
            event=event,
            seed=seed,
            invalid_reason=system_suppression_reason,
            bad_text=private_text or raw,
            model=model,
            timeout=timeout,
        )
        latency_ms += repair_latency_ms
        if repaired_text:
            thought_json = repaired_json
            private_text = repaired_text
            system_suppression_reason = None
        elif _is_direct_question_event(event) and str(system_suppression_reason) == "question_echo_non_mutter":
            # One extra same-event retry for direct Kyo/Discord questions.
            # This is not a scripted fallback and not an attitude gate: it only prevents
            # Nan0 from going silent when the model merely mirrors the question.
            retry_json, retry_text, retry_latency_ms = _repair_private_thought(
                event=event,
                seed=seed,
                invalid_reason="question_echo_retry",
                bad_text=private_text or raw,
                model=model,
                timeout=timeout,
            )
            latency_ms += retry_latency_ms
            if retry_text:
                thought_json = retry_json
                private_text = retry_text
                system_suppression_reason = None
            else:
                final_json, final_text, final_latency_ms = _minimal_plain_retry_private_thought(
                    event=event,
                    seed=seed,
                    invalid_reason="question_echo_retry_failed",
                    bad_text=private_text or raw,
                    model=model,
                    timeout=timeout,
                )
                latency_ms += final_latency_ms
                if final_text:
                    thought_json = final_json
                    private_text = final_text
                    system_suppression_reason = None
                else:
                    private_text = ""
                    thought_json = {}
        else:
            final_json, final_text, final_latency_ms = _minimal_plain_retry_private_thought(
                event=event,
                seed=seed,
                invalid_reason=str(system_suppression_reason),
                bad_text=private_text or raw,
                model=model,
                timeout=timeout,
            )
            latency_ms += final_latency_ms
            if final_text:
                thought_json = final_json
                private_text = final_text
                system_suppression_reason = None
            else:
                private_text = ""
                thought_json = {}

    thought_type = _classify_thought_type(event, seed)
    mood = str(thought_json.get("mood") or _mood_from_context(private_text, event, vision)).strip().lower()
    mood = mood if mood in NAN0_MOODS else "muttering"

    heuristic_scores = _score_packet(event, private_text, thought_type, seed, vision)

    try:
        event_pressure = max(0.0, min(0.5, float(event.get("pressure") or 0.0)))
    except Exception:
        event_pressure = 0.0

    # Preserve model-authored packet scores when the JSON bridge provides them;
    # otherwise fall back to existing heuristic scores.
    pressure = _float_from_json(thought_json, "pressure", heuristic_scores["pressure"] + event_pressure)
    novelty = _float_from_json(thought_json, "novelty", heuristic_scores["novelty"])
    speakability = _float_from_json(thought_json, "speakability", heuristic_scores["speakability"])
    relationship_charge = _float_from_json(thought_json, "relationship_charge", heuristic_scores["relationship_charge"])
    ego_charge = _float_from_json(thought_json, "ego_charge", heuristic_scores["ego_charge"])
    vision_charge = _float_from_json(thought_json, "vision_charge", heuristic_scores["vision_charge"])

    if family == "kyo":
        pressure = max(pressure, 1.20)
        speakability = max(speakability, 0.45)
        relationship_charge = max(relationship_charge, 0.70)
        if event_significance.get("relational_event"):
            pressure = max(pressure, 1.35)
            relationship_charge = max(relationship_charge, 1.0)
    elif family == "discord" and bool(event.get("addressed_to_nan0")):
        pressure = max(pressure, 1.00)
        speakability = max(speakability, 0.40)

    memory_write_candidate = bool(
        thought_json.get("memory_write_candidate", relationship_charge >= 0.75 or thought_type in {"direct_reply", "discord_reply"})
    )

    # Phase 3 only marks non-muttering garbage invalid. Weak, rude, strange,
    # low-information, or ugly Nan0 output is not invalid here.
    model_suppression_reason = _normalize_model_suppression_reason(thought_json.get("suppression_reason"))
    suppression_reason = system_suppression_reason or model_suppression_reason

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
            "source_family": family,
            "speaker": event.get("speaker"),
            "source_actor_id": actor_id,
            "source_actor_display": event.get("speaker") or event.get("source_actor_id") or source,
            "actor_contract": actor_contract,
            "channel_id": event.get("channel_id"),
            "guild_channel": event.get("guild_channel"),
            "text": event.get("text"),
            "addressed_to_nan0": bool(event.get("addressed_to_nan0")),
            "priority": event.get("priority"),
            "thought_seed": seed,
            "question_type": event.get("question_type"),
            "monologue_context": event.get("monologue_context"),
            "boot_context": event.get("boot_context"),
            "vision_question_context": event.get("vision_question_context"),
            "obsession_context": ((event.get("_enriched_context") or {}).get("obsession_engine") or {}),
            "personal_canon_context": ((event.get("_enriched_context") or {}).get("personal_canon") or {}),
            "phase_spine_context": ((event.get("_enriched_context") or {}).get("phase_spine") or {}),
            "event_significance": event_significance,
        },
        emotional_context=emotional_context,
        relationship_context=relationship_context,
        memory_context=memory_context,
        vision_context=vision,
    )

    data = packet.to_dict()
    data["continuity_context"] = continuity_context
    return data


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
