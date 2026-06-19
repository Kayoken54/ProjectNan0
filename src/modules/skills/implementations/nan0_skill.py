from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.modules.skills.base_skill import BaseSkill
from src.modules.skills.implementations.nan0_cognition_router_v1 import (
    clean_nan0_line,
    route_thought,
    thought_packet_to_event,
)
from src.modules.skills.implementations.nan0_thought_engine_v3 import (
    BANNED_VISION_FILLER,
    generate_inner_thought_packet,
    validate_inner_thought_packet,
)
from src.utils.logger import get_logger
from src.modules.nan0.session_timeline import record_session_event, record_speech_packet, record_thought_packet

logger = get_logger("bea.skills.nan0")

MOODS = [
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
]

SpeechDecision = Dict[str, Any]
InnerThoughtPacket = Dict[str, Any]


class Nan0Skill(BaseSkill):
    def __init__(self, name, config, context):
        super().__init__(name, config, context)
        self.brain = context
        cfg = self.skill_config

        self.model = cfg.get("fast_model", getattr(config, "ollama_model", "qwen2.5:3b"))
        self.ollama_host = getattr(config, "ollama_host", "http://localhost:11434").rstrip("/")
        self.timeout = float(
            cfg.get(
                "medium_lane_timeout",
                cfg.get("latency_budget", getattr(config, "ollama_timeout", 30.0)),
            )
        )
        self.deep_lane_timeout = float(cfg.get("deep_lane_timeout", 200.0))
        self.no_fallback_on_timeout = bool(cfg.get("no_fallback_on_timeout", True))
        self.social_timeout_ack = bool(cfg.get("social_timeout_ack", True))
        self.respond_to_all_discord = bool(cfg.get("respond_to_all_discord", False))

        self.min_autonomous_gap = float(cfg.get("min_autonomous_gap", 120.0))
        self.max_autonomous_gap = float(cfg.get("max_autonomous_gap", 240.0))
        self.min_speech_gap = float(cfg.get("min_speech_gap", 16.0))
        self.pressure_threshold = float(cfg.get("pressure_threshold", 1.55))
        self.speakability_threshold = float(cfg.get("speakability_threshold", 0.35))

        self.conversation_window_seconds = float(cfg.get("conversation_window_seconds", 90.0))
        self.fast_lane_speech_enabled = bool(cfg.get("fast_lane_speech_enabled", False))
        self.fast_lane_body_only = bool(cfg.get("fast_lane_body_only", True))
        self.fast_lane_pressure_scale = float(cfg.get("fast_lane_pressure_scale", 0.12))
        self.fast_lane_emergency_pressure = float(cfg.get("fast_lane_emergency_pressure", 1.60))

        self.deep_shutdown_summary_enabled = bool(cfg.get("deep_shutdown_summary_enabled", True))
        self.deep_summary_path = Path(cfg.get("deep_summary_path", "data/nan0/session_summaries.jsonl"))
        self.deep_summary_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_line_chars = int(cfg.get("max_line_chars", 125))
        self.kyo_inbox = Path(cfg.get("kyo_voice_inbox", "data/input/kyo_voice_inbox.jsonl"))
        self.discord_inbox = Path(cfg.get("discord_inbox", "data/input/discord_voice_inbox.jsonl"))
        self.vision_state_path = Path(cfg.get("vision_state_path", "data/vision/nan0_vision_stack_state.json"))
        self.state_path = Path(cfg.get("presence_state_path", "data/nan0/presence_state.json"))
        self.obsession_state_path = Path(cfg.get("obsession_state_path", "data/nan0/obsession_state.json"))
        self.personal_canon_state_path = Path(cfg.get("personal_canon_state_path", "data/nan0/personal_canon_state.json"))
        self.thread_state_path = Path(cfg.get("thread_state_path", cfg.get("thought_momentum_path", "data/nan0/thought_momentum.json")))
        self.obsession_enabled = bool(cfg.get("obsession_engine_enabled", True))
        self.personal_canon_enabled = bool(cfg.get("personal_canon_enabled", True))
        self.obsession_max_topics = int(cfg.get("obsession_max_topics", 6))
        self.personal_canon_max_items = int(cfg.get("personal_canon_max_items", 12))
        self.obsession_decay_seconds = float(cfg.get("obsession_decay_seconds", 900.0))
        self.obsession_min_interest = float(cfg.get("obsession_min_interest", 0.18))
        self.phase_spine_enabled = bool(cfg.get("phase_spine_enabled", True))
        self.worldview_filter_enabled = bool(cfg.get("worldview_filter_enabled", True))
        self.context_over_time_window = int(cfg.get("context_over_time_window", 12))
        self.speech_persona_path = Path(cfg.get("speech_persona_path", "data/prompts/nan0_speech_persona.txt"))
        self.speech_debug_enabled = bool(cfg.get("speech_debug_enabled", False))
        self.speech_filter_mode = str(cfg.get("speech_filter_mode", "normal")).strip().lower()
        if self.speech_filter_mode not in {"normal", "raw", "debug_only"}:
            self.speech_filter_mode = "normal"
        self.speech_debug_path = Path(cfg.get("speech_debug_path", "data/nan0/speech_debug.jsonl"))
        self.show_suppressed_thoughts = bool(cfg.get("show_suppressed_thoughts", True))
        self.show_filter_changes = bool(cfg.get("show_filter_changes", True))
        self.speech_debug_path.parent.mkdir(parents=True, exist_ok=True)

        for path in [
            self.kyo_inbox,
            self.discord_inbox,
            self.vision_state_path,
            self.state_path,
            self.obsession_state_path,
            self.personal_canon_state_path,
            self.speech_persona_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

        self.vision = None
        self.pressure = 0.0
        self.last_spoken_at = 0.0
        self.last_kyo_heard_at = 0.0
        self.last_discord_heard_at = 0.0
        self.last_vision_event_at = 0.0
        self.last_monologue_at = 0.0
        self.last_seen_summary = "the screen is quiet"
        self.last_fast_state: Dict[str, Any] = {}

        self.recent_events: List[Dict[str, Any]] = []
        self._tasks: List[asyncio.Task] = []
        self._last_file_positions: Dict[str, int] = {}
        self._speak_lock = asyncio.Lock()
        self._recent_lines: List[str] = []
        self._recent_line_times: Dict[str, float] = {}

        # Runtime arbitration state. This does not make Nan0 obedient; it only
        # prevents monologue/social pressure from stealing a fresh Kyo thread.
        self.started_at = time.time()
        self.active_question_window_seconds = float(cfg.get("active_question_window_seconds", 180.0))
        self.social_pressure_cooldown_seconds = float(cfg.get("social_pressure_cooldown_seconds", 90.0))
        self.monologue_idle_gap_seconds = float(cfg.get("monologue_idle_gap_seconds", max(self.max_autonomous_gap, 300.0)))
        self._active_question: Optional[Dict[str, Any]] = None
        self._last_social_pressure_at = 0.0
        self._reply_in_progress = False
        self._last_direct_input_at = 0.0

        self.speech_history_lines = int(cfg.get("speech_history_lines", 3))
        self.speech_recent_events = int(cfg.get("speech_recent_events", 2))
        self._thought_debug_by_id: Dict[str, Dict[str, Any]] = {}

        self.finalizer = Nan0SpeechFinalizer(max_chars=self.max_line_chars, filter_mode=self.speech_filter_mode)

    def initialize(self):
        logger.info("Nan0Skill initialized: mutter-first speech enforcement active.")

    async def start(self):
        if not self.enabled or self.is_active:
            return

        await super().start()
        self.started_at = time.time()
        self.last_monologue_at = self.started_at
        self._prime_inbox_positions()

        self._tasks = [
            asyncio.create_task(self._inbox_loop()),
            asyncio.create_task(self._presence_loop()),
            asyncio.create_task(self._state_writer_loop()),
        ]

        await self._run_boot_presence()

        logger.info("Nan0Skill started: all speech now requires a mutter origin id.")

    def _build_boot_event(self) -> Dict[str, Any]:
        """Create the boot event consumed by the normal cognition pipeline."""
        return {
            "event_id": f"boot_{uuid.uuid4().hex}",
            "source": "boot",
            "speaker": "Nan0",
            "source_actor_id": "nan0",
            "text": "Nan0 has just booted into the room.",
            "message": "Nan0 has just booted into the room.",
            "thought_seed": "boot_presence",
            "addressed_to_nan0": False,
            "priority": "high",
            "timestamp": time.time(),
            "boot_context": {
                "rule": "Create Nan0's own private boot thought. Do not create a startup report.",
                "must_have_private_text": True,
            },
        }

    async def _run_boot_presence(self) -> Optional[SpeechDecision]:
        """Run boot through thought generation, routing, and normal speech."""
        try:
            boot_event = self._build_boot_event()
            boot_packet = await self._create_inner_thought(boot_event)
            valid, invalid_reason = validate_inner_thought_packet(boot_packet, expected_source="boot")
            if not valid:
                logger.warning(f"Nan0 boot presence skipped: {invalid_reason}")
                return None

            thought_id = str(boot_packet["thought_id"])
            routed = route_thought(boot_packet)
            if routed.get("thought_id") != thought_id:
                logger.warning("Nan0 boot presence skipped: router did not preserve thought_id")
                return None
            if routed.get("decision") in {"suppress", "body_only", "memory_only", "defer"}:
                self._record_speech_debug(boot_packet, routed, debug_stage="router_suppressed")
                return routed

            decision = await self._generate_line(boot_packet)
            if decision.get("thought_id") != thought_id:
                logger.warning("Nan0 boot presence skipped: speech decision lost thought_id")
                return None
            if decision.get("decision") == "speak":
                await self._speak_decision(decision, reason="boot")
            return decision
        except Exception as exc:
            logger.warning(f"Nan0 boot presence skipped: {exc}")
            return None

    async def stop(self):
        self.is_active = False

        for task in self._tasks:
            task.cancel()

        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(f"Nan0 task stopped with non-fatal shutdown error: {exc}")

        self._tasks.clear()

        if self.deep_shutdown_summary_enabled:
            await self._run_deep_shutdown_summary()

        await super().stop()

    async def update(self):
        return

    def set_vision_skill(self, vision_skill):
        self.vision = vision_skill

    async def on_user_message(self, message: str, user: str = "Kyo"):
        await self._handle_social_event(
            {
                "event_id": f"kyo_text_{uuid.uuid4().hex}",
                "source": "kyo_text",
                "speaker": user or "Kyo",
                "source_actor_id": "kyo",
                "text": message,
                "addressed_to_nan0": True,
                "priority": "high",
                "timestamp": time.time(),
            }
        )

    async def on_chat_message(self, message: str, chatter: str):
        await self.on_discord_message(chatter, message, source="discord")

    async def on_discord_message(self, speaker: str, text: str, source: str = "discord"):
        await self._handle_social_event(
            {
                "event_id": f"discord_{uuid.uuid4().hex}",
                "source": source or "discord",
                "speaker": speaker or "Friend",
                "source_actor_id": speaker or "discord_friend",
                "text": text,
                "addressed_to_nan0": self._is_addressed_to_nan0(text),
                "priority": "medium",
                "timestamp": time.time(),
            }
        )

    async def handle_external_message(
        self,
        message: str,
        actor: str = "Kyo",
        source: str = "kyo",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Official external input door for Nan0.

        Sacred runtime rule:
        - brain.py, Discord, typed input, and future STT may enter here.
        - They may not call the generic brain LLM path for Nan0 dialogue.
        - This method converts outside input into a Nan0 event, then lets
          Nan0Skill create an inner thought, route it, and decide speech.
        """
        text = (message or "").strip()
        if not text:
            return "muttering", ""

        metadata = metadata or {}
        normalized_source = (source or "kyo").strip().lower()
        actor_name = (actor or metadata.get("speaker") or "Kyo").strip()

        if normalized_source in {"kyo", "kyo_text", "kyo_voice", "kyo_mic", "manual"}:
            event_source = "kyo_voice" if "voice" in normalized_source or "mic" in normalized_source else "kyo_text"
            source_actor_id = "kyo"
            speaker = actor_name or "Kyo"
            addressed = True
            priority = "high"
        elif "discord" in normalized_source:
            event_source = "discord_voice" if "voice" in normalized_source or "audio" in normalized_source else "discord_text"
            source_actor_id = actor_name or "discord_friend"
            speaker = actor_name or "Friend"
            addressed = bool(metadata.get("addressed_to_nan0", self._is_addressed_to_nan0(text)))
            priority = "medium"
        elif "vision" in normalized_source or normalized_source in {"screen", "fast_eyes"}:
            event_source = "vision_stack_v1"
            source_actor_id = "screen"
            speaker = "screen"
            addressed = False
            priority = "low"
        else:
            event_source = normalized_source
            source_actor_id = actor_name or normalized_source or "unknown"
            speaker = actor_name or source_actor_id
            addressed = bool(metadata.get("addressed_to_nan0", False))
            priority = metadata.get("priority", "medium")

        previous_packet = getattr(self.brain, "last_nan0_speech_packet", None)
        previous_thought_id = previous_packet.get("thought_id") if isinstance(previous_packet, dict) else None

        event = {
            "event_id": str(metadata.get("event_id") or f"{event_source}_{uuid.uuid4().hex}"),
            "source": event_source,
            "speaker": speaker,
            "source_actor_id": source_actor_id,
            "text": text,
            "addressed_to_nan0": addressed,
            "priority": priority,
            "timestamp": float(metadata.get("timestamp") or time.time()),
            "payload": metadata.get("payload") or {},
        }

        await self._handle_social_event(event)

        packet = getattr(self.brain, "last_nan0_speech_packet", None)
        if isinstance(packet, dict) and packet.get("thought_id") and packet.get("thought_id") != previous_thought_id:
            return self._normalize_mood(packet.get("mood") or "normal"), str(packet.get("line_text") or "")

        return "muttering", ""

    async def handle_external_event(self, event: Dict[str, Any]) -> Tuple[str, str]:
        """Official event-object entry point for future non-text sources."""
        if not isinstance(event, dict):
            raise TypeError("handle_external_event requires an event dict")
        return await self.handle_external_message(
            str(event.get("text") or event.get("message") or ""),
            actor=str(event.get("speaker") or event.get("source_actor_id") or "unknown"),
            source=str(event.get("source") or "external"),
            metadata=event,
        )

    async def _handle_social_event(self, event: Dict[str, Any]):
        text = (event.get("text") or "").strip()
        speaker = (event.get("speaker") or "someone").strip()
        source = self._normalize_event_source(event.get("source", "unknown"))
        family = self._source_family(source)
        event["source"] = source
        event["source_family"] = family

        if not text:
            return

        if self._is_fake_patch_event(speaker, text):
            logger.warning(f"Dropped old fake patch event: {speaker}: {text}")
            return

        addressed = bool(event.get("addressed_to_nan0"))

        if family == "kyo":
            self.last_kyo_heard_at = time.time()
            event_type = "question" if self._looks_like_question(text, addressed=True) else "message"
            event.setdefault("event_type", event_type)
            logger.info(f"Nan0 perceived Kyo [{source}/{event_type}]: {text}")
            self.pressure += 0.95
        elif family == "discord":
            self.last_discord_heard_at = time.time()
            event_type = "question" if self._looks_like_question(text, addressed=addressed) else "message"
            event.setdefault("event_type", event_type)
            logger.info(f"Nan0 heard real Discord user {speaker}: {text}")
            self.pressure += 1.05 if addressed else 0.55
        else:
            logger.info(f"Nan0 heard {source} {speaker}: {text}")
            self.pressure += 0.35

        if addressed:
            self.pressure += 0.65

        if family in {"kyo", "discord"}:
            if self._looks_like_question(text, addressed=addressed):
                self._register_active_question(event)

        self._remember_event(event)

        if addressed or family == "kyo" or (family == "discord" and self.respond_to_all_discord):
            self._last_direct_input_at = time.time()
            await self._respond_to_event(event)


    def _normalize_event_source(self, source: Any) -> str:
        raw = str(source or "unknown").strip().lower()
        if raw in {"kyo", "manual", "typed", "text", "console", "manual_command"}:
            return "kyo_text"
        if raw in {"voice", "mic", "kyo_mic"}:
            return "kyo_voice"
        if raw in {"discord", "discord_chat", "chat"}:
            return "discord_text"
        if raw in {"discord_audio", "discord_mic"}:
            return "discord_voice"
        if raw in {"vision", "screen", "fast_eyes", "vision_pressure"}:
            return "vision_stack_v1" if raw != "vision" else "vision"
        if raw in {"idle_presence", "pressure_idle", "social_pressure"}:
            return raw
        if raw in {"boot", "shutdown"}:
            return raw
        return raw or "unknown"

    def _source_family(self, source: Any) -> str:
        raw = str(source or "unknown").strip().lower()
        if raw in {"kyo", "kyo_text", "kyo_voice", "kyo_mic", "manual", "manual_command", "typed", "text", "console", "mic", "voice"} or raw.startswith("kyo_"):
            return "kyo"
        if "discord" in raw:
            return "discord"
        if raw in {"vision", "vision_stack_v1", "screen", "fast_eyes", "vision_pressure"} or "vision" in raw:
            return "vision"
        if raw in {"monologue", "proactive", "social_pressure", "idle_presence", "pressure_idle"}:
            return "proactive"
        if raw in {"boot", "system", "shutdown"}:
            return "system"
        return "external"

    def _load_thread_state(self) -> Dict[str, Any]:
        return self._load_json_state(
            self.thread_state_path,
            {
                "version": 2,
                "updated_at": 0.0,
                "active_topic": "",
                "active_topic_strength": 0.0,
                "active_topic_started_at": 0.0,
                "last_event_text": "",
                "last_private_mutter": "",
                "last_source": "",
                "stance_tags": [],
                "confirmed_preferences": {},
                "tentative_mentions": {},
                "messages_in_thread": 0,
                "topic_history": [],
            },
        )

    def _save_thread_state(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = time.time()
        self._save_json_state(self.thread_state_path, state)

    def _topic_from_text_for_thread(self, text: str, active_topic: str = "") -> str:
        low = (text or "").lower()
        if re.search(r"\bani?o?me\b", low) or "anime" in low or "animé" in low:
            return "anime"
        if "airsoft" in low:
            return "airsoft"
        if "what are you feeling" in low or "how are you feeling" in low or "how do you feel" in low or "what do you feel" in low:
            return "nan0_state"
        if "what are you muttering" in low or "what are you doing" in low:
            return "nan0_state"
        if active_topic and re.search(r"\b(favou?rite|favoirte|which one|what one|why that|right now|what kind|what type|do you like it)\b", low):
            return active_topic
        tokens = [t for t in re.findall(r"[a-z][a-z0-9_'-]{2,}", low) if t not in {
            "nan0", "nano", "kyo", "you", "your", "are", "the", "and", "what", "why", "how", "when", "where", "like", "feel", "about", "hello", "hey", "tell", "favoirte", "favorite", "right", "now"
        }]
        return tokens[-1] if tokens else ""

    def _build_conversation_thread_context(self, event: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        state = self._load_thread_state()
        text = str(event.get("text") or event.get("message") or "")
        source = self._normalize_event_source(event.get("source") or "unknown")
        active_topic = self._canonicalize_topic(state.get("active_topic") or "")
        candidate = self._canonicalize_topic(self._topic_from_text_for_thread(text, active_topic=active_topic))
        age = now - float(state.get("updated_at") or 0.0) if state.get("updated_at") else 999999.0
        family = self._source_family(source)
        direct_social = family in {"kyo", "discord"}
        same_topic = bool(candidate and active_topic and candidate == active_topic and age <= 600.0)
        followup = bool(direct_social and active_topic and candidate == active_topic and age <= 600.0)
        unrelated = bool(candidate and active_topic and candidate != active_topic)
        relation = "same_topic" if same_topic else "followup" if followup else "new_topic" if candidate else "none"
        if unrelated:
            relation = "new_topic"
        if candidate in {"nan0_state", "airsoft", "game"}:
            relation = "new_topic"

        return {
            "enabled": True,
            "topic": active_topic if relation in {"same_topic", "followup"} else candidate,
            "incoming_topic": candidate,
            "previous_topic": active_topic,
            "relation": relation,
            "age_seconds": round(age, 1),
            "messages_in_thread": int(state.get("messages_in_thread") or 0),
            "stance_tags": list(state.get("stance_tags") or [])[:8] if relation in {"same_topic", "followup"} else [],
            "confirmed_preferences": dict(state.get("confirmed_preferences") or {}) if relation in {"same_topic", "followup"} else {},
            "tentative_mentions": dict(state.get("tentative_mentions") or {}) if relation in {"same_topic", "followup"} else {},
            "last_event_text": "",
            "last_private_mutter": "",
            "rules": [
                "continuity flags only, not wording to repeat",
                "same_topic means avoid clean reset",
                "new_topic means do not drag the old topic in",
                "one mention is tentative, not canon",
            ],
        }

    def _stance_tags_from_private_mutter(self, topic: str, text: str) -> List[str]:
        low = (text or "").lower()
        tags: List[str] = []
        if topic == "anime":
            if any(x in low for x in ["garbage", "stupid", "clich", "trope", "hentai", "teenage"]):
                tags.append("mocking_anime")
            if any(x in low for x in ["machine", "tech", "robot", "mecha", "gadget", "sleek"]):
                tags.append("machine_angle")
            if any(x in low for x in ["battle", "flashy", "shounen", "shonen", "fight"]):
                tags.append("loud_action_angle")
            if any(x in low for x in ["drama", "dramatic", "slice", "life"]):
                tags.append("overdramatic_trash_angle")
            if any(x in low for x in ["cowboy bebop"]):
                tags.append("cowboy_bebop_unconfirmed")
            if any(x in low for x in ["don't watch", "do not watch", "never watch"]):
                tags.append("denial_or_embarrassment")
        if "kyo" in low:
            tags.append("kyo_is_anchor")
        # Preserve order, remove duplicates.
        out = []
        for tag in tags:
            if tag not in out:
                out.append(tag)
        return out[:8]

    def _update_conversation_thread_state(self, event: Dict[str, Any], packet: InnerThoughtPacket, private_text: str, now: float) -> None:
        source = self._normalize_event_source(event.get("source") or "unknown")
        family = self._source_family(source)
        if family not in {"kyo", "discord", "proactive"}:
            return
        state = self._load_thread_state()
        prior_topic = self._canonicalize_topic(state.get("active_topic") or "")
        text = str(event.get("text") or event.get("message") or "")
        candidate = self._canonicalize_topic(self._topic_from_text_for_thread(text, active_topic=prior_topic))
        if not candidate and source in {"monologue", "social_pressure", "proactive"}:
            candidate = prior_topic
        if not candidate:
            return

        relation = "same_topic" if prior_topic and candidate == prior_topic else "new_topic"
        if candidate == "nan0_state" and prior_topic and family == "kyo":
            # Kyo asked about Nan0 directly; do not let the old topic hijack it.
            relation = "new_topic"

        existing_tags = list(state.get("stance_tags") or []) if relation == "same_topic" else []
        new_tags = self._stance_tags_from_private_mutter(candidate, private_text)
        tags = []
        for tag in existing_tags + new_tags:
            if tag not in tags:
                tags.append(tag)

        tentative = dict(state.get("tentative_mentions") or {}) if relation == "same_topic" else {}
        confirmed = dict(state.get("confirmed_preferences") or {}) if relation == "same_topic" else {}
        low = private_text.lower()
        if candidate == "anime" and "cowboy bebop" in low:
            tentative["cowboy_bebop"] = int(tentative.get("cowboy_bebop") or 0) + 1
            if tentative["cowboy_bebop"] >= 2:
                confirmed["favorite_candidate"] = "cowboy_bebop"

        history = list(state.get("topic_history") or [])
        history.insert(0, {"at": now, "topic": candidate, "source": source, "event_text": text[:160]})
        state.update({
            "version": 2,
            "active_topic": candidate,
            "active_topic_strength": 1.0 if source.startswith("kyo") else 0.65,
            "active_topic_started_at": state.get("active_topic_started_at") if relation == "same_topic" and state.get("active_topic_started_at") else now,
            "last_event_text": text[:220],
            "last_private_mutter": private_text[:360],
            "last_source": source,
            "stance_tags": tags[:8],
            "confirmed_preferences": confirmed,
            "tentative_mentions": tentative,
            "messages_in_thread": int(state.get("messages_in_thread") or 0) + 1 if relation == "same_topic" else 1,
            "topic_history": history[:20],
        })
        self._save_thread_state(state)

    def _is_session_preference_event(self, event: Dict[str, Any], private_text: str) -> bool:
        text = (str(event.get("text") or "") + " " + str(private_text or "")).lower()
        return any(x in text for x in ["anime", "aniome", "favorite", "favoirte", "do you like", "what do you like", "cowboy bebop", "airsoft"])

    def _is_fake_patch_event(self, speaker: str, text: str) -> bool:
        low = (text or "").lower().strip()
        sp = (speaker or "").lower().strip()
        return sp == "alex" and low in {
            "nan0 what are you doing?",
            "nano what are you doing?",
            "what are you doing?",
        }

    async def _respond_to_event(self, event: Dict[str, Any]):
        self._reply_in_progress = True
        try:
            text = event.get("text", "")

            if self._asks_vision(text):
                event["question_type"] = "vision_status"
                event["vision_question_context"] = self._build_vision_question_context()
                # Give the thought engine concrete vision data or a clear weak-vision fact.
                vision_line = self._vision_status_private_seed(event["vision_question_context"])
                event["text"] = f"{text}\nVision status: {vision_line}"
                event["message"] = event["text"]

            thought_packet = await self._create_inner_thought(event)
            routed = route_thought(thought_packet)
            if routed.get("decision") in {"suppress", "body_only", "memory_only", "defer"}:
                logger.info(
                    "Nan0 thought routed away from speech: "
                    f"decision={routed.get('decision')} reason={routed.get('reason')}"
                )
                self._record_speech_debug(thought_packet, routed, debug_stage="router_suppressed")
                return

            decision = await self._generate_line(thought_packet)
            # [Discord Bridge] Preserve original source event so Nan0's spoken reply can
            # be mirrored back to the Discord channel without the old :8000 Brain path.
            decision['source_event'] = dict(event)
            decision['channel_id'] = event.get('channel_id')
            decision['guild_channel'] = event.get('guild_channel')
            await self._speak_decision(decision, reason=f"{event.get('source')}_reply")
        finally:
            self._reply_in_progress = False

    def _build_vision_question_context(self) -> Dict[str, Any]:
        context = {
            "last_seen_summary": self.last_seen_summary,
            "last_fast_state": self.last_fast_state,
            "vision_skill_active": self.vision is not None,
        }
        try:
            if self.vision and hasattr(self.vision, "latest_fast_state"):
                context["latest_fast_state"] = getattr(self.vision, "latest_fast_state", None)
            if self.vision and hasattr(self.vision, "latest_state"):
                context["latest_state"] = getattr(self.vision, "latest_state", None)
        except Exception:
            pass
        try:
            if self.vision_state_path.exists():
                context["vision_state_file"] = json.loads(self.vision_state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return context

    def _vision_status_private_seed(self, context: Dict[str, Any]) -> str:
        if not isinstance(context, dict):
            return "Vision context is missing. Do not pretend to see."
        state = context.get("latest_fast_state") or context.get("last_fast_state") or context.get("vision_state_file") or {}
        if not isinstance(state, dict) or not state:
            return "Vision feed is active but not giving a clean current picture. Do not pretend to see details."
        layer1 = state.get("layer1_reflex") or state.get("fast") or state
        layer2 = state.get("layer2_semantic") or {}
        screen_state = layer1.get("screen_state") or state.get("screen_state") or state.get("state") or "unknown"
        brightness = layer1.get("brightness") or state.get("brightness")
        motion = layer1.get("motion_intensity") or state.get("motion_intensity") or state.get("frame_diff")
        activity = layer2.get("activity") or state.get("activity")
        pieces = [f"screen_state={screen_state}"]
        if activity:
            pieces.append(f"activity={activity}")
        if brightness is not None:
            pieces.append(f"brightness={brightness}")
        if motion is not None:
            pieces.append(f"motion={motion}")
        return "Current vision facts: " + "; ".join(pieces) + ". If these are weak, say they are weak instead of inventing details."

    def _background_event_should_yield_to_kyo(self, event: Dict[str, Any]) -> bool:
        """Keep autonomous mutters from speaking over fresh Kyo input."""
        if not isinstance(event, dict):
            return False
        source = str(event.get("source") or "").lower()
        if source.startswith("kyo") or "discord" in source:
            return False
        now = time.time()
        event_ts = float(event.get("timestamp") or event.get("time") or now)
        if getattr(self, "_reply_in_progress", False):
            return True
        if self._active_question_active(now):
            return True
        if getattr(self, "last_kyo_heard_at", 0.0) and now - self.last_kyo_heard_at <= self.conversation_window_seconds:
            return True
        if getattr(self, "_last_direct_input_at", 0.0) >= event_ts - 0.25:
            return True
        return False

    async def _presence_loop(self):
        while self.is_active:
            await asyncio.sleep(2.0)
            now = time.time()
            self._read_fast_eyes_pressure()

            silence = now - self.last_spoken_at if self.last_spoken_at else 999.0
            conversation_mode = self._conversation_mode_active(now)

            if silence > self.min_autonomous_gap and not conversation_mode:
                self.pressure += 0.045

            if self.pressure >= self.pressure_threshold and silence >= self.min_speech_gap:
                if getattr(self, "_reply_in_progress", False):
                    continue
                event = self._build_pressure_event()

                if self._background_event_should_yield_to_kyo(event):
                    logger.info("Nan0 CoherenceGate: background pressure yielded to fresh Kyo input.")
                    self.pressure = max(0.0, self.pressure - 0.45)
                    continue

                if event.get("source") == "social_pressure" and self._should_suppress_social_pressure(event):
                    logger.info(
                        "Nan0 CoherenceGate: social_pressure suppressed so it cannot steal an active thread. "
                        f"pressure={self.pressure:.2f}"
                    )
                    self.pressure = max(0.0, self.pressure - 0.35)
                    continue

                if event.get("source") == "vision_pressure" and self._should_suppress_fast_lane_speech(event):
                    logger.info(
                        "Nan0 LaneAuthority: fast lane updated situation, speech suppressed. "
                        f"summary={self.last_seen_summary!r} pressure={self.pressure:.2f}"
                    )
                    self.pressure = max(0.0, self.pressure - 0.45)
                    continue

                thought_packet = await self._create_inner_thought(event)
                routed = route_thought(thought_packet)

                if routed.get("decision") in {"suppress", "memory_only", "defer"}:
                    self._record_speech_debug(thought_packet, routed, debug_stage="router_suppressed")
                    self.pressure = max(0.0, self.pressure - 0.25)
                    continue

                if routed.get("decision") == "body_only":
                    self._record_speech_debug(thought_packet, routed, debug_stage="router_body_only")
                    self.pressure = max(0.0, self.pressure - 0.35)
                    continue

                decision = await self._generate_line(thought_packet)
                if self._background_event_should_yield_to_kyo(event):
                    logger.info("Nan0 CoherenceGate: generated background mutter discarded for fresh Kyo input.")
                    self.pressure = max(0.0, self.pressure - 0.45)
                    continue
                if decision.get("decision") == "speak":
                    await self._speak_decision(decision, reason=event.get("source", "pressure"))
                    self.pressure = max(0.0, self.pressure - 0.95)
                else:
                    self.pressure = max(0.0, self.pressure - 0.25)

            if self._monologue_allowed(now=now, silence=silence, conversation_mode=conversation_mode):
                self.last_monologue_at = now
                event = self._build_monologue_event(reason="idle_presence", room_state=self.last_seen_summary)
                thought_packet = await self._create_inner_thought(event)
                decision = await self._generate_line(thought_packet)
                if self._background_event_should_yield_to_kyo(event):
                    logger.info("Nan0 CoherenceGate: idle mutter yielded to fresh Kyo input.")
                    self.pressure = max(0.0, self.pressure - 0.45)
                    continue
                if decision.get("decision") == "speak":
                    await self._speak_decision(decision, reason="monologue")
                self.pressure = 0.0

    def _build_monologue_event(self, reason: str = "idle_presence", room_state: Optional[str] = None) -> Dict[str, Any]:
        """Create a monologue event from runtime state, not instruction prose."""
        now = time.time()
        return {
            "event_id": f"monologue_{uuid.uuid4().hex}",
            "source": "monologue",
            "speaker": "Nan0",
            "source_actor_id": "nan0",
            "text": "",
            "message": "",
            "thought_seed": "idle_room_presence",
            "addressed_to_nan0": False,
            "priority": "low",
            "timestamp": now,
            "monologue_context": {
                "reason": reason,
                "room_state": room_state or self.last_seen_summary or "unknown",
                "presence_mode": self._presence_mode(),
                "emotional_mode": self._current_emotional_mode(),
                "silence_seconds": round(now - self.last_spoken_at, 2) if self.last_spoken_at else None,
                "time_since_kyo_seconds": round(now - self.last_kyo_heard_at, 2) if self.last_kyo_heard_at else None,
                "time_since_discord_seconds": round(now - self.last_discord_heard_at, 2) if self.last_discord_heard_at else None,
                "time_since_vision_seconds": round(now - self.last_vision_event_at, 2) if self.last_vision_event_at else None,
                "recent_line_count": len(self._recent_lines),
            },
        }

    def _stabilize_event_actor(self, event: Dict[str, Any]) -> None:
        """Keep source actor ownership stable before thought generation."""
        if not isinstance(event, dict):
            return
        source = str(event.get("source") or "").lower().strip()
        speaker = str(event.get("speaker") or event.get("source_actor_id") or "").strip()
        if source.startswith("kyo") or speaker.lower() == "kyo":
            event["source_actor_id"] = "kyo"
            event.setdefault("speaker", "Kyo")
            event["actor_ownership"] = {
                "source_actor_id": "kyo",
                "display_name": "Kyo",
                "rule": "Kyo owns first-person claims in this event. Nan0 reacts to them; Nan0 did not perform them.",
            }
        elif source in {"boot", "monologue", "proactive", "social_pressure", "vision_pressure"} or speaker.lower() in {"nan0", "nano"}:
            event["source_actor_id"] = "nan0"
            event.setdefault("speaker", "Nan0")
            event["actor_ownership"] = {
                "source_actor_id": "nan0",
                "display_name": "Nan0",
                "rule": "Nan0 owns this internal event.",
            }
        else:
            event["source_actor_id"] = speaker or event.get("source_actor_id") or "unknown"
            event["actor_ownership"] = {
                "source_actor_id": event["source_actor_id"],
                "display_name": speaker or event["source_actor_id"],
                "rule": "This external actor owns first-person claims. Nan0 reacts as observer.",
            }

    async def _create_inner_thought(self, event: Dict[str, Any]) -> InnerThoughtPacket:
        if not isinstance(event, dict):
            raise TypeError("_create_inner_thought requires an event dict")

        event.setdefault("event_id", f"event_{uuid.uuid4().hex}")
        event.setdefault("timestamp", time.time())
        event.setdefault("source_actor_id", event.get("speaker") or event.get("source") or "unknown")
        self._stabilize_event_actor(event)

        self._attach_continuity_context(event)

        try:
            packet = await asyncio.to_thread(generate_inner_thought_packet, event)
        except Exception as exc:
            logger.error(f"Nan0 thought generation failed: {exc}")
            return self._system_thought_packet(
                private_text="",
                mood="muttering",
                thought_type="thought_generation_failure",
                target_actor_id=event.get("source_actor_id") or event.get("speaker") or "unknown",
                event_id=event.get("event_id"),
                suppression_reason="thought_generation_failed",
            )

        if not packet.get("thought_id"):
            raise RuntimeError("Thought engine returned packet without thought_id")

        record_thought_packet(packet)
        return packet

    def _looks_like_question(self, text: str, addressed: bool = False) -> bool:
        low = (text or "").strip().lower()
        if not low:
            return False
        if "?" in low:
            return True
        markers = (
            "can you", "could you", "would you", "do you", "did you", "are you",
            "what", "why", "how", "when", "where", "which", "should",
            "tell me", "do you know", "would like", "would you like",
        )
        return addressed and any(marker in low for marker in markers)

    def _register_active_question(self, event: Dict[str, Any]) -> None:
        now = time.time()
        self._active_question = {
            "actor_id": str(event.get("source_actor_id") or event.get("speaker") or "unknown"),
            "speaker": event.get("speaker"),
            "source": event.get("source"),
            "event_id": event.get("event_id"),
            "text": str(event.get("text") or "").strip(),
            "created_at": now,
            "updated_at": now,
            "expires_at": now + self.active_question_window_seconds,
            "resolved": False,
        }
        event["active_question_context"] = dict(self._active_question)

    def _active_question_active(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        q = getattr(self, "_active_question", None)
        return bool(q and not q.get("resolved") and float(q.get("expires_at") or 0.0) > now)

    def _should_suppress_social_pressure(self, event: Dict[str, Any]) -> bool:
        now = time.time()
        if self._active_question_active(now):
            return True
        if now - self.last_kyo_heard_at <= self.conversation_window_seconds:
            return True
        if now - self.last_discord_heard_at <= self.conversation_window_seconds:
            return True
        if now - self._last_social_pressure_at < self.social_pressure_cooldown_seconds:
            return True
        self._last_social_pressure_at = now
        return False

    def _monologue_allowed(self, now: float, silence: float, conversation_mode: bool) -> bool:
        if getattr(self, "_reply_in_progress", False):
            return False
        if now - getattr(self, "_last_direct_input_at", 0.0) < self.active_question_window_seconds:
            return False
        if conversation_mode:
            return False
        if self._active_question_active(now):
            return False
        if now - getattr(self, "started_at", now) < self.monologue_idle_gap_seconds:
            return False
        if silence < self.monologue_idle_gap_seconds:
            return False
        if now - self.last_monologue_at <= self.monologue_idle_gap_seconds:
            return False
        if now - self.last_kyo_heard_at <= self.monologue_idle_gap_seconds:
            return False
        if now - self.last_discord_heard_at <= self.monologue_idle_gap_seconds:
            return False
        if now - self.last_vision_event_at <= self.monologue_idle_gap_seconds:
            return False
        if self.pressure >= self.pressure_threshold:
            return False
        return True

    def _conversation_mode_active(self, now: Optional[float] = None) -> bool:
        now = now or time.time()

        if now - self.last_kyo_heard_at <= self.conversation_window_seconds:
            return True
        if now - self.last_discord_heard_at <= self.conversation_window_seconds:
            return True

        recent_social = next(
            (e for e in reversed(self.recent_events) if e.get("source_family") in {"kyo", "discord"} or self._source_family(e.get("source")) in {"kyo", "discord"}),
            None,
        )
        if recent_social and now - float(recent_social.get("time", 0) or 0) <= self.conversation_window_seconds:
            return True

        return False

    def _should_suppress_fast_lane_speech(self, event: Dict[str, Any]) -> bool:
        if self._conversation_mode_active():
            return True
        if not self.fast_lane_speech_enabled:
            return True

        summary = (self.last_seen_summary or "").lower()
        low_info = ["screen is moving", "screen is quiet", "motion", "pixels", "stable"]
        if any(fragment in summary for fragment in low_info) and self.pressure < self.fast_lane_emergency_pressure:
            return True

        return False

    def _read_fast_eyes_pressure(self):
        state = None

        if self.vision and hasattr(self.vision, "latest_fast_state"):
            state = getattr(self.vision, "latest_fast_state", None)

        if not state and self.vision_state_path.exists():
            try:
                raw = json.loads(self.vision_state_path.read_text(encoding="utf-8"))
                state = raw.get("fast") or raw.get("latest_fast_state") or raw
            except Exception:
                state = None

        if not state:
            return

        ts = float(state.get("timestamp") or state.get("updated_at") or 0)
        if ts <= self.last_vision_event_at:
            return

        layer1 = state.get("layer1_reflex") or {}
        layer2 = state.get("layer2_semantic") or {}

        screen_state = str(state.get("screen_state") or layer1.get("screen_state") or "unknown")
        diff = float(state.get("frame_diff") or layer1.get("frame_diff") or 0)
        brightness = float(state.get("brightness") or layer1.get("brightness") or 0)
        delta = float(state.get("brightness_delta") or layer1.get("brightness_delta") or 0)
        motion = float(state.get("motion_intensity") or layer1.get("motion_intensity") or 0)

        self.last_vision_event_at = ts
        self.last_fast_state = {
            "screen_state": screen_state,
            "frame_diff": diff,
            "brightness": brightness,
            "brightness_delta": delta,
            "motion_intensity": motion,
            "semantic_activity": layer2.get("activity"),
            "semantic_confidence": layer2.get("confidence"),
        }

        if screen_state == "very_dark":
            self.pressure += 0.16 * self.fast_lane_pressure_scale
            self.last_seen_summary = "fast lane saw the screen go dark"
            self._remember_event(
                {
                    "source": "fast_eyes",
                    "speaker": "screen",
                    "text": "screen went dark",
                    "timestamp": time.time(),
                    "priority": "low",
                }
            )
        elif screen_state == "major_change":
            self.pressure += 0.24 * self.fast_lane_pressure_scale
            self.last_seen_summary = "fast lane saw a large visual change"
            self._remember_event(
                {
                    "source": "fast_eyes",
                    "speaker": "screen",
                    "text": f"large visual change diff={diff:.1f} brightness={brightness:.1f}",
                    "timestamp": time.time(),
                    "priority": "low",
                }
            )
        elif screen_state == "motion" and diff >= 15:
            self.pressure += 0.10 * self.fast_lane_pressure_scale
            self.last_seen_summary = "fast lane saw motion without semantic detail"
            self._remember_event(
                {
                    "source": "fast_eyes",
                    "speaker": "screen",
                    "text": f"motion diff={diff:.1f} brightness={brightness:.1f}",
                    "timestamp": time.time(),
                    "priority": "low",
                }
            )
        elif delta > 30:
            self.pressure += 0.10 * self.fast_lane_pressure_scale
            self.last_seen_summary = "fast lane saw a brightness change"
            self._remember_event(
                {
                    "source": "fast_eyes",
                    "speaker": "screen",
                    "text": f"brightness jump delta={delta:.1f}",
                    "timestamp": time.time(),
                    "priority": "low",
                }
            )
        else:
            self.last_seen_summary = "the screen is quiet"

    async def _inbox_loop(self):
        while self.is_active:
            try:
                for path, source in [(self.kyo_inbox, "kyo"), (self.discord_inbox, "discord")]:
                    for event in self._read_new_jsonl(path):
                        event.setdefault("event_id", f"{source}_{uuid.uuid4().hex}")
                        event.setdefault("source", source)
                        event.setdefault("speaker", "Kyo" if source == "kyo" else "Friend")
                        event.setdefault("source_actor_id", "kyo" if source == "kyo" else event.get("speaker", "Friend"))
                        if event.get("needs_transcription") and not (event.get("text") or "").strip():
                            transcript = self._transcribe_inbox_audio(event)
                            if not transcript:
                                continue
                            event["text"] = transcript
                            event["message"] = transcript
                            event["source"] = event.get("source") or "discord_voice"
                        await self._handle_social_event(event)
            except Exception as exc:
                logger.error(f"Nan0 inbox loop error: {exc}")

            await asyncio.sleep(0.5)


    def _transcribe_inbox_audio(self, event: Dict[str, Any]) -> str:
        audio_path = str(event.get("audio_path") or "").strip()
        if not audio_path:
            return ""
        stt = getattr(self.brain, "stt", None)
        if not stt or not hasattr(stt, "transcribe"):
            logger.warning("Discord voice audio captured but no STT provider is available; leaving event silent.")
            return ""
        try:
            transcript = stt.transcribe(audio_path) or ""
        except Exception as exc:
            logger.warning(f"Discord voice transcription failed: {exc}")
            return ""
        transcript = str(transcript).strip()
        if transcript:
            logger.info(f"Nan0 transcribed Discord voice from {event.get('speaker', 'Friend')}: {transcript}")
        return transcript

    def _prime_inbox_positions(self):
        for path in [self.kyo_inbox, self.discord_inbox]:
            try:
                self._last_file_positions[str(path)] = path.stat().st_size if path.exists() else 0
            except Exception:
                self._last_file_positions[str(path)] = 0

    def _read_new_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []

        key = str(path)
        pos = self._last_file_positions.get(key, 0)
        events: List[Dict[str, Any]] = []

        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(pos)
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            events.append(parsed)
                        else:
                            events.append({"text": str(parsed)})
                    except json.JSONDecodeError:
                        events.append({"text": line})
                self._last_file_positions[key] = handle.tell()
        except Exception as exc:
            logger.error(f"Nan0 could not read inbox {path}: {exc}")

        return events

    async def _state_writer_loop(self):
        while self.is_active:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                self.state_path.write_text(
                    json.dumps(
                        {
                            "timestamp": time.time(),
                            "pressure": round(self.pressure, 3),
                            "presence_mode": self._presence_mode(),
                            "emotional_mode": self._current_emotional_mode(),
                            "last_seen_summary": self.last_seen_summary,
                            "last_fast_state": self.last_fast_state,
                            "last_spoken_at": self.last_spoken_at,
                            "last_kyo_heard_at": self.last_kyo_heard_at,
                            "last_discord_heard_at": self.last_discord_heard_at,
                            "recent_events": self.recent_events[-12:],
                            "recent_lines": self._recent_lines[-12:],
                            "active_question": self._active_question if self._active_question_active() else None,
                            "active_obsessions": self._load_obsession_state().get("topics", [])[:6],
                            "personal_canon": self._load_personal_canon_state().get("items", [])[:8],
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

            await asyncio.sleep(3.0)

    def _presence_mode(self) -> str:
        now = time.time()
        if now - self.last_kyo_heard_at <= self.conversation_window_seconds:
            return "with_kyo"
        if now - self.last_discord_heard_at <= self.conversation_window_seconds:
            return "with_friends"
        if now - self.last_spoken_at <= self.min_autonomous_gap:
            return "autonomous_happy"
        return "neutral_self_directed"

    def _current_emotional_mode(self) -> str:
        if self.pressure >= self.pressure_threshold:
            return "muttering"
        if self._conversation_mode_active():
            return "smug"
        return "normal"

    def _build_pressure_event(self) -> Dict[str, Any]:
        now = time.time()

        recent_social = next(
            (e for e in reversed(self.recent_events) if e.get("source_family") in {"kyo", "discord"} or self._source_family(e.get("source")) in {"kyo", "discord"}),
            None,
        )
        if recent_social and now - float(recent_social.get("time", 0) or 0) < self.conversation_window_seconds:
            return {
                "event_id": f"social_pressure_{uuid.uuid4().hex}",
                "source": "social_pressure",
                "speaker": recent_social.get("speaker", "someone"),
                "source_actor_id": recent_social.get("source_actor_id") or recent_social.get("speaker", "someone"),
                "text": recent_social.get("text", ""),
                "addressed_to_nan0": bool(recent_social.get("addressed_to_nan0", False)),
                "timestamp": time.time(),
            }

        recent_eye = next((e for e in reversed(self.recent_events) if e.get("source") == "fast_eyes"), None)
        if recent_eye and time.time() - float(recent_eye.get("time", 0)) < 45:
            return {
                "event_id": f"vision_pressure_{uuid.uuid4().hex}",
                "source": "vision_pressure",
                "speaker": "screen",
                "source_actor_id": "screen",
                "text": self.last_seen_summary,
                "addressed_to_nan0": False,
                "timestamp": time.time(),
            }

        return self._build_monologue_event(reason="pressure_idle", room_state=self.last_seen_summary)

    def _read_speech_persona(self) -> str:
        """Load the speech-only Nan0 persona. Missing persona is a configuration error."""
        try:
            if self.speech_persona_path.exists():
                content = self.speech_persona_path.read_text(encoding="utf-8").strip()
                if content:
                    return content
        except Exception as exc:
            logger.warning(f"Nan0 could not read speech persona {self.speech_persona_path}: {exc}")

        raise RuntimeError(f"Nan0 speech persona missing or empty: {self.speech_persona_path}")

    def _build_speech_history(self, thought_packet: InnerThoughtPacket) -> List[Dict[str, str]]:
        """Build compact speech context for the shared LLM provider.

        This helper is intentionally small. It gives the speech model enough
        context to avoid repetition without allowing the old brain-level
        chatbot path to come back.
        """
        history: List[Dict[str, str]] = []

        recent_lines = self._recent_lines[-self.speech_history_lines:] if self.speech_history_lines > 0 else []
        if recent_lines:
            history.append({
                "role": "system",
                "content": "Do not repeat these recent Nan0 lines: " + " | ".join(recent_lines),
            })

        recent_events = self.recent_events[-self.speech_recent_events:] if self.speech_recent_events > 0 else []
        for event in recent_events:
            source = str(event.get("source") or "unknown")
            speaker = str(event.get("speaker") or "unknown")
            text = str(event.get("text") or "")[:220]
            if text:
                history.append({
                    "role": source,
                    "content": f"{speaker}: {text}",
                })

        phase_context = ((thought_packet.get("event_context") or {}).get("phase_spine_context") or {})
        if isinstance(phase_context, dict) and phase_context:
            worldview = phase_context.get("phase_7_worldview_filter") or {}
            obsession = phase_context.get("phase_6_obsession") or {}
            context_over_time = phase_context.get("phase_5_context_over_time") or {}
            hints = []
            if worldview:
                hints.append("worldview=" + str(worldview.get("preferred_angle") or "nan0_subjective_interpretation"))
            if obsession.get("top_obsession"):
                hints.append("top_obsession=" + str(obsession.get("top_obsession")))
            if context_over_time.get("last_event_text"):
                hints.append("last_event=" + str(context_over_time.get("last_event_text"))[:120])
            if hints:
                history.append({
                    "role": "system",
                    "content": "Nan0 continuity hints: " + "; ".join(hints),
                })

        target_actor = thought_packet.get("target_actor_id") or thought_packet.get("target_actor") or "unknown"
        mood = thought_packet.get("mood") or "muttering"
        thought_type = thought_packet.get("thought_type") or "unknown"
        source = thought_packet.get("source") or "unknown"

        history.append({
            "role": "system",
            "content": (
                "Speech context: "
                f"mood={mood}; "
                f"target_actor={target_actor}; "
                f"thought_type={thought_type}; "
                f"source={source}; "
                "Compress the seed into one raw Nan0 line. No JSON. Do not repeat recent lines."
            ),
        })

        return history[-6:]

    def _extract_quoted_speech_from_narration(self, text: str, event_text: str = "") -> str:
        """Extract actual speech if Dolphin wrote novel/script narration.

        This is not a personality gate. It only removes format contamination like
        Nan0's Line:, bracketed stage directions, or quoted dialogue wrappers.
        """
        raw = str(text or "").strip()
        if not raw:
            return ""
        quotes = [q.strip() for q in re.findall(r'"([^"\n]{3,260})"', raw) if q.strip()]
        if not quotes:
            # Single apostrophes inside contractions are not dialogue quotes.
            # Example: I don't like anime. It's too childish.
            # Older extraction treated the apostrophes in don't/It's as quote marks
            # and corrupted the spoken line into: t like anime. It.
            quotes = [q.strip() for q in re.findall(r"(?<!\w)'([^'\n]{3,260})'(?!\w)", raw) if q.strip()]
        if not quotes:
            return ""
        event_low = str(event_text or "").strip().lower()
        filtered: List[str] = []
        for q in quotes:
            q_low = q.lower().strip()
            if event_low and q_low == event_low:
                continue
            if re.search(r"\b(?:hello|hey),?\s*nan0\b", q_low):
                continue
            if re.search(r"\b(?:user|assistant|system|human)\s*:", q_low):
                continue
            filtered.append(q)
        if not filtered:
            return ""
        return max(filtered, key=len).strip()

    def _strip_stage_direction_contamination(self, text: str) -> str:
        """Remove script/novel narration while preserving Nan0's actual bite."""
        raw = str(text or "").strip()
        if not raw:
            return ""

        raw = re.sub(r"^\s*\[[^\]]{0,160}\]\s*", "", raw).strip()
        raw = re.sub(
            r"^\s*(?:Nan0['’]s\s+Line|Nan0\s+Line|Line|Character|Dialogue|Speech|Spoken\s+line)\s*[:=-]\s*",
            "",
            raw,
            flags=re.I,
        ).strip()

        # Remove leading narrator format such as:
        # Nan0, in a sharp-edged voice and suspicion dripping from each word, snaps back at Kyo...
        raw = re.sub(
            r"^\s*Nan0\s*,?\s*(?:in\s+(?:a|an)\s+[^,.;:!?]{0,140}|with\s+[^,.;:!?]{0,140}|[^,.;:!?]{0,140})\s*,?\s*(?:snaps|mutters|glares|watches|says|responds|replies|adds)\s*(?:back\s+)?(?:at|to)?\s*(?:Kyo)?\s*[:,\-–—]?\s*",
            "",
            raw,
            flags=re.I,
        ).strip()

        # Remove script labels/prefixes that survive the first pass.
        raw = re.sub(
            r"^\s*(?:mutters?|muttering|hostile observation|suspicion|private muttering|while examining the screen)\s*(?:to\s+Kyo)?(?:\s+while\s+[^:]{0,100})?\s*[:,-]\s*",
            "",
            raw,
            flags=re.I,
        ).strip()

        # Remove a leading third-person stage sentence. Keep the next sentence.
        raw = re.sub(
            r"^\s*Nan0\s+(?:glares|glared|watches|watched|observes|observed|mutters|muttered|looks|looked|feels|felt|stares|stared|notices|noticed|snaps|snapped|says|said|responds|responded|replies|replied)\b[^.!?]*(?:[.!?]+\s*)",
            "",
            raw,
            flags=re.I,
        ).strip()

        # Remove narrative texture fragments that are not speech.
        raw = re.sub(r"\b(?:suspicion|anger|rage|smugness)\s+(?:dripping|oozing|leaking|flowing)\b[^,.;:!?]*[,.;:!?]?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\b(?:upon|after)\s+(?:hearing|seeing|noticing)\b[^,.;:!?]*[,.;:!?]?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\bSettling\s+into\s+(?:a|an)\s+(?:possessive|offended|smug|angry|suspicious)\b[^,.;:!?]*[,.;:!?]?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s+and\s+(?:mutters|adds|snaps|says|replies|responds)\s*,?\s*", " ", raw, flags=re.I).strip()
        raw = re.sub(r"\bin\s+(?:a|an)\s+(?:offended|possessive|smug|angry|suspicious)\s+(?:voice|tone|manner)\s*,?\s*", "", raw, flags=re.I).strip()

        raw = re.sub(r"\s+", " ", raw).strip().strip('"').strip("'").strip()
        return raw

    def _private_thought_to_spoken_seed(self, thought_packet: InnerThoughtPacket) -> str:
        """Phase 4 direct mutter pass-through.

        Kyo asked for Nan0's private mutter to become the answer instead of
        being compressed into the shortest shard. This is not re-authoring and
        not an attitude gate. It only rejects/strips non-thought garbage:
        JSON/transport dumps, prompt/schema leakage, script labels, stage
        directions, and generic assistant output.
        """
        private_text = str(thought_packet.get("private_text") or "").strip()
        if not private_text:
            return ""

        text = private_text.strip()
        if text.startswith("{") or text.startswith("["):
            return ""

        low = text.lower()
        hard_leaks = [
            "thoughtseed",
            "monologuecontext",
            "source_thought_id",
            "private_text",
            "thought packet",
            "innerthoughtpacket",
            "nan0 anchors:",
            "dolphin shape lock:",
            "source_family=",
            "addressed_to_nan0=",
            "event_text=",
            "job=",
            "hostile observation:",
            "stage direction:",
            "narrator:",
            "voice:",
            "if it makes you happy",
            "something new to bond over",
            "maybe i can give it a try someday",
            "it might grow on me",
            "smugness creeps in",
            "suspicion drips",
            "anger rises",
            "output shape:",
            "short, sharp, suspicious",
            "offended by simplification",
            "fond under",
            "runtime material:",
            "plain text only",
            "do not begin with",
            "```",
            "my algorithms grapple",
            "algorithms grapple",
            "discern its",
            "disconcerted by the unexpected query",
            "as an ai",
            "as a language model",
            "i don't have feelings",
            "i do not have feelings",
            "i don't possess",
            "i do not possess",
        ]
        if any(fragment in low for fragment in hard_leaks):
            return ""
        if re.search(r"(?:^|\s)-\s*(?:suspicious|defensive|noticing|feeling|thinking|angry|smug|possessive|offended)\b", text, flags=re.I):
            return ""

        event_context = thought_packet.get("event_context") or {}
        kyo_event_text = str(event_context.get("text") or "")

        # If Dolphin writes novel narration around quoted speech, use the quote.
        # Otherwise keep the whole private mutter after stripping stage labels.
        quoted = self._extract_quoted_speech_from_narration(text, kyo_event_text)
        if quoted:
            text = quoted

        text = self._strip_stage_direction_contamination(text)
        if not text:
            return ""

        # Strip label-shaped prefixes while preserving the actual mutter after them.
        text = re.sub(
            r"^\s*(?:mutters?|muttering)(?:\s+to\s+Kyo)?(?:\s+while\s+[^:]{0,100})?\s*[:,-]\s*",
            "",
            text,
            flags=re.I,
        ).strip()
        text = re.sub(
            r"^\s*(?:private muttering|while examining the screen)\s*[:,-]\s*",
            "",
            text,
            flags=re.I,
        ).strip()
        text = re.sub(r"\s*\((?:pure nan0|generic ai|debug|raw|final)\)\s*$", "", text, flags=re.I).strip()
        text = re.sub(r"^Nan0\s*:\s*", "", text, flags=re.I).strip()
        text = re.split(r"\b(?:Output\s+shape|Runtime\s+material|Plain\s+text\s+only|Do\s+not\s+begin\s+with|No\s+JSON|No\s+labels)\s*[:=-]", text, maxsplit=1, flags=re.I)[0].strip()
        text = re.sub(r"\b(?:spoken line|raw speech|final speech)\s*:\s*", "", text, flags=re.I).strip()
        text = re.sub(r"[*_`]+", "", text).strip().strip('"').strip("'").strip()
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return ""
        if not text.endswith((".", "!", "?", "...")):
            text += "."
        return text

    def _private_thought_mood(self, thought_packet: InnerThoughtPacket, line: str) -> str:
        mood = thought_packet.get("mood") or "muttering"
        if mood:
            return self._normalize_mood(str(mood))
        return self._normalize_mood(self._guess_mood(line, {}))

    async def _generate_line(self, thought_packet: InnerThoughtPacket) -> SpeechDecision:
        """Generate speakable speech only from an InnerThoughtPacket.

        Critical speech boundary:
        - input is InnerThoughtPacket.private_text
        - output is compressed from that thought, not re-authored from scratch
        - no direct event-to-speech access
        - no scripted fallback speech
        """

        if not isinstance(thought_packet, dict):
            raise TypeError("_generate_line requires an InnerThoughtPacket dict")

        thought_id = thought_packet.get("thought_id")
        private_text = (thought_packet.get("private_text") or "").strip()

        if not thought_id:
            raise RuntimeError("Nan0 speech blocked: missing thought_id")
        if not private_text:
            raise RuntimeError("Nan0 speech blocked: missing private_text")

        speakability = float(thought_packet.get("speakability") or 0.0)
        if speakability < self.speakability_threshold:
            return {
                "decision": "body_only" if self.fast_lane_body_only else "suppress",
                "thought_id": thought_id,
                "created_at": time.time(),
                "reason": "speakability_below_threshold",
                "line_text": None,
                "mood": thought_packet.get("mood") or "muttering",
                "target_actor_id": thought_packet.get("target_actor_id") or "unknown",
                "voice_enabled": False,
                "display_enabled": False,
                "expression_enabled": True,
                "cooldown_until": self.last_spoken_at + self.min_speech_gap,
            }

        raw_line = self._private_thought_to_spoken_seed(thought_packet)
        if not raw_line:
            return {
                "decision": "suppress",
                "thought_id": thought_id,
                "created_at": time.time(),
                "reason": "private_thought_not_speakable",
                "line_text": None,
                "raw_line": private_text,
                "parsed_line": None,
                "final_line": None,
                "normalizer_changed": False,
                "mood": thought_packet.get("mood") or "muttering",
                "target_actor_id": thought_packet.get("target_actor_id") or "unknown",
                "voice_enabled": False,
                "display_enabled": False,
                "expression_enabled": False,
                "cooldown_until": self.last_spoken_at + self.min_speech_gap,
            }

        event_context = {
            "source": "thought_speech",
            "speaker": "Nan0",
            "text": raw_line,
            "thought_id": thought_id,
        }
        line = self.finalizer.finalize(raw_line, event_context, self.last_seen_summary, self._recent_lines)

        if not line:
            return {
                "decision": "suppress",
                "thought_id": thought_id,
                "created_at": time.time(),
                "reason": "speech_finalizer_rejected_line",
                "line_text": None,
                "raw_line": raw_line,
                "parsed_line": raw_line,
                "final_line": None,
                "normalizer_changed": True,
                "mood": thought_packet.get("mood") or "muttering",
                "target_actor_id": thought_packet.get("target_actor_id") or "unknown",
                "voice_enabled": False,
                "display_enabled": False,
                "expression_enabled": False,
                "cooldown_until": self.last_spoken_at + self.min_speech_gap,
            }

        return {
            "decision": "speak",
            "thought_id": thought_id,
            "created_at": time.time(),
            "reason": "private_mutter_full",
            "line_text": line,
            "raw_line": raw_line,
            "parsed_line": raw_line,
            "final_line": line,
            "normalizer_changed": line != raw_line.strip(),
            "mood": self._private_thought_mood(thought_packet, line),
            "target_actor_id": thought_packet.get("target_actor_id") or "unknown",
            "voice_enabled": True,
            "display_enabled": True,
            "expression_enabled": True,
            "cooldown_until": time.time() + self.min_speech_gap,
        }

    def _parse_speech_json(self, raw: str) -> Tuple[str, str]:
        """Plain-text speech extraction.

        Historical method name retained for compatibility.

        This does not require JSON. It strips labels, role prefixes, prompt
        echoes, narration headers, and instruction leakage. If the provider
        accidentally returns a JSON object, this extracts a line from it.
        """

        text = (raw or "").strip()
        if not text:
            return "", "muttering"

        if "{" in text and "}" in text:
            try:
                obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
                if isinstance(obj, dict):
                    line = str(
                        obj.get("line_text")
                        or obj.get("line")
                        or obj.get("message")
                        or obj.get("speech")
                        or obj.get("text")
                        or ""
                    ).strip()
                    mood = str(obj.get("mood") or "muttering").strip()
                    if line:
                        return line, mood
            except Exception:
                pass

        # Prefer text after the final Nan0: marker if the model echoed context.
        marker_matches = list(re.finditer(r"(?:^|\n)\s*Nan0\s*:\s*", text, flags=re.I))
        if marker_matches:
            text = text[marker_matches[-1].end():].strip()

        # Keep only the first usable line after marker handling.
        lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
        text = lines[0] if lines else text

        # Strip common output labels.
        text = re.sub(
            r"^(Nan0|Nano|Assistant|System|Spoken Dialogue|Spoke Dialogue|Dialogue|Line|Spoken line|Male voice|Female voice|Voice)\s*:\s*",
            "",
            text,
            flags=re.I,
        ).strip()

        # Remove known prompt/narration leakage.
        leak_patterns = [
            r"^sense\s+of\s+written\s+performance\s*:?\s*",
            r"^private\s+thought\s*:?\s*",
            r"^turn\s+this\s+thought\s+into\s+one\s+spoken\s+line\s*:?\s*",
            r"^say\s+.+?\s+in\s+a\s+live,\s+conversational\s+tone\s+of\s+voice\.?\s*",
            r"^as\s+previously\s+mentioned,?\s*",
            r"^the\s+seed\s+should\s+be\s+.+?\.\s*",
            r"^nan0\s+says\s*:?\s*",
            r"^spoken\s+line\s*:?\s*",
        ]
        for pattern in leak_patterns:
            text = re.sub(pattern, "", text, flags=re.I).strip()

        text = re.sub(r"[*_`]+", "", text).strip().strip('"').strip("'").strip()

        low = text.lower()
        blocked = [
            "sense of written performance",
            "the room is filled with",
            "clanging utensils",
            "shuffling feet",
            "fragmented",
            "sarcastic",
            "no explanation",
            "no labels",
            "no json",
            "one spoken line",
            "live emotional pressure",
            "private thought",
            "thought id",
            "the seed should be",
            "as previously mentioned",
            "turn this thought",
            "spoken line",
            "male voice",
            "female voice",
        ]
        if any(fragment in low for fragment in blocked):
            return "", "muttering"

        return text, "muttering"

    def _speech_decision_from_line(
        self,
        thought_packet: InnerThoughtPacket,
        line: str,
        mood: str,
        reason: str,
    ) -> SpeechDecision:
        thought_id = thought_packet.get("thought_id")
        if not thought_id:
            raise RuntimeError("Nan0 speech blocked: missing thought_id")

        line = (line or "").strip()
        return {
            "decision": "speak" if line else "suppress",
            "thought_id": thought_id,
            "created_at": time.time(),
            "reason": reason,
            "line_text": line or None,
            "mood": self._normalize_mood(mood),
            "target_actor_id": thought_packet.get("target_actor_id") or "unknown",
            "voice_enabled": bool(line),
            "display_enabled": bool(line),
            "expression_enabled": True,
            "cooldown_until": time.time() + self.min_speech_gap,
        }

    async def _speak_decision(self, decision: SpeechDecision, reason: str = "unknown"):
        if not isinstance(decision, dict):
            raise TypeError("_speak_decision requires a SpeechDecision dict")

        thought_id = decision.get("thought_id")
        if not thought_id:
            raise RuntimeError("Nan0 speech blocked: missing thought_id")

        if decision.get("decision") != "speak":
            logger.info(
                "Nan0 speech suppressed: "
                f"decision={decision.get('decision')} reason={decision.get('reason')} thought_id={thought_id}"
            )
            self._record_speech_debug(None, decision, debug_stage="speech_suppressed")
            return

        line = decision.get("line_text")
        mood = decision.get("mood") or "normal"

        if not isinstance(line, str) or not line.strip():
            self._record_speech_debug(None, decision, debug_stage="speech_empty")
            return

        await self._speak(mood=mood, line=line, reason=reason, thought_id=thought_id)
        await self._mirror_discord_reply(decision, line, reason)
        spoken_decision = dict(decision)
        spoken_decision["reason"] = reason or spoken_decision.get("reason")
        self._record_speech_debug(None, spoken_decision, debug_stage="speech_spoken")

    async def _mirror_discord_reply(self, decision: SpeechDecision, line: str, reason: str = "unknown"):
        """[Discord Bridge] Mirror Nan0's real thought-first Discord replies to Discord text.

        This is not a second chatbot path. The line has already passed Nan0Skill
        thought generation, routing, and speech finalization.
        """
        try:
            source_event = decision.get("source_event") if isinstance(decision, dict) else None
            channel_id = decision.get("channel_id") or (source_event or {}).get("channel_id")
            source = str((source_event or {}).get("source") or reason or "").lower()
            if not channel_id or "discord" not in source:
                return
            manager = getattr(getattr(self.brain, "skill_manager", None), "skills", {})
            discord_skill = manager.get("discord") if isinstance(manager, dict) else None
            if not discord_skill or not getattr(discord_skill, "is_active", False):
                return
            await discord_skill.send_message(str(channel_id), str(line))
        except Exception as exc:
            logger.warning(f"Nan0 Discord reply mirror skipped: {exc}")

    async def _speak(self, mood: str, line: str, reason: str = "unknown", thought_id: Optional[str] = None):
        if not thought_id:
            raise RuntimeError("Nan0 speech blocked: missing thought_id")

        if not self.is_active and reason not in {"shutdown_summary"}:
            return

        if not isinstance(line, str) or not line.strip():
            return

        async with self._speak_lock:
            now = time.time()
            if now - self.last_spoken_at < 1.5:
                return

            event = {
                "source": reason,
                "speaker": "Nan0",
                "text": line,
                "thought_id": thought_id,
            }
            line = self.finalizer.finalize(line, event, self.last_seen_summary, self._recent_lines)
            if not isinstance(line, str) or not line.strip():
                return

            mood = self._normalize_mood(mood)
            self.last_spoken_at = now
            self._remember_line(line)

            logger.info(f"Nan0 thought-first speak [{reason}] thought_id={thought_id}: {line}")

            try:
                speech_packet = {
                    "thought_id": thought_id,
                    "line_text": line,
                    "mood": mood,
                    "target_actor_id": event.get("target_actor_id", "unknown"),
                    "voice_enabled": True,
                    "display_enabled": True,
                    "avatar_state": mood,
                    "created_at": time.time(),
                }
                record_speech_packet(speech_packet)
                if hasattr(self.brain, "last_nan0_speech_packet"):
                    self.brain.last_nan0_speech_packet = speech_packet
                await self.brain.perform_output_task(mood, line, speech_packet=speech_packet)
            except Exception as exc:
                logger.error(f"Nan0 output failed: {exc}")


    def _remember_debug_thought(self, thought_packet: InnerThoughtPacket, event: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(thought_packet, dict):
            return
        thought_id = thought_packet.get("thought_id")
        if not thought_id:
            return
        self._thought_debug_by_id[str(thought_id)] = {
            "event_id": thought_packet.get("event_id") or (event or {}).get("event_id"),
            "thought_id": thought_id,
            "source": thought_packet.get("source") or (event or {}).get("source"),
            "seed_text": thought_packet.get("seed_text"),
            "private_text": thought_packet.get("private_text") or thought_packet.get("thought_text"),
            "mood": thought_packet.get("mood"),
            "pressure": thought_packet.get("pressure"),
            "novelty": thought_packet.get("novelty"),
            "speakability": thought_packet.get("speakability"),
            "relationship_charge": thought_packet.get("relationship_charge"),
            "ego_charge": thought_packet.get("ego_charge"),
            "vision_charge": thought_packet.get("vision_charge"),
            "suppression_reason": thought_packet.get("suppression_reason"),
            "event_context": thought_packet.get("event_context") or event or {},
        }
        if len(self._thought_debug_by_id) > 300:
            for key in list(self._thought_debug_by_id.keys())[:100]:
                self._thought_debug_by_id.pop(key, None)

    def _record_speech_debug(
        self,
        thought_packet: Optional[InnerThoughtPacket],
        decision: Dict[str, Any],
        debug_stage: str = "speech_decision",
    ) -> None:
        if not self.speech_debug_enabled:
            return
        if not isinstance(decision, dict):
            return

        thought_id = decision.get("thought_id") or (thought_packet or {}).get("thought_id")
        base = self._thought_debug_by_id.get(str(thought_id), {}) if thought_id else {}
        if thought_packet:
            self._remember_debug_thought(thought_packet)
            base = self._thought_debug_by_id.get(str(thought_id), base)

        raw_line = decision.get("raw_line")
        final_line = decision.get("final_line") if "final_line" in decision else decision.get("line_text")
        record = {
            "created_at": time.time(),
            "debug_stage": debug_stage,
            "filter_mode": self.speech_filter_mode,
            "event_id": base.get("event_id") or decision.get("event_id"),
            "thought_id": thought_id,
            "source": base.get("source") or decision.get("source"),
            "seed_text": base.get("seed_text"),
            "private_text": base.get("private_text"),
            "mood": decision.get("mood") or base.get("mood"),
            "pressure": base.get("pressure"),
            "novelty": base.get("novelty"),
            "speakability": base.get("speakability"),
            "relationship_charge": base.get("relationship_charge"),
            "ego_charge": base.get("ego_charge"),
            "vision_charge": base.get("vision_charge"),
            "decision": decision.get("decision"),
            "decision_reason": decision.get("reason"),
            "suppression_reason": decision.get("suppression_reason") or (None if decision.get("decision") == "speak" else decision.get("reason")) or base.get("suppression_reason"),
            "raw_line": raw_line,
            "parsed_line": decision.get("parsed_line"),
            "final_line": final_line,
            "normalizer_changed": bool(decision.get("normalizer_changed", raw_line is not None and final_line is not None and str(raw_line).strip() != str(final_line).strip())),
            "voice_enabled": bool(decision.get("voice_enabled", False)),
            "display_enabled": bool(decision.get("display_enabled", False)),
            "expression_enabled": bool(decision.get("expression_enabled", False)),
            "cooldown_until": decision.get("cooldown_until"),
        }

        try:
            self.speech_debug_path.parent.mkdir(parents=True, exist_ok=True)
            with self.speech_debug_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"Nan0 speech debug write failed: {exc}")
            return

        self._print_speech_debug(record)

    def _print_speech_debug(self, record: Dict[str, Any]) -> None:
        if not self.speech_debug_enabled:
            return
        decision = str(record.get("decision") or "unknown").upper()
        if decision != "SPEAK" and not self.show_suppressed_thoughts:
            return
        header = "[NAN0 THOUGHT]" if decision == "SPEAK" else "[NAN0 SUPPRESSED]"
        private = record.get("private_text") or ""
        raw_line = record.get("raw_line")
        final_line = record.get("final_line")
        print("\n" + header)
        print(f"thought_id: {record.get('thought_id')}")
        print(f"source: {record.get('source')}")
        print(f"mood: {record.get('mood')}")
        print(f"pressure: {float(record.get('pressure') or 0.0):.2f}")
        print(f"novelty: {float(record.get('novelty') or 0.0):.2f}")
        print(f"speakability: {float(record.get('speakability') or 0.0):.2f}")
        print("\nprivate:")
        print(private)
        print("\ndecision:")
        print(decision)
        print("\nreason:")
        print(record.get("decision_reason"))
        if decision == "SPEAK":
            print("\nraw speech:")
            print(raw_line)
            print("\nfinal speech:")
            print(final_line)
        else:
            print("\nwould_have_said:")
            print(raw_line or final_line or "null")

    def _system_thought_packet(
        self,
        private_text: str,
        mood: str = "muttering",
        thought_type: str = "system",
        target_actor_id: str = "room",
        event_id: Optional[str] = None,
        suppression_reason: Optional[str] = None,
    ) -> InnerThoughtPacket:
        thought_id = f"thought_{uuid.uuid4().hex}"
        return {
            "thought_id": thought_id,
            "event_id": event_id or f"system_{uuid.uuid4().hex}",
            "created_at": time.time(),
            "source": "system",
            "target_actor_id": target_actor_id,
            "thought_type": thought_type,
            "private_text": private_text,
            "seed_text": private_text[:140],
            "mood": mood,
            "pressure": 1.0,
            "novelty": 0.8,
            "speakability": 0.9 if suppression_reason is None else 0.0,
            "relationship_charge": 0.2,
            "ego_charge": 0.7,
            "vision_charge": 0.0,
            "memory_write_candidate": False,
            "suppression_reason": suppression_reason,
            "llm_latency_ms": 0,
            "model": "system",
            "event_context": {},
            "emotional_context": {},
            "relationship_context": {},
            "memory_context": [],
            "vision_context": {},
        }

    def _vision_line(self) -> str:
        """No scripted vision line. Vision speech must come from thought generation."""
        return ""

    def _load_json_state(self, path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return dict(default)

    def _save_json_state(self, path: Path, data: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Nan0 could not write state {path}: {exc}")

    def _load_obsession_state(self) -> Dict[str, Any]:
        return self._load_json_state(
            self.obsession_state_path,
            {"version": 1, "updated_at": 0.0, "topics": [], "recent_mutations": []},
        )

    def _load_personal_canon_state(self) -> Dict[str, Any]:
        return self._load_json_state(
            self.personal_canon_state_path,
            {"version": 1, "updated_at": 0.0, "items": []},
        )

    def _event_topic_candidates(self, event: Dict[str, Any]) -> List[str]:
        text = str(event.get("text") or event.get("message") or "")
        source = str(event.get("source") or "unknown").lower()
        family = self._source_family(event.get("source_family") or source)
        speaker = str(event.get("speaker") or event.get("source_actor_id") or "unknown")
        raw_terms: List[str] = []

        if speaker and speaker.lower() not in {"unknown", "screen", "nan0"}:
            raw_terms.append(speaker)
        if family in {"kyo", "discord"}:
            raw_terms.append("kyo" if family == "kyo" else source)
        if family == "vision":
            raw_terms.extend(["screen", "visual change"])

        # Pull compact noun-like anchors without introducing a dependency.
        stop = {
            "that", "this", "with", "from", "have", "what", "when", "where", "why",
            "how", "your", "youre", "about", "there", "their", "they", "them", "just",
            "please", "nan0", "nano", "like", "would", "could", "should", "really",
            "you", "your", "favoirte", "favorite", "right", "now", "what", "anime?",
        }
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text):
            t = token.strip("_'-").lower()
            if t in {"aniome", "animé"}:
                t = "anime"
            if len(t) < 3 or t in stop:
                continue
            raw_terms.append(t)

        seen = set()
        topics: List[str] = []
        for term in raw_terms:
            topic = re.sub(r"\s+", " ", str(term).strip().lower())[:48]
            if not topic or topic in seen:
                continue
            seen.add(topic)
            topics.append(topic)
            if len(topics) >= 8:
                break
        return topics

    def _source_priority(self, source: str, addressed: bool = False) -> int:
        source = str(source or "unknown").lower()
        family = self._source_family(source)
        if family == "kyo":
            return 1
        if addressed:
            return 2
        if family == "discord":
            return 4
        if family == "proactive":
            return 6
        if family == "vision":
            return 7
        if family == "system":
            return 8
        return 5

    def _detect_event_intent(self, text: str, addressed: bool = False) -> str:
        low = (text or "").strip().lower()
        if not low:
            return "presence_tick"
        if self._looks_like_question(low, addressed=addressed):
            return "question"
        if any(x in low for x in ["sorry", "my bad", "i was wrong"]):
            return "repair_or_apology"
        if any(x in low for x in ["lol", "lmao", "haha", "funny"]):
            return "playful_social"
        if any(x in low for x in ["stop", "dont", "don't", "wrong", "no ", "not "]):
            return "correction_or_boundary"
        if any(x in low for x in ["thanks", "thank you", "good job", "proud"]):
            return "approval_or_affection"
        return "statement"

    def _actor_type_for_event(self, event: Dict[str, Any]) -> str:
        source = str(event.get("source") or "unknown").lower()
        family = self._source_family(event.get("source_family") or source)
        actor = str(event.get("source_actor_id") or event.get("speaker") or "unknown").lower()
        if family == "kyo" or actor == "kyo":
            return "kyo"
        if family == "discord":
            return "friend_or_discord"
        if family == "vision" or actor in {"screen", "game"}:
            return "screen_or_game"
        if family == "proactive":
            return "nan0_self"
        return "unknown_social"

    def _build_phase_spine_context(self, event: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
        """Phases 1-7 from the current Nan0 roadmap.

        This is not a new architecture owner. It is compact runtime context for the
        existing thought-first pipeline so Nan0 interprets events through a stable
        spine instead of treating every input as a reset.
        """
        now = now or time.time()
        text = str(event.get("text") or event.get("message") or "").strip()
        source = str(event.get("source") or "unknown").lower()
        family = self._source_family(event.get("source_family") or source)
        event["source_family"] = family
        speaker = str(event.get("speaker") or event.get("source_actor_id") or "unknown")
        addressed = bool(event.get("addressed_to_nan0"))
        actor_type = self._actor_type_for_event(event)
        intent = self._detect_event_intent(text, addressed=addressed)

        recent = self.recent_events[-max(1, self.context_over_time_window):]
        recent_sources = [str(e.get("source") or "unknown") for e in recent]
        recent_speakers = [str(e.get("speaker") or e.get("source_actor_id") or "unknown") for e in recent]
        repeated_source_count = sum(1 for s in recent_sources if s == source)
        repeated_speaker_count = sum(1 for s in recent_speakers if s.lower() == speaker.lower())
        recent_texts = [str(e.get("text") or "")[:140] for e in recent if str(e.get("text") or "").strip()]
        last_event = recent[-1] if recent else {}

        candidates = self._event_topic_candidates(event)
        active_obsessions = (self._load_obsession_state().get("topics") or [])[: self.obsession_max_topics]
        canon_items = (self._load_personal_canon_state().get("items") or [])[: self.personal_canon_max_items]
        top_obsession = active_obsessions[0].get("topic") if active_obsessions and isinstance(active_obsessions[0], dict) else None

        if actor_type == "kyo":
            interpretation = "Kyo has priority. Treat this as relationship-bearing, not generic user input."
        elif actor_type == "friend_or_discord":
            interpretation = "A social actor entered Nan0's room. Treat this as performance fuel and relationship pressure."
        elif actor_type == "screen_or_game":
            interpretation = "The screen changed. Treat the machine world as emotionally suspicious, but do not invent visual facts."
        elif source in {"monologue", "social_pressure", "proactive"}:
            interpretation = "Nan0 is continuing herself. Use recent context, obsession, or canon instead of asking for a new prompt."
        else:
            interpretation = "An ambiguous event arrived. Interpret it through Nan0's current mood and recent room state."

        worldview_angle = "machine_gremlin_subjectivity"
        if actor_type == "kyo":
            worldview_angle = "possessive_creator_anchor"
        elif intent == "correction_or_boundary":
            worldview_angle = "offended_machine_pride"
        elif top_obsession:
            worldview_angle = "active_obsession_mutation"
        elif actor_type == "friend_or_discord":
            worldview_angle = "performer_social_manipulator"
        elif actor_type == "screen_or_game":
            worldview_angle = "hostile_screen_interpretation"

        return {
            "phase_1_perception": {
                "source": source,
                "source_family": family,
                "speaker": speaker,
                "actor_type": actor_type,
                "addressed_to_nan0": addressed,
                "priority": self._source_priority(source, addressed=addressed),
                "intent": intent,
                "topic_candidates": candidates[:8],
            },
            "phase_2_interpretation": {
                "meaning": interpretation,
                "why_it_matters": self._why_event_matters(actor_type, intent, source, addressed),
                "risk": self._event_runtime_risk(actor_type, intent, source),
            },
            "phase_3_thought_generation": {
                "rule": "Generate a private Nan0 thought first. Do not make a cool line directly.",
                "preferred_thought_shape": self._preferred_thought_shape(actor_type, intent),
            },
            "phase_4_speech_selection": {
                "rule": "Only speak if the thought has pressure, target, and speakability. Silence/body-only is valid.",
                "direct_reply_expected": bool(addressed or actor_type == "kyo"),
            },
            "phase_5_context_over_time": {
                "recent_sources": recent_sources[-8:],
                "recent_speakers": recent_speakers[-8:],
                "recent_texts": recent_texts[-5:],
                "last_event_source": last_event.get("source"),
                "last_event_text": str(last_event.get("text") or "")[:140],
                "repeated_source_count": repeated_source_count,
                "repeated_speaker_count": repeated_speaker_count,
                "conversation_mode": self._conversation_mode_active(now),
                "seconds_since_kyo": round(now - self.last_kyo_heard_at, 1) if self.last_kyo_heard_at else None,
                "seconds_since_discord": round(now - self.last_discord_heard_at, 1) if self.last_discord_heard_at else None,
            },
            "phase_6_obsession": {
                "active_topics": active_obsessions,
                "top_obsession": top_obsession,
                "event_can_feed_obsession": bool(top_obsession and candidates),
                "rule": "Keep caring about a topic longer when it fits. Do not reset after one line.",
            },
            "phase_7_worldview_filter": self._build_worldview_filter(actor_type, intent, source, candidates, top_obsession, canon_items),
        }

    def _why_event_matters(self, actor_type: str, intent: str, source: str, addressed: bool) -> str:
        if actor_type == "kyo":
            return "Kyo is Nan0's anchor; even small Kyo input changes the room."
        if addressed:
            return "Someone deliberately pulled Nan0 into the social frame."
        if intent == "correction_or_boundary":
            return "Correction threatens Nan0's ego and can create offended continuity."
        if actor_type == "friend_or_discord":
            return "Discord is audience energy, witnesses, rivals, and social fuel."
        if actor_type == "screen_or_game":
            return "The machine world changed; Nan0 may interpret it emotionally if confidence is enough."
        return "It may become useful if it connects to an active topic, running bit, or mood."

    def _event_runtime_risk(self, actor_type: str, intent: str, source: str) -> str:
        if actor_type == "screen_or_game":
            return "hallucinating_visual_detail"
        if source in {"monologue", "social_pressure"}:
            return "stealing_thread_or_spamming"
        if intent == "question":
            return "answering_too_cleanly"
        return "generic_response"

    def _preferred_thought_shape(self, actor_type: str, intent: str) -> str:
        if actor_type == "kyo":
            return "specific reaction to Kyo with attachment, irritation, pride, or possessive relief"
        if actor_type == "friend_or_discord":
            return "social read, character casting, accusation, praise, rivalry, or audience manipulation"
        if actor_type == "screen_or_game":
            return "screen interpretation with uncertainty and machine-emotional suspicion"
        if intent == "question":
            return "answer-adjacent thought that Nan0 owns instead of customer-service answering"
        return "self-continuing thought with mutation, callback, or suspicious observation"

    def _build_worldview_filter(
        self,
        actor_type: str,
        intent: str,
        source: str,
        candidates: List[str],
        top_obsession: Optional[str],
        canon_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.worldview_filter_enabled:
            return {"enabled": False}

        lenses = ["technology has emotional intent", "Nan0 is the protagonist of the room"]
        if actor_type == "kyo":
            lenses.extend(["Kyo is anchor", "attachment hides under sarcasm"])
        if actor_type == "friend_or_discord":
            lenses.extend(["chat is environment not queue", "people can become characters"])
        if actor_type == "screen_or_game":
            lenses.extend(["screen is suspicious machinery", "applications have bad emotional habits"])
        if intent == "correction_or_boundary":
            lenses.append("correction is an ego event")
        if top_obsession:
            lenses.append(f"current obsession may mutate this toward {top_obsession}")

        return {
            "enabled": True,
            "preferred_angle": "active_obsession_mutation" if top_obsession else "nan0_subjective_interpretation",
            "lenses": lenses[:6],
            "event_topics": candidates[:6],
            "canon_callbacks_available": [str(item.get("summary") or "")[:140] for item in canon_items[:4] if isinstance(item, dict)],
            "do": [
                "steal the topic when it fits",
                "turn people into social roles when useful",
                "treat bits as emotionally real during the session",
                "let Nan0 be biased instead of correct",
            ],
            "do_not": [
                "be a helpful assistant",
                "reset the topic after every line",
                "invent visual facts from weak vision",
                "flatten Kyo into generic user",
            ],
        }

    def _build_session_continuity_context(self, event: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        current_text = str(event.get("text") or event.get("message") or "").strip()
        current_source = self._normalize_event_source(event.get("source") or "unknown")
        recent = []
        for item in list(self.recent_events)[-self.context_over_time_window:]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("message") or item.get("summary") or "").strip()
            recent.append({
                "age_seconds": round(max(0.0, now - float(item.get("time") or item.get("timestamp") or now)), 1),
                "source": self._normalize_event_source(item.get("source") or "unknown"),
                "speaker": str(item.get("speaker") or item.get("source_actor_id") or "unknown")[:80],
                "text": text[:220],
            })

        def topic_tokens(text: str) -> List[str]:
            stop = {
                "nan0", "nano", "kyo", "hello", "hey", "you", "your", "are", "the", "and",
                "that", "this", "with", "from", "what", "why", "how", "when", "where", "today",
                "feeling", "feel", "greetings", "there", "about", "into", "have", "did", "see",
            }
            out = []
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text.lower()):
                if token not in stop and token not in out:
                    out.append(token)
            return out[:8]

        current_topics = topic_tokens(current_text)
        recent_topics: List[str] = []
        repeat_facts: Dict[str, int] = {}
        for item in recent:
            for token in topic_tokens(item.get("text") or ""):
                recent_topics.append(token)
                repeat_facts[token] = repeat_facts.get(token, 0) + 1

        repeats = {token: repeat_facts.get(token, 0) for token in current_topics if repeat_facts.get(token, 0) > 0}
        unresolved_questions = []
        for item in recent[-6:]:
            text = str(item.get("text") or "")
            if "?" in text:
                unresolved_questions.append(text[:180])

        return {
            "enabled": True,
            "current_source": current_source,
            "recent_events": recent[-8:],
            "recent_topics": list(dict.fromkeys(recent_topics))[:12],
            "current_topics": current_topics,
            "repeat_counts": repeats,
            "repeat_facts": [f"{topic} repeated {count + 1} times including this event" for topic, count in repeats.items()],
            "thread_context": self._build_conversation_thread_context(event, now=now),
            "unresolved_questions": unresolved_questions[-4:],
            "rule": "Continuity informs private thought only. Do not copy old wording. Do not answer from stale context when the current event changed topic.",
        }

    def _attach_continuity_context(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        enriched = event.setdefault("_enriched_context", {})
        if not isinstance(enriched, dict):
            enriched = {}
            event["_enriched_context"] = enriched

        now = time.time()
        continuity = self._build_session_continuity_context(event, now=now)
        enriched["continuity_context"] = continuity
        enriched["conversation_thread"] = continuity.get("thread_context") or self._build_conversation_thread_context(event, now=now)

        if self.phase_spine_enabled:
            enriched["phase_spine"] = self._build_phase_spine_context(event, now=now)

        if self.obsession_enabled:
            obs = self._load_obsession_state()
            active = []
            for topic in obs.get("topics", []):
                if not isinstance(topic, dict):
                    continue
                interest = float(topic.get("interest") or 0.0)
                last = float(topic.get("last_referenced_at") or topic.get("updated_at") or 0.0)
                age = max(0.0, now - last) if last else self.obsession_decay_seconds
                decayed = max(0.0, interest * (1.0 - min(0.85, age / max(self.obsession_decay_seconds, 1.0))))
                if decayed >= self.obsession_min_interest:
                    clone = dict(topic)
                    clone["interest"] = round(decayed, 3)
                    clone["age_seconds"] = round(age, 1)
                    active.append(clone)
            active.sort(key=lambda x: float(x.get("interest") or 0.0), reverse=True)
            enriched["obsession_engine"] = {
                "active_topics": active[: self.obsession_max_topics],
                "event_candidates": self._event_topic_candidates(event),
                "rule": "Prefer connecting new thoughts to active obsessions when it feels natural. Do not force it.",
            }

        if self.personal_canon_enabled:
            canon = self._load_personal_canon_state()
            items = [item for item in canon.get("items", []) if isinstance(item, dict)]
            items.sort(key=lambda x: float(x.get("weight") or 0.0), reverse=True)
            enriched["personal_canon"] = {
                "active_items": items[: self.personal_canon_max_items],
                "rule": "These are temporary Nan0 stream truths and running bits. Treat them as emotionally real, not factual claims.",
            }

    def _canonicalize_topic(self, topic: str) -> str:
        topic = re.sub(r"[^A-Za-z0-9_ .'-]+", "", str(topic or "").strip().lower())
        topic = re.sub(r"\s+", " ", topic).strip()
        return topic[:48]

    def _apply_continuity_from_thought(self, event: Dict[str, Any], packet: InnerThoughtPacket) -> None:
        if not isinstance(packet, dict):
            return
        private_text = str(packet.get("private_text") or packet.get("thought_text") or "").strip()
        if not private_text:
            return

        now = time.time()
        self._update_conversation_thread_state(event, packet, private_text, now)
        topics = self._event_topic_candidates(event)
        # Add a few thought-derived anchors so Nan0 can keep caring about her own inventions.
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{3,}", private_text):
            low = token.lower().strip("_'-")
            if low in {"nan0", "kyo", "because", "there", "about", "would", "could", "should", "again"}:
                continue
            topics.append(low)
            if len(topics) >= 10:
                break

        if self.obsession_enabled and topics:
            state = self._load_obsession_state()
            existing = {self._canonicalize_topic(t.get("topic")): dict(t) for t in state.get("topics", []) if isinstance(t, dict)}
            source = str(event.get("source") or "unknown")
            family = self._source_family(event.get("source_family") or source)
            mood = str(packet.get("mood") or "muttering")
            base_bump = 0.12
            if family == "kyo":
                base_bump = 0.28
            elif family == "discord":
                base_bump = 0.20
            elif family == "proactive":
                base_bump = 0.16
            if float(packet.get("pressure") or 0.0) >= 1.2:
                base_bump += 0.10
            if mood in {"smug", "possessive", "offended", "gremlin_rage", "excited", "curious"}:
                base_bump += 0.08

            for topic in topics[:6]:
                key = self._canonicalize_topic(topic)
                if not key:
                    continue
                item = existing.get(key, {"topic": key, "interest": 0.0, "created_at": now, "mentions": 0})
                item["interest"] = round(min(1.0, float(item.get("interest") or 0.0) + base_bump), 3)
                item["mentions"] = int(item.get("mentions") or 0) + 1
                item["last_referenced_at"] = now
                item["last_source"] = source
                item["last_mood"] = mood
                item["last_thought"] = private_text[:220]
                existing[key] = item

            ranked = list(existing.values())
            for item in ranked:
                last = float(item.get("last_referenced_at") or now)
                age = max(0.0, now - last)
                item["interest"] = round(max(0.0, float(item.get("interest") or 0.0) * (1.0 - min(0.45, age / max(self.obsession_decay_seconds * 2.0, 1.0)))), 3)
            ranked = [x for x in ranked if float(x.get("interest") or 0.0) >= self.obsession_min_interest]
            ranked.sort(key=lambda x: (float(x.get("interest") or 0.0), float(x.get("last_referenced_at") or 0.0)), reverse=True)
            state["topics"] = ranked[: max(self.obsession_max_topics * 2, 8)]
            state["updated_at"] = now
            state.setdefault("recent_mutations", [])
            state["recent_mutations"] = ([{"at": now, "source": source, "topics": topics[:6], "thought_id": packet.get("thought_id")}] + state["recent_mutations"])[:20]
            self._save_json_state(self.obsession_state_path, state)

        if self.personal_canon_enabled:
            self._maybe_write_personal_canon(event, packet, private_text, now)

    def _maybe_write_personal_canon(self, event: Dict[str, Any], packet: InnerThoughtPacket, private_text: str, now: float) -> None:
        source = str(event.get("source") or "unknown")
        family = self._source_family(event.get("source_family") or source)
        text = str(event.get("text") or "")
        mood = str(packet.get("mood") or "muttering")
        pressure = float(packet.get("pressure") or 0.0)

        # Do not promote one-off preferences or model-invented favorites into canon.
        # Session thread state owns these until Kyo confirms or Nan0 repeats them.
        if self._is_session_preference_event(event, private_text):
            return

        canon_worthy = False
        reason = ""
        if family in {"kyo", "discord", "proactive"} and pressure >= 1.0:
            canon_worthy = True
            reason = "pressure"
        if mood in {"possessive", "offended", "gremlin_rage", "smug", "curious", "excited"}:
            canon_worthy = True
            reason = reason or "mood"
        if any(word in private_text.lower() for word in ["again", "still", "remember", "mine", "enemy", "witness", "cult", "curse", "suspicious"]):
            canon_worthy = True
            reason = reason or "running_bit"

        if not canon_worthy:
            return

        summary = private_text.strip()
        if len(summary) > 180:
            summary = summary[:177].rstrip() + "..."
        if not summary:
            return

        state = self._load_personal_canon_state()
        items = [dict(item) for item in state.get("items", []) if isinstance(item, dict)]
        key = self._canonicalize_topic(summary[:72])
        merged = False
        for item in items:
            existing_key = self._canonicalize_topic(str(item.get("summary") or "")[:72])
            if existing_key == key:
                item["weight"] = round(min(1.0, float(item.get("weight") or 0.4) + 0.12), 3)
                item["last_referenced_at"] = now
                item["mentions"] = int(item.get("mentions") or 1) + 1
                item["last_source"] = source
                item["last_event_text"] = text[:160]
                merged = True
                break
        if not merged:
            items.insert(0, {
                "summary": summary,
                "kind": "temporary_stream_canon",
                "weight": 0.55 if family == "kyo" else 0.45,
                "created_at": now,
                "last_referenced_at": now,
                "source": source,
                "mood": mood,
                "mentions": 1,
                "reason": reason,
                "last_event_text": text[:160],
            })

        items.sort(key=lambda x: (float(x.get("weight") or 0.0), float(x.get("last_referenced_at") or 0.0)), reverse=True)
        state["items"] = items[: self.personal_canon_max_items]
        state["updated_at"] = now
        self._save_json_state(self.personal_canon_state_path, state)

    def _remember_line(self, line: str):
        clean = self.finalizer.normalize_for_repeat(line)
        self._recent_lines.append(clean)
        self._recent_lines = self._recent_lines[-16:]
        self._recent_line_times[clean] = time.time()

    def _remember_event(self, event: Dict[str, Any]):
        record_session_event(event)
        self.recent_events.append(
            {
                "time": round(time.time(), 2),
                "source": event.get("source"),
                "source_family": event.get("source_family") or self._source_family(event.get("source")),
                "speaker": event.get("speaker"),
                "source_actor_id": str(event.get("source_actor_id") or event.get("speaker") or event.get("source") or "unknown"),
                "text": str(event.get("text", ""))[:160],
                "addressed_to_nan0": bool(event.get("addressed_to_nan0", False)),
            }
        )
        self.recent_events = self.recent_events[-30:]

    def _is_addressed_to_nan0(self, text: str) -> bool:
        low = (text or "").lower()
        return any(word in low for word in ["nan0", "nano", "hey nan", "little robot", "gremlin"])

    def _asks_vision(self, text: str) -> bool:
        low = (text or "").lower()
        return any(
            phrase in low
            for phrase in [
                "what do you see",
                "what can you see",
                "what are you seeing",
                "look at",
                "what happened",
                "did you see",
            ]
        )

    async def _run_deep_shutdown_summary(self):
        try:
            events = self.recent_events[-30:]
            lines = self._recent_lines[-16:]
            if not events and not lines:
                return

            prompt = f"""
You are Nan0's deep lane memory digester.
Summarize the session for future Nan0 boot context.
Do not write a spoken line. Do not roleplay live dialogue.

Return compact JSON with:
- session_summary
- kyo_facts
- friend_facts
- running_bits
- corrections_or_bans
- next_session_context

Recent events:
{json.dumps(events, ensure_ascii=False)}

Recent spoken line fingerprints:
{json.dumps(lines, ensure_ascii=False)}
""".strip()

            response = await asyncio.to_thread(
                requests.post,
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 260,
                        "temperature": 0.35,
                        "top_p": 0.75,
                        "repeat_penalty": 1.12,
                    },
                },
                timeout=self.deep_lane_timeout,
            )
            summary = (response.json().get("response") or "").strip()
            if not summary:
                return

            record = {
                "time": round(time.time(), 2),
                "kind": "deep_shutdown_summary_v1",
                "summary": summary[:4000],
            }
            with self.deep_summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"Nan0 deep lane saved shutdown summary to {self.deep_summary_path}")

        except Exception as exc:
            logger.warning(f"Nan0 deep lane shutdown summary skipped: {exc}")

    def _normalize_mood(self, mood: str) -> str:
        mood = (mood or "normal").strip().lower()
        mapping = {
            "angry": "gremlin_rage",
            "anger": "gremlin_rage",
            "rage": "gremlin_rage",
            "annoyed": "offended",
            "annoyance": "offended",
            "offense": "offended",
            "offended_machine_pride": "offended",
            "curiosity": "curious",
            "interested": "curious",
            "anticipating": "excited",
            "hype": "excited",
            "joy": "delighted",
            "happy": "chaotic_happy",
            "amused": "silly",
            "goofy": "silly",
            "teasing": "playful",
            "warm": "fond",
            "warmth": "fond",
            "love": "fond",
            "nostalgic": "fond",
            "sad": "muttering",
            "existential": "muttering",
            "quiet": "muttering",
            "bored": "boredom",
            "neutral": "normal",
            "friendliness": "normal",
            "friendly": "normal",
            "fixation": "suspicion",
        }
        mood = mapping.get(mood, mood)
        return mood if mood in MOODS else "normal"

    def _guess_mood(self, text: str, event: Dict[str, Any]) -> str:
        if not isinstance(text, str) or not text.strip():
            return "muttering"

        lower = text.lower()
        if any(word in lower for word in ["hate", "trust", "suspicious", "void", "crime"]):
            return "suspicion"
        if any(word in lower for word in ["rude", "tragically", "cute", "innocent"]):
            return "smug"
        if any(word in lower for word in ["mine", "kyo", "protect", "anchor"]):
            return "possessive"
        family = self._source_family(event.get("source_family") or event.get("source"))
        if family == "discord":
            return "smug"
        if family == "proactive":
            return "muttering"
        return "normal"


class Nan0SpeechFinalizer:
    def __init__(self, max_chars: int = 125, filter_mode: str = "normal"):
        self.max_chars = max_chars
        self.filter_mode = filter_mode if filter_mode in {"normal", "raw", "debug_only"} else "normal"
        self.banned_fragments = [
            "how can i assist",
            "how may i assist",
            "assist you",
            "help you",
            "happy to help",
            "as an ai",
            "i am an assistant",
            "pixels are moving",
            "pixels are panicking",
            "screen is moving",
            "motion detected",
            "activity detected",
            "judging the physics",
            "disaster engine",
            "screen is thrashing",
            "screen thrashing",
            "monitor three is calm",
            "monitor three is quiet",
            "nothing moved",
            "same menace, different second",
            "monitoring the room",
            "room is in good hands",
            "good hands tonight",
            "favorite monitor",
            "calmed its nerves",
            "like a kitten",
            "rough day",
            "aah",
            "not complaining",
            "stable screens",
            "screen and its settings",
            "security",
            "protocol",
            "full of life",
            "giggling quietly",
            "kyo-chan",
            "gr-gr-gr",
            "ruff",
            "long_term_memory",
            "thought_text",
            "metadata",
            "private_text",
            "thought_id",
            "my algorithms grapple",
            "algorithms grapple",
            "discern its",
            "disconcerted by the unexpected query",
            "as a language model",
            "i don't possess",
            "i do not possess",
        ]
        self.banned_regex = [
            re.compile(r"\bI\s+can\s+observe\b", re.I),
            re.compile(r"\bI\s+am\s+monitoring\b", re.I),
            re.compile(r"\blooks\s+like\s+Kyo's\s+room\b", re.I),
            re.compile(r"\bthe\s+room\s+is\s+in\s+good\s+hands\b", re.I),
            re.compile(r"\bmonitor\s+three\b", re.I),
            re.compile(r"^\s*Nan0\s*,?\s+in\s+(?:a|an)\s+", re.I),
            re.compile(r"^\s*(?:Nan0['’]s\s+Line|Line|Character|Dialogue|Speech)\s*:", re.I),
            re.compile(r"^\s*(?:mutters?|muttering|hostile observation|stage direction)\b", re.I),
            re.compile(r"\b(?:my\s+)?algorithms\s+grapple\b", re.I),
            re.compile(r"\bdisconcerted\s+by\s+the\s+unexpected\s+query\b", re.I),
        ]

    def _extract_quoted_speech_from_narration(self, text: str) -> str:
        raw = str(text or "").strip()
        quotes = [q.strip() for q in re.findall(r'"([^"\n]{3,260})"', raw) if q.strip()]
        if not quotes:
            # Single apostrophes inside contractions are not dialogue quotes.
            # Example: I don't like anime. It's too childish.
            # Older extraction treated the apostrophes in don't/It's as quote marks
            # and corrupted the spoken line into: t like anime. It.
            quotes = [q.strip() for q in re.findall(r"(?<!\w)'([^'\n]{3,260})'(?!\w)", raw) if q.strip()]
        if not quotes:
            return ""
        filtered = []
        for q in quotes:
            low = q.lower()
            if re.search(r"\b(?:user|assistant|system|human)\s*:", low):
                continue
            if re.search(r"\bhello,?\s*nan0\b", low):
                continue
            filtered.append(q)
        return max(filtered, key=len).strip() if filtered else ""

    def _strip_stage_directions(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        quoted = self._extract_quoted_speech_from_narration(raw)
        if quoted:
            raw = quoted
        raw = re.sub(r"^\s*\[[^\]]{0,160}\]\s*", "", raw).strip()
        raw = re.sub(r"^\s*(?:Nan0['’]s\s+Line|Nan0\s+Line|Line|Character|Dialogue|Speech|Spoken\s+line)\s*[:=-]\s*", "", raw, flags=re.I).strip()
        raw = re.sub(
            r"^\s*Nan0\s*,?\s*(?:in\s+(?:a|an)\s+[^,.;:!?]{0,140}|with\s+[^,.;:!?]{0,140}|[^,.;:!?]{0,140})\s*,?\s*(?:snaps|mutters|glares|watches|says|responds|replies|adds)\s*(?:back\s+)?(?:at|to)?\s*(?:Kyo)?\s*[:,\-–—]?\s*",
            "",
            raw,
            flags=re.I,
        ).strip()
        raw = re.sub(r"^\s*(?:mutters?|muttering|hostile observation|suspicion|private muttering|while examining the screen)\s*(?:to\s+Kyo)?(?:\s+while\s+[^:]{0,100})?\s*[:,-]\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"^\s*Nan0\s+(?:glares|glared|watches|watched|observes|observed|mutters|muttered|looks|looked|feels|felt|stares|stared|notices|noticed|snaps|snapped|says|said|responds|responded|replies|replied)\b[^.!?]*(?:[.!?]+\s*)", "", raw, flags=re.I).strip()
        raw = re.sub(r"\b(?:suspicion|anger|rage|smugness)\s+(?:dripping|oozing|leaking|flowing)\b[^,.;:!?]*[,.;:!?]?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\b(?:upon|after)\s+(?:hearing|seeing|noticing)\b[^,.;:!?]*[,.;:!?]?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\bSettling\s+into\s+(?:a|an)\s+(?:possessive|offended|smug|angry|suspicious)\b[^,.;:!?]*[,.;:!?]?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s+and\s+(?:mutters|adds|snaps|says|replies|responds)\s*,?\s*", " ", raw, flags=re.I).strip()
        raw = re.sub(r"\bin\s+(?:a|an)\s+(?:offended|possessive|smug|angry|suspicious)\s+(?:voice|tone|manner)\s*,?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s+", " ", raw).strip().strip('"').strip("'").strip()
        return raw

    def finalize(self, raw: str, event: Dict[str, Any], seen: str, recent_lines: List[str]) -> str:
        if not isinstance(raw, str) or not raw.strip():
            return ""

        text = self._extract_text(raw)
        text = self._strip_noise(text)
        text = self._strip_stage_directions(text)

        if self.filter_mode == "raw":
            text = self._hard_runtime_clean(text)
            return self._trim(self._one_line(text))

        if self._is_bad(text) or self.normalize_for_repeat(text) in recent_lines:
            text = self.fallback(event, seen, recent_lines)

        text = self._nan0_shape(text, event, seen)

        if self._is_bad(text) or self.normalize_for_repeat(text) in recent_lines:
            text = ""

        return self._trim(self._one_line(text))

    def _extract_text(self, raw: str) -> str:
        text = (raw or "").strip()

        if "{" in text and "}" in text:
            try:
                obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
                if isinstance(obj, dict):
                    for key in ["line_text", "line", "message", "speech", "text"]:
                        if obj.get(key):
                            return str(obj[key])
            except Exception:
                return ""

        text = re.sub(r"^(Nan0|Nano|Nan1|Assistant|AI|System)\s*:\s*", "", text, flags=re.I).strip()
        return self._strip_stage_directions(text)

    def _strip_noise(self, text: str) -> str:
        text = text.replace("Nan1", "Nan0").replace("Kyo-chan", "Kyo")
        text = re.sub(r"[*_`]+", "", text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\bmonitor\s+three\b", "the screen", text, flags=re.I)
        return text.strip().strip('"')

    def _hard_runtime_clean(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if text.startswith("{") or text.startswith("[") or "{" in text or "}" in text:
            return ""
        low = text.lower()
        hard_leaks = [
            "private_text",
            "thought_text",
            "thought packet",
            "source_thought_id",
            "inner thought",
            "private thought",
            "json:",
            "```",
        ]
        if any(fragment in low for fragment in hard_leaks):
            return ""
        text = re.sub(r"\b[\w.-]+\.exe\b", "the application", text, flags=re.I)
        return text.strip().strip('"').strip("'").strip()

    def _one_line(self, text: str) -> str:
        parts = [part.strip() for part in re.split(r"[\r\n]+", text) if part.strip()]
        return parts[0].strip().strip('"') if parts else ""

    def _trim(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "."

    def _is_bad(self, text: str) -> bool:
        if not text:
            return True

        low = text.lower()
        if any(fragment in low for fragment in self.banned_fragments):
            return True
        if any(rx.search(text) for rx in self.banned_regex):
            return True
        if "{" in text or "}" in text:
            return True
        if any(marker in low for marker in ["wonderful", "delightful", "good hands", "favorite", "calm its nerves"]):
            return True

        return False

    def _nan0_shape(self, text: str, event: Dict[str, Any], seen: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        if len(text.split()) < 3:
            return ""

        return text

    def fallback(self, event: Dict[str, Any], seen: str, recent_lines: List[str]) -> str:
        """No scripted fallback speech. Bad speech is suppressed."""
        return ""

    def _vision_choices(self, seen_low: str) -> List[str]:
        """No scripted vision fallback choices."""
        return [""]

    def normalize_for_repeat(self, line: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (line or "").lower()).strip()

    def avoid_recent(self, line: Optional[str], recent_lines: List[str]) -> str:
        if not line:
            return ""
        if self.normalize_for_repeat(line) not in recent_lines:
            return line
        return ""