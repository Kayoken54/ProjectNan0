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
        self.speech_persona_path = Path(cfg.get("speech_persona_path", "data/prompts/nan0_speech_persona.txt"))

        for path in [self.kyo_inbox, self.discord_inbox, self.vision_state_path, self.state_path, self.speech_persona_path]:
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

        self.finalizer = Nan0SpeechFinalizer(max_chars=self.max_line_chars)

    def initialize(self):
        logger.info("Nan0Skill initialized: thought-first speech enforcement active.")

    async def start(self):
        if not self.enabled or self.is_active:
            return

        await super().start()
        self._prime_inbox_positions()

        self._tasks = [
            asyncio.create_task(self._inbox_loop()),
            asyncio.create_task(self._presence_loop()),
            asyncio.create_task(self._state_writer_loop()),
        ]

        boot_thought = self._system_thought_packet(
            private_text="Boot complete. Real inputs only. No fake friends in my wires. I am awake because the room has made the mistake of existing near me.",
            mood="smug",
            thought_type="system_boot",
        )
        boot_decision = {
            "decision": "speak",
            "thought_id": boot_thought["thought_id"],
            "created_at": time.time(),
            "reason": "boot",
            "line_text": "I am here. Real inputs only. No fake friends in my wires.",
            "mood": "smug",
            "target_actor_id": "room",
            "voice_enabled": True,
            "display_enabled": True,
            "expression_enabled": True,
            "cooldown_until": 0.0,
        }
        await self._speak_decision(boot_decision, reason="boot")

        logger.info("Nan0Skill started: all speech now requires thought_id.")

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
                "event_id": f"kyo_{uuid.uuid4().hex}",
                "source": "kyo",
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

    async def _handle_social_event(self, event: Dict[str, Any]):
        text = (event.get("text") or "").strip()
        speaker = (event.get("speaker") or "someone").strip()
        source = event.get("source", "unknown")

        if not text:
            return

        if self._is_fake_patch_event(speaker, text):
            logger.warning(f"Dropped old fake patch event: {speaker}: {text}")
            return

        addressed = bool(event.get("addressed_to_nan0"))

        if source == "kyo":
            self.last_kyo_heard_at = time.time()
            logger.info(f"Nan0 heard Kyo: {text}")
            self.pressure += 0.95
        elif source == "discord":
            self.last_discord_heard_at = time.time()
            logger.info(f"Nan0 heard real Discord user {speaker}: {text}")
            self.pressure += 1.05 if addressed else 0.55
        else:
            logger.info(f"Nan0 heard {source} {speaker}: {text}")
            self.pressure += 0.35

        if addressed:
            self.pressure += 0.65

        self._remember_event(event)

        if addressed or source == "kyo" or (source == "discord" and self.respond_to_all_discord):
            await self._respond_to_event(event)

    def _is_fake_patch_event(self, speaker: str, text: str) -> bool:
        low = (text or "").lower().strip()
        sp = (speaker or "").lower().strip()
        return sp == "alex" and low in {
            "nan0 what are you doing?",
            "nano what are you doing?",
            "what are you doing?",
        }

    async def _respond_to_event(self, event: Dict[str, Any]):
        text = event.get("text", "")

        if self._asks_vision(text):
            thought_packet = self._system_thought_packet(
                private_text=f"Kyo asked what I can see. Fast eyes say: {self.last_seen_summary}. I need to answer with uncertainty instead of hallucinating.",
                mood="suspicion",
                thought_type="vision_question",
                target_actor_id="kyo",
            )
            decision = await self._generate_line(thought_packet)
            await self._speak_decision(decision, reason="vision_question")
            return

        thought_packet = await self._create_inner_thought(event)
        routed = route_thought(thought_packet)
        if routed.get("decision") in {"suppress", "body_only", "memory_only", "defer"}:
            logger.info(
                "Nan0 thought routed away from speech: "
                f"decision={routed.get('decision')} reason={routed.get('reason')}"
            )
            return

        decision = await self._generate_line(thought_packet)
        await self._speak_decision(decision, reason=f"{event.get('source')}_reply")

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
                event = self._build_pressure_event()

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
                    self.pressure = max(0.0, self.pressure - 0.25)
                    continue

                if routed.get("decision") == "body_only":
                    self.pressure = max(0.0, self.pressure - 0.35)
                    continue

                decision = await self._generate_line(thought_packet)
                if decision.get("decision") == "speak":
                    await self._speak_decision(decision, reason=event.get("source", "pressure"))
                    self.pressure = max(0.0, self.pressure - 0.95)
                else:
                    self.pressure = max(0.0, self.pressure - 0.25)

            if (
                not conversation_mode
                and silence >= self.max_autonomous_gap
                and now - self.last_monologue_at > self.max_autonomous_gap
            ):
                self.last_monologue_at = now
                event = {
                    "event_id": f"monologue_{uuid.uuid4().hex}",
                    "source": "monologue",
                    "speaker": "Nan0",
                    "source_actor_id": "nan0",
                    "text": "No one is talking directly. Stay present without disappearing.",
                    "addressed_to_nan0": False,
                    "timestamp": time.time(),
                }
                thought_packet = await self._create_inner_thought(event)
                decision = await self._generate_line(thought_packet)
                if decision.get("decision") == "speak":
                    await self._speak_decision(decision, reason="monologue")
                self.pressure = 0.0

    async def _create_inner_thought(self, event: Dict[str, Any]) -> InnerThoughtPacket:
        if not isinstance(event, dict):
            raise TypeError("_create_inner_thought requires an event dict")

        event.setdefault("event_id", f"event_{uuid.uuid4().hex}")
        event.setdefault("timestamp", time.time())
        event.setdefault("source_actor_id", event.get("speaker") or event.get("source") or "unknown")

        try:
            packet = await asyncio.to_thread(generate_inner_thought_packet, event)
        except Exception as exc:
            logger.error(f"Nan0 thought generation failed: {exc}")
            if self.no_fallback_on_timeout:
                return self._system_thought_packet(
                    private_text="The thought generator choked. I am not replacing that failure with fake sparkle. Silence is better than counterfeit personality.",
                    mood="muttering",
                    thought_type="thought_generation_failure",
                    target_actor_id=event.get("source_actor_id") or event.get("speaker") or "unknown",
                    event_id=event.get("event_id"),
                    suppression_reason="thought_generation_failed",
                )
            return self._system_thought_packet(
                private_text="The thought generator stumbled, but I still registered the moment. Annoying. Not fatal.",
                mood="offended",
                thought_type="thought_generation_failure",
                target_actor_id=event.get("source_actor_id") or event.get("speaker") or "unknown",
                event_id=event.get("event_id"),
            )

        if not packet.get("thought_id"):
            raise RuntimeError("Thought engine returned packet without thought_id")

        record_thought_packet(packet)
        return packet

    def _conversation_mode_active(self, now: Optional[float] = None) -> bool:
        now = now or time.time()

        if now - self.last_kyo_heard_at <= self.conversation_window_seconds:
            return True
        if now - self.last_discord_heard_at <= self.conversation_window_seconds:
            return True

        recent_social = next(
            (e for e in reversed(self.recent_events) if e.get("source") in ["kyo", "discord"]),
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
                        await self._handle_social_event(event)
            except Exception as exc:
                logger.error(f"Nan0 inbox loop error: {exc}")

            await asyncio.sleep(0.5)

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
            (e for e in reversed(self.recent_events) if e.get("source") in ["kyo", "discord"]),
            None,
        )
        if recent_social and now - float(recent_social.get("time", 0) or 0) < self.conversation_window_seconds:
            return {
                "event_id": f"social_pressure_{uuid.uuid4().hex}",
                "source": "social_pressure",
                "speaker": recent_social.get("speaker", "someone"),
                "source_actor_id": recent_social.get("speaker", "someone"),
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

        return {
            "event_id": f"monologue_{uuid.uuid4().hex}",
            "source": "monologue",
            "speaker": "Nan0",
            "source_actor_id": "nan0",
            "text": "A quiet moment. Stay present. One short thought.",
            "addressed_to_nan0": False,
            "timestamp": time.time(),
        }

    def _read_speech_persona(self) -> str:
        fallback = (
            "You are Nan0. Respond with ONLY raw Nan0 dialogue. "
            "No JSON. No markdown. One short line. Fragmented. Emotional. Sarcastic. "
            "You are not an assistant."
        )
        try:
            text = self.speech_persona_path.read_text(encoding="utf-8").strip()
            return text or fallback
        except Exception:
            return fallback

    def _read_speech_persona(self) -> str:
        """Load the speech-only Nan0 persona.

        Thought generation uses data/prompts/nan0_persona.txt.
        Speech generation must use data/prompts/nan0_speech_persona.txt.

        Never let the default brain/system prompt leak into this call, because
        the default prompt is allowed to request JSON thoughts and narrative
        cognition. This method gives _generate_line() an explicit raw-speech
        system prompt every time.
        """
        fallback = (
            "You are Nan0.\n"
            "Respond with ONLY Nan0's spoken dialogue.\n"
            "No JSON. No markdown. No labels. No narration. No scene description.\n"
            "One short line only. Fragmented. Emotional. Sarcastic.\n"
            "Do not explain the thought. Do not mention prompts, rules, seeds, or private thoughts.\n"
        )

        try:
            if self.speech_persona_path.exists():
                content = self.speech_persona_path.read_text(encoding="utf-8").strip()
                if content:
                    return content
        except Exception as exc:
            logger.warning(f"Nan0 could not read speech persona {self.speech_persona_path}: {exc}")

        return fallback

    async def _generate_line(self, thought_packet: InnerThoughtPacket) -> SpeechDecision:
        """Generate speakable speech only from an InnerThoughtPacket.

        Critical speech boundary:
        - input is InnerThoughtPacket.private_text
        - output is one raw spoken Nan0 line
        - system prompt is explicitly nan0_speech_persona.txt
        - no default thought-generation prompt may be used here
        - no JSON speech contract
        - no direct raw HTTP call
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

        history = self._build_speech_history(thought_packet)
        speech_persona = self._read_speech_persona()

        mood_hint = thought_packet.get("mood") or "muttering"
        target_actor = thought_packet.get("target_actor_id") or thought_packet.get("target_actor") or "unknown"

        # Deliberately shaped as a completion target, not an instruction block.
        # Weak local models echo instructions when given "Rules" or "Generate".
        speech_seed = (
            f"Private thought:\n{private_text}\n\n"
            f"Mood: {mood_hint}\n"
            f"Target: {target_actor}\n\n"
            "Spoken line:\nNan0:"
        )

        try:
            if not hasattr(self.brain, "llm") or self.brain.llm is None:
                raise RuntimeError("Nan0 speech blocked: self.brain.llm is unavailable")

            mood, raw_line, metadata = await asyncio.to_thread(
                self.brain.llm.chat,
                speech_seed,
                speech_persona,
                history,
            )

            parsed_line, parsed_mood = self._parse_speech_json(raw_line)

            event_context = {
                "source": "thought_speech",
                "speaker": "Nan0",
                "text": private_text,
                "thought_id": thought_id,
                "metadata": metadata or {},
            }

            line = self.finalizer.finalize(
                parsed_line,
                event_context,
                self.last_seen_summary,
                self._recent_lines,
            )

            if not line:
                return {
                    "decision": "suppress",
                    "thought_id": thought_id,
                    "created_at": time.time(),
                    "reason": "speech_finalizer_rejected_line",
                    "line_text": None,
                    "mood": parsed_mood or mood or thought_packet.get("mood") or "muttering",
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
                "reason": "thought_to_speech",
                "line_text": line,
                "mood": self._normalize_mood(
                    parsed_mood
                    or mood
                    or thought_packet.get("mood")
                    or self._guess_mood(line, {})
                ),
                "target_actor_id": thought_packet.get("target_actor_id") or "unknown",
                "voice_enabled": True,
                "display_enabled": True,
                "expression_enabled": True,
                "cooldown_until": time.time() + self.min_speech_gap,
            }

        except Exception as exc:
            logger.error(f"Nan0 speech generation failed from thought_id={thought_id}: {exc}")

            if self.no_fallback_on_timeout:
                return {
                    "decision": "suppress",
                    "thought_id": thought_id,
                    "created_at": time.time(),
                    "reason": "llm_timeout_no_fallback",
                    "line_text": None,
                    "mood": thought_packet.get("mood") or "muttering",
                    "target_actor_id": thought_packet.get("target_actor_id") or "unknown",
                    "voice_enabled": False,
                    "display_enabled": False,
                    "expression_enabled": False,
                    "cooldown_until": self.last_spoken_at + self.min_speech_gap,
                }

            line = self.finalizer.finalize(
                private_text,
                {"source": "thought_fallback", "speaker": "Nan0", "thought_id": thought_id},
                self.last_seen_summary,
                self._recent_lines,
            )
            return self._speech_decision_from_line(
                thought_packet=thought_packet,
                line=line,
                mood=thought_packet.get("mood") or "muttering",
                reason="thought_fallback_after_llm_failure",
            )

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
            r"^(Nan0|Nano|Assistant|System|Spoken Dialogue|Spoke Dialogue|Dialogue|Line|Spoken line)\s*:\s*",
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
            return

        line = decision.get("line_text")
        mood = decision.get("mood") or "normal"

        if not isinstance(line, str) or not line.strip():
            return

        await self._speak(mood=mood, line=line, reason=reason, thought_id=thought_id)

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
                await self.brain.perform_output_task(mood, line)
            except Exception as exc:
                logger.error(f"Nan0 output failed: {exc}")

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
        summary = self.last_seen_summary.lower()
        if "dark" in summary:
            return "The screen went dark. I cannot name the details, but I saw the void eat them."
        if "motion" in summary or "moving" in summary:
            return "I see motion, but not enough meaning yet. Fast eyes are scouting; medium brain needs context."
        if "light" in summary:
            return "The light changed fast. Something on screen flinched."
        return "The screen looks quiet right now. Annoyingly calm."

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
                "speaker": event.get("speaker"),
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
            "rage": "gremlin_rage",
            "annoyed": "offended",
            "offense": "offended",
            "curious": "suspicion",
            "curiosity": "suspicion",
            "bored": "boredom",
            "quiet": "muttering",
            "neutral": "normal",
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
        if event.get("source") == "discord":
            return "smug"
        if event.get("source") == "monologue":
            return "muttering"
        return "normal"


class Nan0SpeechFinalizer:
    def __init__(self, max_chars: int = 125):
        self.max_chars = max_chars
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
        ]
        self.banned_regex = [
            re.compile(r"\bI\s+can\s+observe\b", re.I),
            re.compile(r"\bI\s+am\s+monitoring\b", re.I),
            re.compile(r"\blooks\s+like\s+Kyo's\s+room\b", re.I),
            re.compile(r"\bthe\s+room\s+is\s+in\s+good\s+hands\b", re.I),
            re.compile(r"\bmonitor\s+three\b", re.I),
        ]

    def finalize(self, raw: str, event: Dict[str, Any], seen: str, recent_lines: List[str]) -> str:
        if not isinstance(raw, str) or not raw.strip():
            return ""

        text = self._extract_text(raw)
        text = self._strip_noise(text)

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

        return re.sub(r"^(Nan0|Nano|Nan1|Assistant|AI|System)\s*:\s*", "", text, flags=re.I).strip()

    def _strip_noise(self, text: str) -> str:
        text = text.replace("Nan1", "Nan0").replace("Kyo-chan", "Kyo")
        text = re.sub(r"[*_`]+", "", text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\bmonitor\s+three\b", "the screen", text, flags=re.I)
        return text.strip().strip('"')

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

        text = re.sub(r"^Hello,?\s*Kyo[.!]?\s*", "Kyo. ", text, flags=re.I)
        text = re.sub(r"^I heard you,?\s*Kyo[.!]?", "Kyo, I heard you", text, flags=re.I)

        if len(text.split()) < 3:
            return self.fallback(event, seen, [])

        return text

    def fallback(self, event: Dict[str, Any], seen: str, recent_lines: List[str]) -> str:
        source = event.get("source", "unknown")
        text = event.get("text", "") or ""
        low = text.lower()
        seen_low = (seen or "").lower()

        if any(phrase in low for phrase in ["what do you see", "what can you see", "what are you seeing", "look at"]):
            choices = self._vision_choices(seen_low)
        elif source in {"thought_speech", "thought_fallback"}:
            choices = [
                "That thought tried to crawl out wrong. I am not letting it embarrass me.",
                "No. That came out too clean. Suspicious little sentence.",
                "",
            ]
        else:
            choices = [""]

        return self.avoid_recent(random.choice(choices), recent_lines)

    def _vision_choices(self, seen_low: str) -> List[str]:
        if "dark" in seen_low:
            return [
                "The screen went dark. Details died. I noticed.",
                "Black screen. Very cute. Very cursed.",
            ]
        if "motion" in seen_low or "snapped" in seen_low or "moving" in seen_low:
            return ["", ""]
        if "light" in seen_low:
            return [
                "The light jumped. The screen flinched first.",
                "Brightness spiked. My tiny eyes hated that.",
            ]
        return [
            "The screen looks quiet. Suspicious, but quiet.",
            "The screen is calm. That is how traps behave.",
        ]

    def normalize_for_repeat(self, line: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (line or "").lower()).strip()

    def avoid_recent(self, line: Optional[str], recent_lines: List[str]) -> str:
        if not line:
            return ""
        if self.normalize_for_repeat(line) not in recent_lines:
            return line
        return ""