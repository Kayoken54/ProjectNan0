import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.modules.skills.base_skill import BaseSkill
from src.core.events import EventCategory
from src.modules.skills.implementations.nan0_thought_engine_v3 import generate_inner_thought_packet
from src.utils.logger import get_logger

logger = get_logger("bea.skills.nan0_vision")


class Nan0VisionSkill(BaseSkill):
    """
    Nan0 Vision Skill V2 - Thought-First Architecture Compliant.

    Runtime safety rules:
    - Vision may observe often.
    - Minor motion does not call the LLM.
    - Vision LLM interpretation is reserved for meaningful changes only.
    - Vision never fabricates Nan0 thoughts from local templates.
    - If the thought engine is unavailable, vision stays body/state only.
    """

    def initialize(self):
        cfg = self.skill_config

        self._enabled = bool(cfg.get("enabled", True))
        self.vision_state_path = Path(
            cfg.get("vision_state_path", "data/vision/nan0_vision_stack_state.json")
        )

        self.min_gap_seconds = float(cfg.get("min_gap_seconds", 12))
        self.max_thoughts_per_minute = int(cfg.get("max_thoughts_per_minute", 4))

        self.throttle_combat = bool(cfg.get("throttle_combat", True))
        self.combat_min_gap = float(cfg.get("combat_min_gap", 8))

        self.silence_threshold = float(cfg.get("silence_threshold", 0.15))

        # Frozen-thought-engine protection.
        # No ThreadPoolExecutor here: Nan0 cognition stays on the owned serial path.
        # If the thought engine cannot return a real packet, vision does not speak.
        self.vision_llm_enabled = bool(cfg.get("vision_llm_enabled", True))
        self.vision_llm_timeout_seconds = float(cfg.get("vision_llm_timeout_seconds", 3.0))
        self.thought_engine_cooldown_seconds = float(cfg.get("thought_engine_cooldown_seconds", 15.0))

        self._shutdown_requested = False

        self.last_thought_time = 0.0
        self.last_combat_time = 0.0
        self.last_thought_engine_call_at = 0.0

        self.thoughts_this_minute = 0
        self.minute_start = time.time()

        self.last_l1_hash = None
        self.last_l2_hash = None
        self.last_l3_hash = None
        self._stale_counter = 0

        self._last_menu_open: Optional[bool] = None
        self._last_screen_state: Optional[str] = None

    async def start(self):
        if not self.enabled or self.is_active:
            return
        self._shutdown_requested = False
        await super().start()

    async def stop(self):
        self._shutdown_requested = True
        await super().stop()

    async def update(self):
        if not getattr(self, "_enabled", True):
            return
        if getattr(self, "_shutdown_requested", False):
            return
        if not getattr(self, "is_active", False):
            return

        now = time.time()

        if now - self.last_thought_time < self.min_gap_seconds:
            return

        if now - self.minute_start >= 60:
            self.thoughts_this_minute = 0
            self.minute_start = now

        if self.thoughts_this_minute >= self.max_thoughts_per_minute:
            return

        state = self._read_state()
        if not state:
            return

        l1 = state.get("layer1", {})
        if not l1:
            l1 = state.get("layer1_reflex", {})
        if not l1:
            l1 = self._layer1(state)

        l2 = state.get("layer2", {})
        if not l2:
            l2 = state.get("layer2_semantic", {})
        if not l2:
            l2 = self._layer2(l1)

        l3 = self._layer3(l1, l2, now=now)

        if not l3.get("thoughtworthy"):
            return

        if l3.get("do_not_speak_if_only_silence") and not l1.get("major_change") and not l1.get("combat"):
            return

        if l1.get("combat") and self.throttle_combat:
            if now - self.last_combat_time < self.combat_min_gap:
                return
            self.last_combat_time = now

        thought_packet = self._thought_packet(l1, l2, l3, now)
        if not thought_packet:
            return

        await self._emit_thought(thought_packet)
        self.last_thought_time = now
        self.thoughts_this_minute += 1

    def _read_state(self) -> Dict[str, Any]:
        try:
            if not self.vision_state_path.exists():
                return {}
            return json.loads(self.vision_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _layer1(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "screen_state": raw.get("screen_state", "unknown"),
            "motion_intensity": float(raw.get("motion_intensity", 0)),
            "brightness": float(raw.get("brightness", 0)),
            "brightness_delta": float(raw.get("brightness_delta", 0)),
            "combat": bool(raw.get("combat", False)),
            "menu_open": bool(raw.get("menu_open", False)),
            "inventory_visible": bool(raw.get("inventory_visible", False)),
            "dark_scene": bool(raw.get("dark_scene", False)),
            "major_change": bool(raw.get("major_change", False)),
            "game_ui_detected": raw.get("game_ui_detected", "unknown"),
            "timestamp": raw.get("timestamp", time.time()),
        }

    def _layer2(self, l1: Dict[str, Any]) -> Dict[str, Any]:
        motion = float(l1.get("motion_intensity", 0) or 0)
        brightness_delta = float(l1.get("brightness_delta", 0) or 0)
        combat = bool(l1.get("combat", False))
        menu = bool(l1.get("menu_open", False))
        dark = bool(l1.get("dark_scene", False))
        game_ui = l1.get("game_ui_detected", "unknown")
        major = bool(l1.get("major_change", False))

        semantic = {
            "motion_level": "none",
            "brightness_trend": "stable",
            "scene_type": "idle",
            "attention_demand": "low",
        }

        if combat:
            semantic["motion_level"] = "extreme"
            semantic["scene_type"] = "combat"
            semantic["attention_demand"] = "high"
        elif motion > 0.5:
            semantic["motion_level"] = "high"
            semantic["scene_type"] = "active"
            semantic["attention_demand"] = "medium"
        elif motion > 0.2:
            semantic["motion_level"] = "medium"
            semantic["scene_type"] = "active"
            semantic["attention_demand"] = "low"
        elif motion > self.silence_threshold:
            semantic["motion_level"] = "low"
            semantic["scene_type"] = "calm"
            semantic["attention_demand"] = "low"
        else:
            semantic["motion_level"] = "none"
            semantic["scene_type"] = "idle"
            semantic["attention_demand"] = "none"

        if brightness_delta > 30:
            semantic["brightness_trend"] = "spike"
            semantic["attention_demand"] = "high"
        elif brightness_delta < -30:
            semantic["brightness_trend"] = "drop"
            semantic["attention_demand"] = "medium"
        elif brightness_delta > 10:
            semantic["brightness_trend"] = "rising"
        elif brightness_delta < -10:
            semantic["brightness_trend"] = "falling"

        if menu:
            semantic["scene_type"] = "menu"
            semantic["attention_demand"] = "low"
        if dark:
            semantic["scene_type"] = "dark"
            semantic["attention_demand"] = "medium"
        if major:
            semantic["attention_demand"] = "high"

        semantic["game_ui"] = game_ui
        return semantic

    def _build_vision_event(self, l1: Dict[str, Any], l2: Dict[str, Any]) -> Dict[str, Any]:
        screen_state = l1.get("screen_state", "unknown")
        motion = float(l1.get("motion_intensity", 0) or 0)
        brightness = float(l1.get("brightness", 0) or 0)
        brightness_delta = float(l1.get("brightness_delta", 0) or 0)
        combat = bool(l1.get("combat", False))
        menu_open = bool(l1.get("menu_open", False))
        inventory_visible = bool(l1.get("inventory_visible", False))
        dark_scene = bool(l1.get("dark_scene", False))
        major_change = bool(l1.get("major_change", False))
        game_ui = l1.get("game_ui_detected", "unknown")

        narrative_parts = []
        event_type = "vision_state"
        confidence = 0.35

        if combat:
            narrative_parts.append("The screen is in combat. Fast movement and high pressure.")
            event_type = "vision_combat"
            confidence = 0.72
        elif major_change:
            narrative_parts.append("A major visual change happened on screen.")
            event_type = "vision_major_change"
            confidence = 0.70
        elif motion > 0.5:
            narrative_parts.append("Everything on screen is moving fast.")
            event_type = "vision_motion"
            confidence = 0.58
        elif motion > 0.2:
            narrative_parts.append("Some movement caught my eye.")
            event_type = "vision_motion"
            confidence = 0.48
        else:
            narrative_parts.append("The screen is mostly still.")

        if dark_scene or screen_state == "very_dark":
            narrative_parts.append("The screen is dark.")
            event_type = "vision_dark"
            confidence = max(confidence, 0.62)
        elif brightness_delta > 30:
            narrative_parts.append("Brightness jumped hard.")
            event_type = "vision_brightness"
            confidence = max(confidence, 0.62)
        elif brightness_delta < -30:
            narrative_parts.append("Everything dimmed quickly.")
            event_type = "vision_brightness"
            confidence = max(confidence, 0.62)
        elif brightness_delta > 10:
            narrative_parts.append("The screen is getting brighter.")
        elif brightness_delta < -10:
            narrative_parts.append("The screen is getting darker.")

        if menu_open or inventory_visible:
            narrative_parts.append("A menu or UI layer appears to be open.")
            event_type = "vision_menu"
            confidence = max(confidence, 0.60)

        if game_ui == "coding":
            narrative_parts.append("Code or debugging appears to be on screen.")
        elif game_ui == "gameplay":
            narrative_parts.append("Gameplay appears to be active.")
        elif game_ui == "desktop":
            narrative_parts.append("The desktop appears to be visible.")
        elif game_ui == "browser":
            narrative_parts.append("A browser appears to be open.")

        narrative = " ".join(narrative_parts) if narrative_parts else "The screen changed, but the meaning is uncertain."
        payload = {
            "screen_state": screen_state,
            "motion_intensity": motion,
            "brightness": brightness,
            "brightness_delta": brightness_delta,
            "combat": combat,
            "menu_open": menu_open,
            "inventory_visible": inventory_visible,
            "dark_scene": dark_scene,
            "major_change": major_change,
            "game_ui_detected": game_ui,
            "semantic": l2,
            "layer1_reflex": l1,
            "layer2_semantic": l2,
        }

        return {
            "event_id": f"vision_{int(time.time() * 1000)}",
            "source": "vision",
            "source_family": "vision",
            "event_type": event_type,
            "speaker": "screen",
            "source_actor_id": "screen",
            "text": narrative,
            "message": narrative,
            "addressed_to_nan0": False,
            "priority": 5,
            "priority_label": "vision_or_external",
            "confidence": confidence,
            "timestamp": time.time(),
            "stale_after_seconds": 8.0 if event_type in {"vision_motion", "vision_dark", "vision_major_change", "vision_combat"} else 20.0,
            "screen_state": screen_state,
            "motion_intensity": motion,
            "brightness": brightness,
            "brightness_delta": brightness_delta,
            "combat": combat,
            "menu_open": menu_open,
            "inventory_visible": inventory_visible,
            "dark_scene": dark_scene,
            "major_change": major_change,
            "game_ui_detected": game_ui,
            "semantic": l2,
            "payload": payload,
        }


    def _vision_event_is_llm_worthy(self, l1: Dict[str, Any], now: float) -> bool:
        """Only meaningful vision changes are allowed to call the frozen thought engine."""
        if not getattr(self, "vision_llm_enabled", True):
            return False

        if getattr(self, "_shutdown_requested", False):
            return False

        if now - self.last_thought_engine_call_at < self.thought_engine_cooldown_seconds:
            return False

        menu_open = bool(l1.get("menu_open", False))
        screen_state = str(l1.get("screen_state", "unknown"))
        brightness_delta = float(l1.get("brightness_delta", 0) or 0)

        menu_changed = self._last_menu_open is not None and menu_open != self._last_menu_open
        screen_changed = self._last_screen_state is not None and screen_state != self._last_screen_state

        self._last_menu_open = menu_open
        self._last_screen_state = screen_state

        if bool(l1.get("major_change", False)):
            return True
        if bool(l1.get("combat", False)):
            return True
        if menu_changed:
            return True
        if abs(brightness_delta) > 30:
            return True
        if screen_changed and screen_state in {"very_dark", "major_change"}:
            return True

        return False

    def _call_thought_engine(
        self,
        event: Dict[str, Any],
        vision_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if getattr(self, "_shutdown_requested", False):
            return None
        if not getattr(self, "vision_llm_enabled", True):
            return None

        try:
            return generate_inner_thought_packet(event, vision_context=vision_context)
        except RuntimeError as exc:
            if not getattr(self, "_shutdown_requested", False):
                logger.warning(f"Vision layer3 thought engine unavailable: {exc}. Vision will stay silent.")
            return None
        except Exception as exc:
            if not getattr(self, "_shutdown_requested", False):
                logger.warning(f"Vision layer3 thought engine failed: {exc}. Vision will stay silent.")
            return None

    def _layer3(self, l1: Dict[str, Any], l2: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
        """Generate Nan0's interpretation with throttled, timeout-safe thought engine access."""
        if getattr(self, "_shutdown_requested", False):
            return self._layer3_silent(l1, l2)

        now = now or time.time()

        if not self._vision_event_is_llm_worthy(l1, now):
            return self._layer3_silent(l1, l2)

        self.last_thought_engine_call_at = now
        event = self._build_vision_event(l1, l2)
        vision_context = {
            "screen_state": l1.get("screen_state"),
            "motion_intensity": l1.get("motion_intensity"),
            "brightness": l1.get("brightness"),
            "brightness_delta": l1.get("brightness_delta"),
            "combat": l1.get("combat"),
            "menu_open": l1.get("menu_open"),
            "inventory_visible": l1.get("inventory_visible"),
            "dark_scene": l1.get("dark_scene"),
            "major_change": l1.get("major_change"),
            "game_ui_detected": l1.get("game_ui_detected"),
            "layer1_reflex": l1,
            "layer2_semantic": l2,
        }
        packet = self._call_thought_engine(event, vision_context=vision_context)

        if packet and packet.get("private_text"):
            suppression = packet.get("suppression_reason")
            speakability = float(packet.get("speakability") or 0)

            if suppression == "low_information_thought" and len(event.get("text", "")) > 50:
                packet["suppression_reason"] = None
                packet["speakability"] = max(speakability, 0.4)
                suppression = None
                speakability = float(packet.get("speakability") or 0)

            return {
                "perceived_threat": packet.get("private_text", "screen changed"),
                "emotional_stakes": self._stakes_from_mood(packet.get("mood", "muttering")),
                "ego_position": "superior_tiny_machine_observer",
                "mood": packet.get("mood", "muttering"),
                "thought_seed": packet.get("private_text", ""),
                "thoughtworthy": speakability >= 0.35 and suppression is None,
                "speak_reason": "real_visual_change" if speakability >= 0.35 else "none",
                "do_not_speak_if_only_silence": True,
                "thought_packet": packet,
            }

        return self._layer3_silent(l1, l2)

    def _stakes_from_mood(self, mood: str) -> str:
        mapping = {
            "gremlin_rage": "high",
            "suspicion": "suspicious",
            "offended": "medium",
            "smug": "medium",
            "possessive": "medium",
            "muttering": "low",
            "boredom": "low",
            "normal": "low",
        }
        return mapping.get(mood, "low")

    def _layer3_silent(self, l1: Dict[str, Any], l2: Dict[str, Any]) -> Dict[str, Any]:
        """Body/state-only fallback. Never creates scripted Nan0 speech or private thoughts."""
        perceived = "vision_state_only"
        if l1.get("combat"):
            perceived = "combat_visible"
        elif l1.get("dark_scene"):
            perceived = "dark_scene_visible"
        elif l1.get("menu_open") or l1.get("inventory_visible"):
            perceived = "menu_visible"
        elif l1.get("major_change"):
            perceived = "major_visual_change"
        elif l1.get("game_ui_detected") == "coding":
            perceived = "code_visible"

        return {
            "perceived_threat": perceived,
            "emotional_stakes": "low",
            "ego_position": "observer_only",
            "mood": "muttering",
            "thought_seed": "",
            "thoughtworthy": False,
            "speak_reason": "no_inner_thought_packet",
            "do_not_speak_if_only_silence": True,
        }

    def _thought_packet(self, l1: Dict[str, Any], l2: Dict[str, Any], l3: Dict[str, Any], now: float) -> Dict[str, Any]:
        engine_packet = l3.get("thought_packet")
        if not (engine_packet and isinstance(engine_packet, dict)):
            return {}

        thought_id = engine_packet.get("thought_id")
        private_text = engine_packet.get("private_text") or engine_packet.get("thought_text")
        if not thought_id or not private_text:
            return {}

        packet = dict(engine_packet)
        packet["thought_id"] = thought_id
        packet["source"] = "vision_stack_v1"
        packet["private_text"] = private_text
        packet.setdefault("thought_text", private_text)
        packet.setdefault("created_at", now)
        packet.setdefault("thought_type", "vision_reaction")
        packet.setdefault("target_actor_id", engine_packet.get("target_actor_id") or "screen")
        packet.setdefault("mood", l3.get("mood", "muttering"))

        objective_observation = {
            "screen_state": l1.get("screen_state"),
            "motion_intensity": l1.get("motion_intensity"),
            "brightness": l1.get("brightness"),
            "brightness_delta": l1.get("brightness_delta"),
            "combat": l1.get("combat"),
            "menu_open": l1.get("menu_open"),
            "inventory_visible": l1.get("inventory_visible"),
            "dark_scene": l1.get("dark_scene"),
            "major_change": l1.get("major_change"),
            "game_ui_detected": l1.get("game_ui_detected"),
        }
        vision_context = {
            "objective_observation": objective_observation,
            "semantic_read": l2,
            "nan0_interpretation": l3.get("perceived_threat"),
            "layer1_reflex": l1,
            "layer2_semantic": l2,
        }

        packet["objective_observation"] = objective_observation
        packet["semantic_read"] = l2
        packet["nan0_interpretation"] = l3.get("perceived_threat")
        packet["vision_context"] = vision_context

        event_context = dict(packet.get("event_context") or {})
        event_context.update({
            "source": "vision_stack_v1",
            "source_family": "vision",
            "speaker": "screen",
            "source_actor_id": "screen",
            "text": packet.get("private_text") or packet.get("thought_text") or "",
            "addressed_to_nan0": False,
            "priority": "low",
            "vision_context": vision_context,
        })
        packet["event_context"] = event_context
        packet["source_family"] = "vision"
        return packet

    async def _emit_thought(self, thought_packet: Dict[str, Any]):
        if not (thought_packet and isinstance(thought_packet, dict) and thought_packet.get("thought_id")):
            return

        nan0_skill = None
        try:
            nan0_skill = getattr(self.context.skill_manager, "skills", {}).get("nan0")
        except Exception:
            nan0_skill = None

        if nan0_skill and getattr(nan0_skill, "is_active", False) and hasattr(nan0_skill, "handle_external_thought_packet"):
            await nan0_skill.handle_external_thought_packet(
                thought_packet,
                source_event=thought_packet.get("event_context") or {
                    "source": "vision_stack_v1",
                    "speaker": "screen",
                    "source_actor_id": "screen",
                    "text": thought_packet.get("private_text") or thought_packet.get("thought_text") or "",
                    "addressed_to_nan0": False,
                    "priority": "low",
                    "timestamp": time.time(),
                },
            )
            logger.info(f"Vision thought routed through Nan0Skill: {thought_packet.get('thought_id', 'unknown')}")
            return

        if self.context and hasattr(self.context, "event_manager"):
            self.context.event_manager.publish(
                EventCategory.THOUGHT,
                "skill:nan0_vision",
                thought_packet.get("private_text") or thought_packet.get("thought_text") or "",
                metadata={"thought_id": thought_packet.get("thought_id"), "speech_blocked": "nan0_skill_inactive"},
            )
        logger.info(f"Vision thought recorded but not spoken: {thought_packet.get('thought_id', 'unknown')}")

    def on_config_reload(self):
        self.initialize()
