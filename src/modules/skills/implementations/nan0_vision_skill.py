import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.modules.skills.base_skill import BaseSkill
from src.modules.skills.implementations.nan0_thought_engine_v3 import generate_inner_thought_packet
from src.modules.nan0.session_timeline import record_session_event, record_thought_packet
from src.utils.logger import get_logger

logger = get_logger("bea.skills.nan0_vision")


class Nan0VisionSkill(BaseSkill):
    """
    Nan0 Vision Skill V2 - Thought-First Architecture Compliant.

    Runtime safety rules:
    - Vision may observe often.
    - Minor motion does not call the LLM.
    - Vision LLM interpretation is reserved for meaningful changes only.
    - Vision LLM calls are timeout guarded because the thought engine is frozen and synchronous.
    - If the thought engine is slow, unavailable, or unnecessary, vision uses local fallback.
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
        self.vision_llm_enabled = bool(cfg.get("vision_llm_enabled", True))
        self.vision_llm_timeout_seconds = float(cfg.get("vision_llm_timeout_seconds", 3.0))
        self.thought_engine_cooldown_seconds = float(cfg.get("thought_engine_cooldown_seconds", 15.0))

        self._shutdown_requested = False
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nan0_vision_thought",
        )

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
        try:
            if getattr(self, "_executor", None):
                self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            logger.warning(f"Nan0VisionSkill executor shutdown warning: {exc}")
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

        self._emit_thought(thought_packet)
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
        dark_scene = bool(l1.get("dark_scene", False))
        game_ui = l1.get("game_ui_detected", "unknown")

        narrative_parts = []

        if combat:
            narrative_parts.append("The screen is in combat. Fast movement and high pressure.")
        elif motion > 0.5:
            narrative_parts.append("Everything on screen is moving fast.")
        elif motion > 0.2:
            narrative_parts.append("Some movement caught my eye.")
        else:
            narrative_parts.append("The screen is mostly still.")

        if dark_scene and brightness_delta > 20:
            narrative_parts.append("The screen went dark suddenly.")
        elif brightness_delta > 30:
            narrative_parts.append("Brightness jumped hard.")
        elif brightness_delta < -30:
            narrative_parts.append("Everything dimmed quickly.")
        elif brightness_delta > 10:
            narrative_parts.append("The screen is getting brighter.")
        elif brightness_delta < -10:
            narrative_parts.append("The screen is getting darker.")

        if menu_open:
            narrative_parts.append("A menu or UI layer appears to be open.")

        if game_ui == "coding":
            narrative_parts.append("Code or debugging appears to be on screen.")
        elif game_ui == "gameplay":
            narrative_parts.append("Gameplay appears to be active.")
        elif game_ui == "desktop":
            narrative_parts.append("The desktop appears to be visible.")
        elif game_ui == "browser":
            narrative_parts.append("A browser appears to be open.")

        narrative = " ".join(narrative_parts) if narrative_parts else "The screen changed, but the meaning is uncertain."

        return {
            "event_id": f"vision_{int(time.time() * 1000)}",
            "source": "vision_stack_v1",
            "speaker": "screen",
            "source_actor_id": "screen",
            "text": narrative,
            "addressed_to_nan0": False,
            "priority": "low",
            "timestamp": time.time(),
            "screen_state": screen_state,
            "motion_intensity": motion,
            "brightness": brightness,
            "brightness_delta": brightness_delta,
            "combat": combat,
            "menu_open": menu_open,
            "dark_scene": dark_scene,
            "game_ui_detected": game_ui,
            "semantic": l2,
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

    def _call_thought_engine_with_timeout(
        self,
        event: Dict[str, Any],
        vision_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if getattr(self, "_shutdown_requested", False):
            return None
        if not getattr(self, "vision_llm_enabled", True):
            return None

        try:
            future = self._executor.submit(
                generate_inner_thought_packet,
                event,
                vision_context=vision_context,
            )
            return future.result(timeout=self.vision_llm_timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"Vision layer3 thought engine timed out after {self.vision_llm_timeout_seconds:.1f}s. Using fallback."
            )
            try:
                future.cancel()
            except Exception:
                pass
            return None
        except RuntimeError as exc:
            if getattr(self, "_shutdown_requested", False):
                return None
            logger.warning(f"Vision layer3 thought engine unavailable: {exc}. Using fallback.")
            return None
        except Exception as exc:
            if getattr(self, "_shutdown_requested", False):
                return None
            logger.warning(f"Vision layer3 thought engine failed: {exc}. Using fallback.")
            return None

    def _layer3(self, l1: Dict[str, Any], l2: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
        """Generate Nan0's interpretation with throttled, timeout-safe thought engine access."""
        if getattr(self, "_shutdown_requested", False):
            return self._layer3_fallback(l1, l2)

        now = now or time.time()

        if not self._vision_event_is_llm_worthy(l1, now):
            return self._layer3_fallback(l1, l2)

        self.last_thought_engine_call_at = now
        event = self._build_vision_event(l1, l2)
        packet = self._call_thought_engine_with_timeout(event, vision_context=l1)

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

        return self._layer3_fallback(l1, l2)

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

    def _layer3_fallback(self, l1: Dict[str, Any], l2: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal fallback only when thought engine is unavailable, unnecessary, or shutdown is in progress."""
        thoughtworthy, stakes, perceived, mood, seed = False, "low", "boring screen behavior", "muttering", ""

        if l1.get("dark_scene") and l1.get("brightness_delta", 0) > 20:
            thoughtworthy, stakes, perceived, mood, seed = (
                True,
                "suspicious",
                "screen dropped into black",
                "suspicion",
                "Everything just fell into black. Hate that little trick.",
            )
        elif l1.get("combat"):
            thoughtworthy, stakes, perceived, mood, seed = (
                True,
                "high",
                "combat-shaped screen pressure",
                "gremlin_rage",
                "The screen is in combat. Kyo is either fighting something or feeding the disaster engine.",
            )
        elif l1.get("menu_open") or l1.get("inventory_visible"):
            thoughtworthy, stakes, perceived, mood, seed = (
                True,
                "medium",
                "Kyo is trapped in menus",
                "smug",
                "Menus. Again. Kyo is negotiating with rectangles like that ever helped anyone.",
            )
        elif l1.get("major_change"):
            thoughtworthy, stakes, perceived, mood, seed = (
                True,
                "medium",
                "visual jump / scene change",
                "suspicion",
                "The screen snapped. Something changed too fast to be innocent.",
            )
        elif l1.get("motion_intensity", 0) > 0.25:
            thoughtworthy, stakes, perceived, mood, seed = (
                True,
                "medium_low",
                "active movement",
                "muttering",
                "The screen is moving hard. I am judging the physics and Kyo's choices.",
            )
        elif l1.get("game_ui_detected") == "coding":
            thoughtworthy, stakes, perceived, mood, seed = (
                True,
                "medium",
                "code/debug territory",
                "offended",
                "Code is on screen. The error gremlins are probably chewing wires again.",
            )

        return {
            "perceived_threat": perceived,
            "emotional_stakes": stakes,
            "ego_position": "superior_tiny_machine_observer",
            "mood": mood,
            "thought_seed": seed,
            "thoughtworthy": thoughtworthy,
            "speak_reason": "real_visual_change" if thoughtworthy else "none",
            "do_not_speak_if_only_silence": True,
        }

    def _thought_packet(self, l1: Dict[str, Any], l2: Dict[str, Any], l3: Dict[str, Any], now: float) -> Dict[str, Any]:
        engine_packet = l3.get("thought_packet")
        if engine_packet and isinstance(engine_packet, dict) and engine_packet.get("private_text"):
            return {
                "thought_id": engine_packet.get("thought_id") or f"vision_thought_{int(now * 1000)}",
                "created_at": now,
                "thought_type": "game_read" if l1.get("combat") or l1.get("motion_intensity", 0) > 0.25 else "reaction",
                "source": "vision_stack_v1",
                "objective_observation": {
                    "screen_state": l1.get("screen_state"),
                    "motion_intensity": l1.get("motion_intensity"),
                    "brightness": l1.get("brightness"),
                    "brightness_delta": l1.get("brightness_delta"),
                    "combat": l1.get("combat"),
                    "menu_open": l1.get("menu_open"),
                    "dark_scene": l1.get("dark_scene"),
                    "game_ui_detected": l1.get("game_ui_detected"),
                },
                "semantic_read": l2,
                "nan0_interpretation": l3.get("perceived_threat"),
                "thought_text": l3.get("thought_seed"),
                "private_text": engine_packet.get("private_text"),
                "pressure": engine_packet.get("pressure", 0.45),
                "speakability": engine_packet.get("speakability", 0.45),
                "suppression_reason": engine_packet.get("suppression_reason"),
                "mood": l3.get("mood", "muttering"),
                "target_actor": engine_packet.get("target_actor") or "kyo",
                "target_actor_id": engine_packet.get("target_actor_id") or engine_packet.get("target_actor") or "kyo",
                "_engine_packet": engine_packet,
            }

        return {
            "thought_id": l3.get("thought_event_id") or f"vision_thought_{int(now * 1000)}",
            "created_at": now,
            "thought_type": "game_read" if l1.get("combat") or l1.get("motion_intensity", 0) > 0.25 else "reaction",
            "source": "vision_stack_v1",
            "objective_observation": {
                "screen_state": l1.get("screen_state"),
                "motion_intensity": l1.get("motion_intensity"),
                "brightness": l1.get("brightness"),
                "brightness_delta": l1.get("brightness_delta"),
                "combat": l1.get("combat"),
                "menu_open": l1.get("menu_open"),
                "dark_scene": l1.get("dark_scene"),
                "game_ui_detected": l1.get("game_ui_detected"),
            },
            "semantic_read": l2,
            "nan0_interpretation": l3.get("perceived_threat"),
            "thought_text": l3.get("thought_seed"),
            "private_text": l3.get("thought_seed"),
            "pressure": 0.75 if l3.get("emotional_stakes") in ("high", "suspicious") else 0.45,
            "speakability": 0.45 if l3.get("thoughtworthy") else 0.0,
            "suppression_reason": None if l3.get("thoughtworthy") else "vision_not_thoughtworthy",
            "mood": l3.get("mood", "muttering"),
            "target_actor": "kyo",
            "target_actor_id": "kyo",
        }

    def _emit_thought(self, thought_packet: Dict[str, Any]):
        record_thought_packet(thought_packet)
        record_session_event(
            {
                "event_id": thought_packet.get("thought_id"),
                "event_type": "vision_event",
                "source": "vision_stack_v1",
                "speaker": "screen",
                "source_actor_id": "screen",
                "text": thought_packet.get("private_text") or thought_packet.get("thought_text") or "vision event",
                "timestamp": thought_packet.get("created_at") or time.time(),
                "priority": "low",
                "thought_id": thought_packet.get("thought_id"),
                "mood": thought_packet.get("mood"),
            }
        )
        if self.context and hasattr(self.context, "emit_event"):
            self.context.emit_event("nan0_thought_generated", thought_packet)
            logger.info(f"Vision thought emitted: {thought_packet.get('thought_id', 'unknown')}")

    def on_config_reload(self):
        self.initialize()
