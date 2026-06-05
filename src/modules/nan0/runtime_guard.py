"""
Nan0 V7 Runtime Guard - resource-aware survival layer.
Designed for KayoPC / i5-10600K / 32GB RAM / GTX 1660 Ti 6GB.

Purpose:
- Keep Nan0 responsive during game sessions.
- Never let deep vision or background tasks block the live lane.
- Track service health without crashing the runtime.
- Provide simple degraded-state lines when local inference stalls.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class RuntimeGuardState:
    mode: str = "survival"
    live_lane_ok: bool = True
    deep_lane_ok: bool = False
    obs_ok: bool = False
    memory_ok: bool = True
    vision_fast_ok: bool = True
    vision_deep_ok: bool = False
    last_live_error: str = ""
    last_deep_error: str = ""
    last_deep_vision_at: float = 0.0
    last_live_timeout_at: float = 0.0
    live_timeout_count: int = 0
    deep_timeout_count: int = 0
    last_update: float = 0.0


class RuntimeGuard:
    def __init__(self, state_path: str = "data/nan0/runtime_guard_state.json"):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = RuntimeGuardState(last_update=time.time())
        self._load()

    def _load(self) -> None:
        try:
            if self.state_path.exists():
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                for key, value in raw.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
        except Exception:
            pass

    def save(self) -> None:
        self.state.last_update = time.time()
        try:
            self.state_path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
        except Exception:
            pass

    def record_live_success(self) -> None:
        self.state.live_lane_ok = True
        self.state.live_timeout_count = 0
        self.state.last_live_error = ""
        self.save()

    def record_live_failure(self, error: str) -> None:
        self.state.live_lane_ok = False
        self.state.live_timeout_count += 1
        self.state.last_live_timeout_at = time.time()
        self.state.last_live_error = str(error)[0:300]
        self.save()

    def record_deep_success(self) -> None:
        self.state.deep_lane_ok = True
        self.state.vision_deep_ok = True
        self.state.deep_timeout_count = 0
        self.state.last_deep_error = ""
        self.state.last_deep_vision_at = time.time()
        self.save()

    def record_deep_failure(self, error: str) -> None:
        self.state.deep_lane_ok = False
        self.state.vision_deep_ok = False
        self.state.deep_timeout_count += 1
        self.state.last_deep_error = str(error)[0:300]
        self.save()

    def should_allow_deep_vision(self, reason: str = "") -> bool:
        """Event-driven deep vision gate.
        Allows deep vision only when the live lane is healthy and cooldown passed.
        """
        now = time.time()
        if not self.state.live_lane_ok and self.state.live_timeout_count >= 1:
            return False
        if self.state.deep_timeout_count >= 2:
            # Back off hard after repeated deep lane failure.
            return (now - self.state.last_deep_vision_at) > 600
        cooldown = 180
        if reason in {"user_requested", "manual", "death", "game_over"}:
            cooldown = 30
        elif reason in {"dark_screen_check", "screen_state_changed"}:
            cooldown = 240
        return (now - self.state.last_deep_vision_at) >= cooldown

    def degraded_line(self, context: str = "") -> str:
        """In-character line when the local brain stalls. Short by design."""
        if self.state.live_timeout_count >= 3:
            return "My local brain is choking. I am still here, unfortunately for the hardware."
        if self.state.deep_timeout_count >= 2:
            return "My deep eye is being dramatic. I can still watch the room twitch."
        if context:
            return f"I caught the glitch. {context} is not getting away with that."
        return "Tiny brain spark failed. Still alive. Annoying, but alive."


_guard: Optional[RuntimeGuard] = None


def get_runtime_guard() -> RuntimeGuard:
    global _guard
    if _guard is None:
        _guard = RuntimeGuard()
    return _guard
