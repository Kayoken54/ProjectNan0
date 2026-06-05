"""Nan0 thought-origin monologue gate.

DISABLED: This skill bypasses the thought-first architecture.
All monologue behavior is handled by Nan0Skill._presence_loop().
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.modules.skills.base_skill import BaseSkill
from src.utils.logger import get_logger

logger = get_logger("bea.skills.monologue")


class MonologueSkill(BaseSkill):
    def initialize(self):
        logger.info("Nan0 monologue gate: DISABLED. Monologue skill bypasses thought-first architecture.")
        cfg = self.skill_config
        self.min_gap_seconds = float(cfg.get("min_gap_seconds", 45))
        self.vision_state_path = Path(cfg.get("vision_state_path", "data/vision/nan0_vision_stack_state.json"))
        self.last_speech_time = 0.0
        self.last_thought_id = ""
        self.recent_lines: List[str] = []
        self.banned = [
            "",
            "",
            "same suspicious reality",
            "tragically useful",
            "assistant sludge",
            "real room only",
            "nothing dramatic",
            "",
            "i have opinions",
            "i already judged",
            "become furniture",
        ]

    async def update(self):
        # DISABLED: Monologue skill bypasses thought-first architecture.
        # All monologue behavior is handled by Nan0Skill._presence_loop().
        return

    def _read_state(self) -> Dict[str, Any]:
        try:
            if not self.vision_state_path.exists():
                return {}
            return json.loads(self.vision_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _valid(self, line: str) -> bool:
        low = re.sub(r"[^a-z0-9]+", " ", (line or "").lower()).strip()
        if len(low.split()) < 5:
            return False
        if any(x in low for x in self.banned):
            return False
        if low in self.recent_lines:
            return False
        return True

    def _remember(self, line: str):
        low = re.sub(r"[^a-z0-9]+", " ", (line or "").lower()).strip()
        self.recent_lines.append(low)
        self.recent_lines = self.recent_lines[-18:]

    def on_config_reload(self):
        self.min_gap_seconds = float(self.skill_config.get("min_gap_seconds", self.min_gap_seconds))