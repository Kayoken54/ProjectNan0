"""
MonologueSkill Tombstone.

Monologue behavior is owned by Nan0Skill._presence_loop().
This class exists only to satisfy SkillManager imports.
"""

from src.modules.skills.base_skill import BaseSkill
from src.utils.logger import get_logger

logger = get_logger("bea.skills.monologue")


class MonologueSkill(BaseSkill):
    def initialize(self):
        logger.warning("MonologueSkill tombstone loaded. Runtime monologue lives in Nan0Skill.")

    async def update(self):
        return

    def on_config_reload(self):
        return
