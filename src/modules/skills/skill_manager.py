import asyncio
from typing import Dict, Type

from src.core.config import BrainConfig
from src.modules.skills.base_skill import BaseSkill
from src.modules.skills.implementations.monologue import MonologueSkill
from src.modules.skills.implementations.minecraft_skill import MinecraftSkill
from src.modules.skills.memory.memory_skill import MemorySkill
from src.modules.skills.discord.discord_skill import DiscordSkill
from src.modules.skills.implementations.nan0_skill import Nan0Skill
from src.modules.skills.implementations.nan0_vision_skill import Nan0VisionSkill
from src.utils.logger import get_logger

logger = get_logger("bea.skills.manager")


class SkillManager:
    def __init__(self, config: BrainConfig, brain_context):
        self.config = config
        self.context = brain_context
        self.skills: Dict[str, BaseSkill] = {}
        self.running = False
        self._loop_task = None
        self._skill_classes: Dict[str, Type[BaseSkill]] = {
            "monologue": MonologueSkill,
            "minecraft": MinecraftSkill,
            "memory": MemorySkill,
            "discord": DiscordSkill,
            "nan0_vision": Nan0VisionSkill,
            "nan0": Nan0Skill,
        }

    def log(self, skill_name: str, message: str):
        if hasattr(self.context, "event_manager"):
            from src.core.events import EventCategory

            category = EventCategory.SKILL
            if message.startswith("Thought:"):
                category = EventCategory.THOUGHT
                message = message.replace("Thought:", "").strip()
            self.context.event_manager.publish(category, f"skill:{skill_name}", message)
        else:
            logger.info(f"[{skill_name}] {message}")

    def initialize(self):
        logger.info("Initializing Skill Manager...")

        for name, skill_cls in self._skill_classes.items():
            self._register_skill(name, skill_cls)

        nan0 = self.skills.get("nan0")
        vision = self.skills.get("nan0_vision")
        if nan0 and vision and hasattr(nan0, "set_vision_skill"):
            nan0.set_vision_skill(vision)

        for name, skill in self.skills.items():
            skill.initialize()
            if skill.enabled:
                logger.info(f"Skill '{name}' is enabled in config.")
            else:
                logger.info(f"Skill '{name}' is disabled in config.")

    def _skill_enabled_in_config(self, name: str) -> bool:
        skill_config = {}
        try:
            skill_config = self.config.skills.get(name, {}) or {}
        except Exception:
            skill_config = {}

        return bool(skill_config.get("enabled", True))

    def _register_skill(self, name: str, skill_cls: Type[BaseSkill]):
        if not self._skill_enabled_in_config(name):
            logger.info(f"Skipping skill '{name}' because enabled=false in config.")
            return

        if name in self.skills:
            logger.warning(f"Skill '{name}' is already registered. Skipping duplicate registration.")
            return

        self.skills[name] = skill_cls(name, self.config, self.context)

    async def start(self):
        if self.running:
            logger.warning("Skill Manager start requested while already running. Ignoring duplicate start.")
            return

        self.running = True

        for skill in self.skills.values():
            if skill.enabled and not skill.is_active:
                await skill.start()

        self._loop_task = asyncio.create_task(self._main_loop())
        logger.info("Skill Manager started.")

    async def stop(self):
        self.running = False

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            finally:
                self._loop_task = None

        for skill in self.skills.values():
            if skill.is_active:
                await skill.stop()

        logger.info("Skill Manager stopped.")

    async def _main_loop(self):
        logger.info("Skill Manager loop active.")

        while self.running:
            for name, skill in list(self.skills.items()):
                if not skill.enabled:
                    if skill.is_active:
                        await skill.stop()
                    continue

                if not skill.is_active:
                    logger.warning(
                        f"Skill '{name}' is enabled but inactive. Not auto-restarting in main loop."
                    )
                    continue

                try:
                    await skill.update()
                except Exception as e:
                    logger.error(f"Error in skill '{skill.name}': {e}")

            await asyncio.sleep(1)

    def toggle_skill(self, name: str, state: bool):
        if name not in self._skill_classes and name not in self.skills:
            return False

        self.config.skills.setdefault(name, {})["enabled"] = state
        self.config.save_to_file()

        if state and name not in self.skills:
            self._register_skill(name, self._skill_classes[name])
            skill = self.skills.get(name)
            if skill:
                skill.initialize()

        return True

    def reload_config(self):
        for name, skill in list(self.skills.items()):
            try:
                if hasattr(skill, "on_config_reload"):
                    skill.on_config_reload()
            except Exception as e:
                logger.error(f"Error checking reload for skill '{name}': {e}")