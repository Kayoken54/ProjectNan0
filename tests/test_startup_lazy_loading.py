import asyncio
import sys
import types
import logging

rich_module = types.ModuleType("rich")
rich_logging_module = types.ModuleType("rich.logging")

class RichHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def emit(self, record):
        return None

rich_logging_module.RichHandler = RichHandler
sys.modules.setdefault("rich", rich_module)
sys.modules.setdefault("rich.logging", rich_logging_module)

from src.core.config import BrainConfig
from src.modules.skills.memory.memory_skill import MemorySkill
from src.modules.skills.skill_manager import SkillManager
from src.modules.skills.implementations.minecraft_skill import MinecraftSkill


class FakeStorage:
    initialized = False
    queried = False

    def __init__(self, **kwargs):
        self.collection = None

    def initialize(self):
        type(self).initialized = True
        self.collection = object()
        return True

    def query_similar(self, query, limit=3):
        type(self).queried = True
        return None

    def entry_exists(self, entry_id):
        return False


class FakeHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, **kwargs):
        self.messages.append(kwargs)


class FakeBrain:
    def __init__(self):
        self.llm = object()
        self.is_speaking = False
        self.history_manager = FakeHistory()
        self.output_calls = []

    async def perform_output_task(self, mood, message):
        self.output_calls.append((mood, message))


def _reset_memory_skill():
    MemorySkill._shared_storage = None
    MemorySkill._storage_initialized = False
    MemorySkill._storage_class = None
    MemorySkill._generator_class = None
    FakeStorage.initialized = False
    FakeStorage.queried = False


def test_memory_skill_initialize_does_not_load_storage(monkeypatch):
    _reset_memory_skill()
    monkeypatch.setattr(MemorySkill, "_storage_class", FakeStorage)
    config = BrainConfig()
    config.skills["memory"]["enabled"] = True

    skill = MemorySkill("memory", config, FakeBrain())
    skill.initialize()

    assert FakeStorage.initialized is False
    assert MemorySkill._storage_initialized is False


def test_memory_storage_initializes_only_on_first_retrieve(monkeypatch):
    _reset_memory_skill()
    monkeypatch.setattr(MemorySkill, "_storage_class", FakeStorage)
    config = BrainConfig()
    config.skills["memory"]["enabled"] = True

    skill = MemorySkill("memory", config, FakeBrain())
    skill.initialize()
    assert FakeStorage.initialized is False

    assert skill.retrieve_context("Kyo did the thing") == ""
    assert FakeStorage.initialized is True
    assert FakeStorage.queried is True


def test_skill_manager_import_does_not_eager_load_heavy_skills():
    assert "src.modules.skills.minecraft.mc_agent.core.agent" not in sys.modules
    assert "chromadb" not in sys.modules

    config = BrainConfig()
    config.skills["minecraft"]["enabled"] = False
    config.skills["memory"]["enabled"] = False
    manager = SkillManager(config, object())

    assert "src.modules.skills.minecraft.mc_agent.core.agent" not in sys.modules
    assert "chromadb" not in sys.modules
    assert isinstance(manager._skill_classes["minecraft"], str)
    assert isinstance(manager._skill_classes["memory"], str)


def test_minecraft_auto_speech_blocks_without_thought_id():
    config = BrainConfig()
    config.skills["minecraft"]["enabled"] = True
    brain = FakeBrain()
    skill = MinecraftSkill("minecraft", config, brain)
    skill.pending_speech = 1

    asyncio.run(skill._speak_thought("raw minecraft agent thought"))

    assert brain.output_calls == []
    assert brain.history_manager.messages
    assert brain.history_manager.messages[-1]["metadata"]["reason"] == "missing_thought_id"
