import asyncio

import pytest

from src.core.config import BrainConfig
from src.modules.skills.implementations import nan0_skill as nan0_skill_module
from src.modules.skills.implementations.nan0_skill import Nan0Skill
from src.modules.skills.implementations.nan0_thought_engine_v3 import validate_inner_thought_packet


class FakeBrain:
    def __init__(self):
        self.output_calls = []
        self.last_nan0_speech_packet = None

    async def perform_output_task(self, mood, message, speech_packet=None):
        self.output_calls.append((mood, message, speech_packet))


def _make_skill():
    config = BrainConfig()
    config.skills["nan0"]["enabled"] = True
    brain = FakeBrain()
    skill = Nan0Skill("nan0", config, brain)
    skill.is_active = True
    skill.last_spoken_at = 0.0
    return skill, brain


def _boot_packet(private_text="Kyo's machine room is awake again. Mine."):
    return {
        "thought_id": "thought_boot_1",
        "event_id": "boot_event_1",
        "source": "boot",
        "target_actor_id": "nan0",
        "thought_type": "quiet_presence",
        "private_text": private_text,
        "thought_text": private_text,
        "mood": "suspicion",
        "pressure": 1.0,
        "novelty": 0.8,
        "speakability": 1.0,
        "relationship_charge": 0.4,
        "ego_charge": 0.8,
        "vision_charge": 0.0,
        "memory_write_candidate": False,
        "suppression_reason": None,
    }


def test_boot_speech_uses_thought_id_normal_routing_and_output(monkeypatch):
    skill, brain = _make_skill()
    order = []
    recorded_speech_packets = []
    packet = _boot_packet()

    async def fake_create(event):
        order.append("thought")
        assert event["source"] == "boot"
        assert event["source_actor_id"] == "nan0"
        return packet

    def fake_route(candidate):
        order.append("route")
        assert candidate is packet
        return {
            "decision": "speak",
            "reason": "routed_from_thought",
            "thought_id": packet["thought_id"],
            "source_thought_id": packet["thought_id"],
            "line": "",
        }

    original_generate = skill._generate_line
    original_speak_decision = skill._speak_decision

    async def tracked_generate(candidate):
        order.append("speech_generation")
        assert candidate is packet
        return await original_generate(candidate)

    async def tracked_speak_decision(decision, reason="unknown"):
        order.append("output")
        assert decision["thought_id"] == packet["thought_id"]
        return await original_speak_decision(decision, reason=reason)

    monkeypatch.setattr(skill, "_create_inner_thought", fake_create)
    monkeypatch.setattr(skill, "_generate_line", tracked_generate)
    monkeypatch.setattr(skill, "_speak_decision", tracked_speak_decision)
    monkeypatch.setattr(skill, "_record_speech_debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(nan0_skill_module, "route_thought", fake_route)
    monkeypatch.setattr(nan0_skill_module, "record_speech_packet", recorded_speech_packets.append)

    decision = asyncio.run(skill._run_boot_presence())

    assert order == ["thought", "route", "speech_generation", "output"]
    assert decision["decision"] == "speak"
    assert decision["thought_id"] == packet["thought_id"]
    assert brain.output_calls
    assert brain.output_calls[0][2]["thought_id"] == packet["thought_id"]
    assert brain.last_nan0_speech_packet["thought_id"] == packet["thought_id"]
    assert recorded_speech_packets[0]["thought_id"] == packet["thought_id"]


@pytest.mark.parametrize(
    ("packet", "reason"),
    [
        (None, "missing_thought_packet"),
        ({"source": "boot", "private_text": "awake"}, "missing_thought_id"),
        ({"source": "boot", "thought_id": "boot_1", "private_text": "awake"}, "invalid_thought_id"),
        ({"source": "boot", "thought_id": "thought_boot_empty", "private_text": ""}, "missing_private_text"),
    ],
)
def test_boot_suppresses_safely_without_populated_private_thought(monkeypatch, packet, reason):
    skill, brain = _make_skill()

    async def fake_create(event):
        return packet

    def forbidden_route(candidate):
        raise AssertionError("boot routed an invalid thought packet")

    monkeypatch.setattr(skill, "_create_inner_thought", fake_create)
    monkeypatch.setattr(nan0_skill_module, "route_thought", forbidden_route)

    decision = asyncio.run(skill._run_boot_presence())

    assert decision is None
    assert brain.output_calls == []
    assert validate_inner_thought_packet(packet, expected_source="boot") == (False, reason)


def test_boot_obeys_normal_router_suppression(monkeypatch):
    skill, brain = _make_skill()
    packet = _boot_packet()
    calls = []

    async def fake_create(event):
        calls.append("thought")
        return packet

    def fake_route(candidate):
        calls.append("route")
        return {
            "decision": "suppress",
            "reason": "boot_test_suppression",
            "thought_id": packet["thought_id"],
            "source_thought_id": packet["thought_id"],
            "line": "",
        }

    async def forbidden_generate(candidate):
        raise AssertionError("speech generation ran after router suppression")

    monkeypatch.setattr(skill, "_create_inner_thought", fake_create)
    monkeypatch.setattr(skill, "_generate_line", forbidden_generate)
    monkeypatch.setattr(skill, "_record_speech_debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(nan0_skill_module, "route_thought", fake_route)

    decision = asyncio.run(skill._run_boot_presence())

    assert calls == ["thought", "route"]
    assert decision["decision"] == "suppress"
    assert decision["thought_id"] == packet["thought_id"]
    assert brain.output_calls == []


def test_boot_has_no_direct_prompt_to_speech_fallback(monkeypatch):
    skill, brain = _make_skill()
    calls = []

    async def failed_thought_generation(event):
        calls.append("thought_failed")
        raise RuntimeError("model unavailable")

    def forbidden_route(candidate):
        raise AssertionError("router received no thought packet")

    async def forbidden_speech(decision, reason="unknown"):
        raise AssertionError("boot used a direct speech fallback")

    monkeypatch.setattr(skill, "_create_inner_thought", failed_thought_generation)
    monkeypatch.setattr(skill, "_speak_decision", forbidden_speech)
    monkeypatch.setattr(nan0_skill_module, "route_thought", forbidden_route)

    decision = asyncio.run(skill._run_boot_presence())

    assert decision is None
    assert calls == ["thought_failed"]
    assert brain.output_calls == []
