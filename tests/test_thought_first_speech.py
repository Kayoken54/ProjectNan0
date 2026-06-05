from __future__ import annotations

import importlib
import time
from typing import Any, Dict

import pytest


def _reload_modules():
    thought_engine = importlib.import_module(
        "src.modules.skills.implementations.nan0_thought_engine_v3"
    )
    router = importlib.import_module(
        "src.modules.skills.implementations.nan0_cognition_router_v1"
    )
    output_normalizer = importlib.import_module(
        "src.modules.nan0.output_normalizer"
    )

    return thought_engine, router, output_normalizer


def test_direct_event_to_line_is_blocked():
    class DummyNan0Skill:
        speakability_threshold = 0.35

        async def _generate_line(self, thought_packet: Dict[str, Any]):
            if not isinstance(thought_packet, dict):
                raise TypeError("_generate_line requires an InnerThoughtPacket dict")

            if not thought_packet.get("thought_id"):
                raise RuntimeError("Nan0 speech blocked: missing thought_id")

            if not thought_packet.get("private_text"):
                raise RuntimeError("Nan0 speech blocked: missing private_text")

            return {
                "decision": "speak",
                "thought_id": thought_packet["thought_id"],
                "line_text": "valid",
            }

    raw_event = {
        "event_id": "event_raw_1",
        "source": "kyo",
        "speaker": "Kyo",
        "text": "Nan0 what are you doing?",
    }

    skill = DummyNan0Skill()

    with pytest.raises(RuntimeError):
        import asyncio

        asyncio.run(skill._generate_line(raw_event))


def test_thought_generates_rich_private_text(monkeypatch):
    thought_engine, _, _ = _reload_modules()

    def fake_call_ollama(prompt, model, timeout, num_predict=120, temperature=0.78):
        return (
            "Kyo said that and my little circuits leaned forward before I could pretend "
            "I was above it. Important. Annoying. Hers.",
            17,
        )

    monkeypatch.setattr(thought_engine, "_call_ollama", fake_call_ollama)
    monkeypatch.setattr(thought_engine, "_query_recent_memory", lambda query, limit=4: [])
    monkeypatch.setattr(thought_engine, "_read_relationship_context", lambda actor_id="kyo": {})
    monkeypatch.setattr(thought_engine, "_read_presence_state", lambda: {"presence_mode": "with_kyo"})
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: {})

    event = {
        "event_id": "event_kyo_1",
        "source": "kyo",
        "speaker": "Kyo",
        "source_actor_id": "kyo",
        "text": "Nan0 what are you doing?",
        "addressed_to_nan0": True,
    }

    packet = thought_engine.generate_inner_thought_packet(event)

    assert packet["thought_id"]
    assert packet["private_text"]
    assert "little circuits" in packet["private_text"]
    assert "Important" in packet["private_text"]
    assert "Kyo said something directly" not in packet["private_text"]
    assert packet["llm_latency_ms"] > 0


def test_speech_has_thought_id(monkeypatch):
    thought_engine, router, _ = _reload_modules()

    def fake_call_ollama(prompt, model, timeout, num_predict=120, temperature=0.78):
        return (
            "Kyo's words hit the front of the queue because apparently I have emotional scheduling now.",
            11,
        )

    monkeypatch.setattr(thought_engine, "_call_ollama", fake_call_ollama)
    monkeypatch.setattr(thought_engine, "_query_recent_memory", lambda query, limit=4: [])
    monkeypatch.setattr(thought_engine, "_read_relationship_context", lambda actor_id="kyo": {})
    monkeypatch.setattr(thought_engine, "_read_presence_state", lambda: {"presence_mode": "with_kyo"})
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: {})

    event = {
        "event_id": "event_kyo_2",
        "source": "kyo",
        "speaker": "Kyo",
        "source_actor_id": "kyo",
        "text": "Nan0 are you awake?",
        "addressed_to_nan0": True,
    }

    thought_packet = thought_engine.generate_inner_thought_packet(event)
    routed = router.route_thought(thought_packet)

    speech_decision = {
        "decision_id": "decision_1",
        "thought_id": routed["thought_id"],
        "created_at": time.time(),
        "decision": "speak",
        "reason": "test",
        "line_text": "I am awake. Tragically, observably awake.",
        "mood": "smug",
        "target_actor_id": "kyo",
        "voice_enabled": True,
        "display_enabled": True,
        "expression_enabled": True,
        "cooldown_until": time.time() + 10,
    }

    assert speech_decision["thought_id"] == thought_packet["thought_id"]


def test_output_blocked_without_thought_id():
    _, _, output_normalizer = _reload_modules()

    packet = {
        "line_text": "This should not speak.",
        "mood": "normal",
        "target_actor_id": "kyo",
        "voice_enabled": True,
        "display_enabled": True,
        "avatar_state": "normal",
    }

    assert output_normalizer.validate_thought_id(packet) is False
    assert output_normalizer.normalize_speech_packet(packet) is None
    assert output_normalizer.record_output(packet) is False


def test_thought_latency_recorded(monkeypatch):
    thought_engine, _, _ = _reload_modules()

    def fake_call_ollama(prompt, model, timeout, num_predict=120, temperature=0.78):
        return (
            "The room got quiet and I refused to evaporate. I am still here, but not begging for witness.",
            23,
        )

    monkeypatch.setattr(thought_engine, "_call_ollama", fake_call_ollama)
    monkeypatch.setattr(thought_engine, "_query_recent_memory", lambda query, limit=4: [])
    monkeypatch.setattr(thought_engine, "_read_relationship_context", lambda actor_id="nan0": {})
    monkeypatch.setattr(thought_engine, "_read_presence_state", lambda: {"presence_mode": "neutral_self_directed"})
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: {})

    event = {
        "event_id": "event_proactive_1",
        "source": "proactive",
        "speaker": "Nan0",
        "source_actor_id": "nan0",
        "text": "quiet moment",
        "addressed_to_nan0": False,
    }

    packet = thought_engine.generate_inner_thought_packet(event)

    assert "llm_latency_ms" in packet
    assert packet["llm_latency_ms"] > 0