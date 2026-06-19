import asyncio
import json

import pytest

from src.core.config import BrainConfig
from src.modules.llm import ollama_provider
from src.modules.skills.implementations import nan0_skill as nan0_skill_module
from src.modules.skills.implementations import nan0_thought_engine_v3 as thought_engine
from src.modules.skills.implementations.nan0_cognition_router_v1 import route_thought
from src.modules.skills.implementations.nan0_skill import Nan0Skill


class _FakeBrain:
    async def perform_output_task(self, mood, message, speech_packet=None):
        raise AssertionError("thought creation must not produce output")


class _FakeOllamaResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _model_payload(text):
    payload = {
        "thought_text": text,
        "mood": "suspicion",
        "pressure": 1.0,
        "novelty": 0.7,
        "speakability": 0.8,
        "relationship_charge": 0.7,
        "ego_charge": 0.5,
        "vision_charge": 0.0,
        "memory_write_candidate": False,
        "suppression_reason": None,
    }
    return payload, json.dumps(payload), 7


def _install_valid_inputs(monkeypatch, text):
    monkeypatch.setattr(thought_engine, "_call_ollama", lambda *args, **kwargs: _model_payload(text))
    monkeypatch.setattr(thought_engine, "_read_presence_state", lambda: {})
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: {})
    monkeypatch.setattr(thought_engine, "_query_recent_memory", lambda query, limit=4: [])
    monkeypatch.setattr(thought_engine, "get_session_timeline_context", lambda: {})
    monkeypatch.setattr(thought_engine, "get_conversation_continuity_context", lambda event: {})
    monkeypatch.setattr(thought_engine, "get_relationship_memory_context", lambda actor_id: {})


def _assert_valid_packet(packet, source, actor_id):
    assert packet["thought_id"].startswith("thought_")
    assert packet["private_text"]
    assert packet["thought_text"] == packet["private_text"]
    assert packet["source"] == source
    assert packet["target_actor_id"] == actor_id
    assert packet["thought_type"]
    assert packet["mood"]
    assert "speakability" in packet
    assert packet.get("suppression_reason") != "thought_generation_failed"


def test_ollama_json_adapter_consumes_live_three_value_contract(monkeypatch):
    expected = _model_payload("Private thought survived the adapter boundary.")
    monkeypatch.setattr(thought_engine, "_call_ollama", lambda *args, **kwargs: expected)

    parsed, raw, latency_ms = thought_engine._call_ollama_json("prompt", "qwen2.5:3b", 1.0)

    assert parsed == expected[0]
    assert raw == expected[1]
    assert latency_ms == expected[2]


def test_thought_http_adapter_calls_model_and_extracts_valid_response(monkeypatch):
    payload = _model_payload("The adapter finally carried my thought through.")[0]
    response_text = json.dumps(payload)
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["request"] = json
        captured["timeout"] = timeout
        return _FakeOllamaResponse({"response": response_text, "done": True})

    monkeypatch.setattr(thought_engine.requests, "post", fake_post)

    parsed, raw, latency_ms = thought_engine._call_ollama(
        "thought prompt", "qwen2.5:3b", 18, num_predict=180
    )

    assert parsed == payload
    assert json.loads(raw) == payload
    assert latency_ms >= 1
    assert captured["timeout"] == 18.0
    assert captured["request"]["options"]["num_predict"] == 180


def test_ollama_provider_extracts_only_valid_generate_response(monkeypatch):
    requests_sent = []
    responses = iter([
        _FakeOllamaResponse({"response": "  {\"thought_text\": \"Still here.\"}  "}),
        _FakeOllamaResponse({"response": None}),
        _FakeOllamaResponse(["malformed", "transport"]),
        _FakeOllamaResponse({"response": "spoken text"}),
    ])

    def fake_post(*args, **kwargs):
        requests_sent.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr(ollama_provider.requests, "post", fake_post)
    provider = ollama_provider.OllamaLLM(model_name="qwen2.5:3b")

    assert provider._generate("prompt", json_hint=True) == '{"thought_text": "Still here."}'
    assert provider._generate("prompt", json_hint=True) == ""
    assert provider._generate("prompt", json_hint=True) == ""
    assert provider._generate("prompt", json_hint=False, num_predict=180) == "spoken text"
    assert requests_sent[-1]["options"]["num_predict"] == 110


def test_kyo_runtime_input_produces_valid_private_thought_packet(monkeypatch):
    _install_valid_inputs(monkeypatch, "Kyo touched the generator again. My attention snapped toward her.")

    packet = thought_engine.generate_inner_thought_packet({
        "event_id": "event_kyo_runtime_contract",
        "source": "kyo_text",
        "speaker": "Kyo",
        "source_actor_id": "kyo",
        "text": "The generator failed again.",
        "addressed_to_nan0": True,
        "timestamp": 1,
    })

    _assert_valid_packet(packet, "kyo_text", "kyo")


def test_skill_runtime_boundary_does_not_mark_valid_input_as_thought_generation_failed(monkeypatch):
    config = BrainConfig()
    config.skills["nan0"]["enabled"] = True
    skill = Nan0Skill("nan0", config, _FakeBrain())
    _install_valid_inputs(monkeypatch, "Kyo touched the generator again. I felt that one.")
    monkeypatch.setattr(skill, "_attach_continuity_context", lambda event: None)
    monkeypatch.setattr(nan0_skill_module, "record_thought_packet", lambda packet: None)

    packet = asyncio.run(skill._create_inner_thought({
        "event_id": "event_kyo_skill_contract",
        "source": "kyo_text",
        "speaker": "Kyo",
        "source_actor_id": "kyo",
        "text": "The generator failed again.",
        "addressed_to_nan0": True,
        "timestamp": 1,
    }))

    _assert_valid_packet(packet, "kyo_text", "kyo")


def test_autonomous_background_input_produces_valid_private_thought_packet(monkeypatch):
    _install_valid_inputs(monkeypatch, "The room is too quiet. I can hear my own wires judging it.")

    packet = thought_engine.generate_inner_thought_packet({
        "event_id": "event_autonomous_contract",
        "source": "monologue",
        "speaker": "Nan0",
        "source_actor_id": "nan0",
        "text": "Nan0 has been quiet for a while.",
        "thought_seed": "idle_room_presence",
        "addressed_to_nan0": False,
        "timestamp": 1,
    })

    _assert_valid_packet(packet, "monologue", "nan0")


def test_all_context_inputs_are_consumed_without_unpack_errors(monkeypatch):
    prompt_capture = {}
    payload = _model_payload("Kyo keeps circling this topic. I noticed, obviously.")

    def fake_call(prompt, *args, **kwargs):
        prompt_capture["prompt"] = prompt
        return payload

    monkeypatch.setattr(thought_engine, "_call_ollama", fake_call)
    monkeypatch.setattr(
        thought_engine,
        "_read_presence_state",
        lambda: {"presence_mode": "watching", "pressure": 0.4},
    )
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: {"screen_state": "stable"})
    monkeypatch.setattr(
        thought_engine,
        "_query_recent_memory",
        lambda query, limit=4: [{
            "kind": "retrieved_memory_fact",
            "provider": "memory_storage",
            "facts_only": True,
            "content": "Kyo mentioned the generator before.",
            "source": {"event_id": "old_event", "source_actor_id": "kyo"},
        }],
    )
    monkeypatch.setattr(
        thought_engine,
        "load_identity_memory",
        lambda: {
            "actors": {"kyo": {"display_name": "Kyo", "relationship": "creator_anchor"}},
            "rules": {"never_call_kyo_user": True},
        },
    )
    monkeypatch.setattr(
        thought_engine,
        "get_relationship_memory_context",
        lambda actor_id: {
            "provider": "relationship_memory",
            "facts_only": True,
            "actor_id": actor_id,
            "relationship_status": "bonded",
        },
    )
    monkeypatch.setattr(
        thought_engine,
        "get_session_timeline_context",
        lambda: {
            "provider": "session_timeline",
            "facts_only": True,
            "repeat_facts": [{"kind": "topic", "value": "generator", "count": 2}],
        },
    )
    monkeypatch.setattr(
        thought_engine,
        "get_conversation_continuity_context",
        lambda event: {
            "provider": "conversation_continuity",
            "facts_only": True,
            "thread_id": "thread_generator",
            "current_event": {"source_actor_id": "kyo"},
        },
    )

    packet = thought_engine.generate_inner_thought_packet({
        "event_id": "event_full_context_contract",
        "source": "kyo_text",
        "speaker": "Kyo",
        "source_actor_id": "kyo",
        "text": "It broke twice now.",
        "addressed_to_nan0": True,
        "timestamp": 1,
        "_enriched_context": {
            "continuity_context": {"topic": "generator"},
            "conversation_thread": {"topic": "generator", "relation": "same_topic"},
            "expectation_context": {"expected": "another failure"},
            "goal_context": {"goal": "watch the generator"},
            "reflex_context": {"trigger": "repeat_failure"},
        },
    })

    _assert_valid_packet(packet, "kyo_text", "kyo")
    assert packet["relationship_context"]["actor"]["display_name"] == "Kyo"
    assert packet["relationship_context"]["relationship_memory"]["relationship_status"] == "bonded"
    assert packet["continuity_context"]["session_timeline"]["repeat_facts"][0]["count"] == 2
    assert packet["continuity_context"]["conversation_continuity"]["persistent_thread"]["thread_id"] == "thread_generator"
    event_context = packet["continuity_context"]["event_continuity"]
    assert event_context["expectation_context"]["expected"] == "another failure"
    assert event_context["goal_context"]["goal"] == "watch the generator"
    assert event_context["reflex_context"]["trigger"] == "repeat_failure"
    assert "thread_generator" in prompt_capture["prompt"]


def test_malformed_context_provider_shapes_are_rejected_without_crashing(monkeypatch):
    _install_valid_inputs(monkeypatch, "Kyo poked the same wire again. Irritating, but I noticed.")
    monkeypatch.setattr(thought_engine, "_read_presence_state", lambda: ("bad", "presence", "shape"))
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: ["bad", "vision"])
    monkeypatch.setattr(thought_engine, "_query_recent_memory", lambda query, limit=4: ("bad", "memory"))
    monkeypatch.setattr(thought_engine, "load_identity_memory", lambda: ["bad", "identity"])
    monkeypatch.setattr(thought_engine, "get_relationship_memory_context", lambda actor_id: ("bad", "relationship"))
    monkeypatch.setattr(thought_engine, "get_session_timeline_context", lambda: ("bad", "timeline"))
    monkeypatch.setattr(thought_engine, "get_conversation_continuity_context", lambda event: ("bad", "conversation"))

    packet = thought_engine.generate_inner_thought_packet({
        "event_id": "event_malformed_context_contract",
        "source": "kyo_text",
        "speaker": "Kyo",
        "source_actor_id": "kyo",
        "text": "Can you still think?",
        "addressed_to_nan0": True,
        "timestamp": 1,
        "_enriched_context": {
            "continuity_context": ("bad", "continuity"),
            "expectation_context": object(),
            "goal_context": ("bad", "goal"),
            "reflex_context": object(),
        },
    })

    _assert_valid_packet(packet, "kyo_text", "kyo")
    assert packet["emotional_context"] == {}
    assert packet["vision_context"] == {}
    assert packet["memory_context"] == []
    assert packet["relationship_context"]["source_actor_id"] == "kyo"
    assert packet["relationship_context"]["relationship_memory"] == {}
    assert "conversation_continuity" not in packet["continuity_context"]


def test_malformed_model_adapter_shape_is_rejected_at_adapter_boundary(monkeypatch):
    monkeypatch.setattr(thought_engine, "_call_ollama", lambda *args, **kwargs: ("one", "two", "three", "four"))

    parsed, raw, latency_ms = thought_engine._call_ollama_json("prompt", "qwen2.5:3b", 1.0)

    assert parsed == {}
    assert raw == ""
    assert latency_ms == 0


@pytest.mark.parametrize("provider_payload", [{"response": None}, {"message": "wrong key"}, ["wrong shape"]])
def test_empty_or_malformed_model_output_suppresses_without_speech(monkeypatch, provider_payload):
    monkeypatch.setattr(thought_engine, "_read_presence_state", lambda: {})
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: {})
    monkeypatch.setattr(thought_engine, "_query_recent_memory", lambda query, limit=4: [])
    monkeypatch.setattr(thought_engine, "get_session_timeline_context", lambda: {})
    monkeypatch.setattr(thought_engine, "get_conversation_continuity_context", lambda event: {})
    monkeypatch.setattr(thought_engine, "get_relationship_memory_context", lambda actor_id: {})
    monkeypatch.setattr(
        thought_engine.requests,
        "post",
        lambda *args, **kwargs: _FakeOllamaResponse(provider_payload),
    )
    monkeypatch.setattr(thought_engine, "_call_ollama_plain", lambda *args, **kwargs: ("", 1))

    packet = thought_engine.generate_inner_thought_packet({
        "event_id": "event_bad_model_response",
        "source": "kyo_text",
        "speaker": "Kyo",
        "source_actor_id": "kyo",
        "text": "Did the model answer?",
        "addressed_to_nan0": True,
        "timestamp": 1,
    })

    assert packet["thought_id"].startswith("thought_")
    assert packet["private_text"] == ""
    assert packet["suppression_reason"] == "empty_private_thought"


def test_speech_generation_rejects_direct_event_and_router_does_not_invent_thought():
    config = BrainConfig()
    config.skills["nan0"]["enabled"] = True
    skill = Nan0Skill("nan0", config, _FakeBrain())
    direct_event = {
        "event_id": "event_direct_to_speech",
        "source": "kyo_text",
        "speaker": "Kyo",
        "text": "Speak directly from this event.",
    }

    with pytest.raises(RuntimeError, match="missing thought_id"):
        asyncio.run(skill._generate_line(direct_event))

    decision = route_thought(direct_event)
    assert decision["decision"] == "suppress"
    assert decision["reason"] == "missing_thought_origin"
    assert decision["thought_id"] is None
    assert not decision.get("line")
