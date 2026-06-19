import json

from src.modules.nan0.conversation_continuity import ConversationContinuity
from src.modules.nan0.identity_memory import actor_ownership_from_event
from src.modules.nan0.relationship_memory import RelationshipMemory
from src.modules.nan0.session_timeline import SessionTimeline
from src.modules.skills.implementations import nan0_cognition_router_v1 as router
from src.modules.skills.implementations import nan0_thought_engine_v3 as thought_engine


def _install_fake_generation(monkeypatch, captured):
    def fake_call_ollama(prompt, model, timeout, num_predict=120, temperature=0.78, system=None):
        captured["prompt"] = prompt
        return {
            "thought_text": "Kyo's timing snagged my attention again. Annoying.",
            "mood": "suspicion",
            "pressure": 1.0,
            "novelty": 0.5,
            "speakability": 0.8,
            "relationship_charge": 0.7,
            "ego_charge": 0.4,
            "vision_charge": 0.0,
            "memory_write_candidate": False,
            "suppression_reason": None,
        }, "{}", 1

    monkeypatch.setattr(thought_engine, "_call_ollama_json", fake_call_ollama)
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: {})
    monkeypatch.setattr(thought_engine, "_read_presence_state", lambda: {})
    monkeypatch.setattr(thought_engine, "_query_recent_memory", lambda query, limit=4: [])


def _kyo_event():
    return {
        "event_id": "event_kyo_ownership",
        "source": "kyo_voice",
        "speaker": "Kyo",
        "source_actor_id": "kyo",
        "text": "I watched the generator fail again.",
        "addressed_to_nan0": True,
        "timestamp": 1,
        "_enriched_context": {
            "continuity_context": {
                "source_actor_id": "nan0",
                "speaker": "Nan0",
                "fact": "historical context cannot own this event",
            },
            "conversation_thread": {
                "topic": "generator",
                "source_actor_id": "nan0",
            },
        },
    }


def test_kyo_ownership_survives_context_assembly_and_prompt(monkeypatch):
    captured = {}
    _install_fake_generation(monkeypatch, captured)
    monkeypatch.setattr(
        thought_engine,
        "get_session_timeline_context",
        lambda: {"provider": "session_timeline", "source_actor_id": "nan0"},
    )
    monkeypatch.setattr(
        thought_engine,
        "get_conversation_continuity_context",
        lambda event: {"provider": "conversation_continuity", "current_event": {"source_actor_id": "nan0"}},
    )
    monkeypatch.setattr(thought_engine, "get_relationship_memory_context", lambda actor_id: {})

    packet = thought_engine.generate_inner_thought_packet(_kyo_event())

    assert packet["target_actor_id"] == "kyo"
    assert packet["event_context"]["source_actor_id"] == "kyo"
    assert packet["event_context"]["actor_contract"]["source_actor_id"] == "kyo"
    assert packet["continuity_context"]["actor_ownership"]["source_actor_id"] == "kyo"
    assert packet["continuity_context"]["event_continuity"]["continuity_context"]["source_actor_id"] == "nan0"
    assert "AUTHORITATIVE EVENT OWNERSHIP" in captured["prompt"]
    assert '"ownership_authority": "event"' in captured["prompt"]


def test_explicit_nan0_owner_outranks_source_family():
    ownership = actor_ownership_from_event({
        "event_id": "event_internal_forwarded",
        "source": "kyo_voice",
        "speaker": "Kyo",
        "source_actor_id": "nan0",
    })

    assert ownership["source_actor_id"] == "nan0"
    assert ownership["ownership_authority"] == "event"


def test_relationship_memory_reaches_thought_generation(monkeypatch):
    captured = {}
    _install_fake_generation(monkeypatch, captured)
    monkeypatch.setattr(thought_engine, "get_session_timeline_context", lambda: {})
    monkeypatch.setattr(thought_engine, "get_conversation_continuity_context", lambda event: {})
    monkeypatch.setattr(
        thought_engine,
        "get_relationship_memory_context",
        lambda actor_id: {
            "provider": "relationship_memory",
            "facts_only": True,
            "actor_id": actor_id,
            "relationship_status": "bonded",
            "emotional_balance": 0.8,
        },
    )

    packet = thought_engine.generate_inner_thought_packet(_kyo_event())

    relationship = packet["relationship_context"]["relationship_memory"]
    assert relationship["actor_id"] == "kyo"
    assert relationship["relationship_status"] == "bonded"
    assert '"relationship_status": "bonded"' in captured["prompt"]


def test_conversation_continuity_reaches_thought_generation(monkeypatch):
    captured = {}
    _install_fake_generation(monkeypatch, captured)
    monkeypatch.setattr(thought_engine, "get_session_timeline_context", lambda: {})
    monkeypatch.setattr(thought_engine, "get_relationship_memory_context", lambda actor_id: {})
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

    packet = thought_engine.generate_inner_thought_packet(_kyo_event())

    continuity = packet["continuity_context"]["conversation_continuity"]
    assert continuity["attached_thread"]["topic"] == "generator"
    assert continuity["persistent_thread"]["thread_id"] == "thread_generator"
    assert '"thread_id": "thread_generator"' in captured["prompt"]


def test_session_timeline_reaches_thought_generation(monkeypatch):
    captured = {}
    _install_fake_generation(monkeypatch, captured)
    monkeypatch.setattr(thought_engine, "get_conversation_continuity_context", lambda event: {})
    monkeypatch.setattr(thought_engine, "get_relationship_memory_context", lambda actor_id: {})
    monkeypatch.setattr(
        thought_engine,
        "get_session_timeline_context",
        lambda: {
            "provider": "session_timeline",
            "facts_only": True,
            "repeat_facts": [{"kind": "topic", "value": "generator", "count": 3}],
        },
    )

    packet = thought_engine.generate_inner_thought_packet(_kyo_event())

    timeline = packet["continuity_context"]["session_timeline"]
    assert timeline["repeat_facts"][0]["count"] == 3
    assert packet["continuity_context"]["actor_ownership"]["source_actor_id"] == "kyo"
    assert '"count": 3' in captured["prompt"]


def test_retrieved_memory_is_source_aware_and_fact_only(monkeypatch):
    class FakeMemoryStorage:
        def __init__(self, **kwargs):
            pass

        def initialize(self):
            return True

        def query_similar(self, query, limit):
            return {
                "documents": [["Kyo said the generator failed."]],
                "metadatas": [[{
                    "session_id": "session_1",
                    "event_id": "event_1",
                    "source": "kyo_voice",
                    "source_actor_id": "kyo",
                }]],
                "distances": [[0.12]],
            }

    monkeypatch.setattr(thought_engine, "MemoryStorage", FakeMemoryStorage)
    monkeypatch.setattr(
        thought_engine,
        "_memory_config",
        lambda: {"enabled": True, "chroma_path": "unused", "embedding_model": "local"},
    )

    memories = thought_engine._query_recent_memory("generator failed", limit=1)

    assert memories == [{
        "kind": "retrieved_memory_fact",
        "provider": "memory_storage",
        "facts_only": True,
        "content": "Kyo said the generator failed.",
        "source": {
            "session_id": "session_1",
            "event_id": "event_1",
            "source": "kyo_voice",
            "source_actor_id": "kyo",
        },
        "distance": 0.12,
    }]
    assert "conclusion" not in memories[0]
    assert "thought_text" not in memories[0]


def test_continuity_memory_and_timeline_do_not_create_speech_packets(tmp_path):
    event = _kyo_event()
    conversation = ConversationContinuity(str(tmp_path / "conversation.db")).context_for_event(event)

    relationship_store = RelationshipMemory(str(tmp_path / "relationship.db"))
    relationship_store.record_moment("kyo", "positive", "Kyo returned.", intensity=0.5)
    relationship = relationship_store.get_relationship_context("kyo")

    timeline_store = SessionTimeline()
    timeline_store.add_event(event)
    timeline = timeline_store.continuity_context()

    forbidden = {"line", "line_text", "speech_packet", "voice_enabled", "display_enabled", "decision", "route"}
    for provider_context in (conversation, relationship, timeline):
        serialized = json.dumps(provider_context)
        assert all(f'"{key}"' not in serialized for key in forbidden)

    routed = router.route_thought({
        "private_text": "A continuity fact is not a thought packet.",
        "continuity_context": timeline,
    })
    assert routed["decision"] == "suppress"
    assert routed["reason"] == "missing_thought_origin"
    assert routed["line"] == ""
