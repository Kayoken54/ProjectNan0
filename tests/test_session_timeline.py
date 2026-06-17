from pathlib import Path

from src.modules.nan0.session_timeline import SessionTimeline, get_continuity_context, record_session_event, reset_session_timeline
from src.modules.skills.implementations import nan0_thought_engine_v3 as thought_engine


def test_last_20_meaningful_events_retained_and_21_evicts_oldest():
    timeline = SessionTimeline(max_events=20)

    for index in range(21):
        timeline.add_event(
            {
                "timestamp": index,
                "event_type": "generator_completed",
                "source": "game",
                "speaker": "Kyo",
                "text": f"generator completed {index}",
                "priority": "medium",
            }
        )

    items = timeline.recent_items()
    assert len(items) == 20
    assert items[0]["summary"] == "generator completed 1"
    assert items[-1]["summary"] == "generator completed 20"


def test_repeated_event_types_and_actors_are_counted():
    timeline = SessionTimeline(max_events=20)
    for actor in ["Kyo", "Kyo", "Mira"]:
        timeline.add_event(
            {
                "timestamp": 1,
                "event_type": "Kyo_death",
                "source": "game",
                "speaker": actor,
                "text": f"{actor} died again",
                "priority": "high",
            }
        )

    context = timeline.continuity_context()
    assert context["repeat_counts"]["event_type"]["Kyo_death"] == 3
    assert context["repeat_counts"]["actor"]["Kyo"] == 2
    assert {fact["value"]: fact["count"] for fact in context["repeat_facts"]}["Kyo_death"] == 3


def test_topic_carryover_uses_existing_event_metadata_only():
    timeline = SessionTimeline(max_events=20)
    timeline.add_event(
        {
            "timestamp": 1,
            "event_type": "generator_completed",
            "source": "game",
            "speaker": "Kyo",
            "text": "generator completed",
            "tags": ["generator", "completed"],
            "priority": "high",
        }
    )

    context = timeline.continuity_context()
    assert "generator" in context["recent_topics"]
    assert "completed" in context["recent_topics"]


def test_continuity_context_reaches_thought_generation(monkeypatch):
    reset_session_timeline()
    for _ in range(3):
        record_session_event(
            {
                "timestamp": 1,
                "event_type": "generator_completed",
                "source": "game",
                "speaker": "Kyo",
                "text": "generator completed",
                "priority": "high",
            }
        )

    captured = {}

    def fake_call_ollama(prompt, model, timeout, num_predict=120, temperature=0.78, system=None):
        captured["prompt"] = prompt
        return {
            "thought_text": "That generator thing keeps happening. I noticed the pattern.",
            "mood": "suspicion",
            "pressure": 1.0,
            "novelty": 0.5,
            "speakability": 0.8,
            "relationship_charge": 0.5,
            "ego_charge": 0.6,
            "vision_charge": 0.0,
            "memory_write_candidate": False,
            "suppression_reason": None,
        }, "{}", 1

    monkeypatch.setattr(thought_engine, "_call_ollama_json", fake_call_ollama)
    monkeypatch.setattr(thought_engine, "_read_vision_context", lambda explicit=None: {})
    monkeypatch.setattr(thought_engine, "_read_presence_state", lambda: {})
    monkeypatch.setattr(thought_engine, "_query_recent_memory", lambda query, limit=4: [])

    packet = thought_engine.generate_inner_thought_packet(
        {
            "event_id": "event_continuity_1",
            "source": "game",
            "speaker": "Kyo",
            "source_actor_id": "kyo",
            "text": "another generator completed",
            "timestamp": 1,
        }
    )

    assert packet["thought_id"]
    assert packet["thought_text"] == packet["private_text"]
    assert "SESSION CONTINUITY" in captured["prompt"]
    assert "generator_completed" in captured["prompt"]
    assert packet["continuity_context"]["repeat_counts"]["event_type"]["generator_completed"] == 3


def test_speech_still_requires_thought_id():
    from src.modules.nan0 import output_normalizer

    packet = {"line_text": "No thought origin. No voice.", "mood": "muttering"}
    assert output_normalizer.validate_thought_id(packet) is False
    assert output_normalizer.normalize_speech_packet(packet) is None


def test_no_direct_event_to_speech_shortcut_in_timeline():
    timeline = SessionTimeline(max_events=20)
    item = timeline.add_event({"event_type": "Kyo_death", "source": "game", "speaker": "Kyo", "text": "Kyo died"})
    assert item is not None
    context = timeline.continuity_context()
    assert "line_text" not in context
    assert "speech" not in context
    assert "thought_id" not in context["recent_events"][0]


def test_persona_files_not_edited_by_continuity_patch():
    assert Path("data/prompts/nan0_persona.txt").exists()
    assert Path("data/prompts/nan0_speech_persona.txt").exists()
