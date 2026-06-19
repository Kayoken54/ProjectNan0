from src.modules.skills.implementations import nan0_thought_engine_v3 as thought_engine
from src.modules.skills.memory.memory_skill import MemorySkill


class RecordingStorage:
    def __init__(self):
        self.entries = {}

    def entry_exists(self, entry_id):
        return entry_id in self.entries

    def add_entry(self, content, metadata, entry_id):
        self.entries[entry_id] = {
            "content": content,
            "metadata": metadata,
        }


def _memory_skill_with_storage(storage):
    skill = MemorySkill.__new__(MemorySkill)
    skill.storage = storage
    return skill


def test_episodic_memory_persists_kyo_and_nan0_actor_ownership():
    storage = RecordingStorage()
    skill = _memory_skill_with_storage(storage)

    skill._save_episodic_events("session_1", [
        {
            "role": "user",
            "content": "I watched the generator fail.",
            "event_id": "event_kyo_1",
            "source": "kyo_voice",
            "source_actor_id": "Kyo",
            "timestamp": "2026-06-19T01:00:00",
        },
        {
            "role": "assistant",
            "content": "Of course it failed while I was watching.",
            "event_id": "event_nan0_1",
            "timestamp": "2026-06-19T01:00:01",
        },
    ])

    records = list(storage.entries.values())
    assert records[0]["metadata"]["source_actor_id"] == "kyo"
    assert records[1]["metadata"]["source_actor_id"] == "nan0"
    assert all(record["metadata"]["facts_only"] is True for record in records)
    assert all(record["metadata"]["memory_kind"] == "episodic_event" for record in records)
    assert all(record["metadata"]["character_id"] == "nan0" for record in records)


def test_generated_diary_is_archived_as_non_fact_memory():
    storage = RecordingStorage()
    skill = _memory_skill_with_storage(storage)

    skill._save_diary("session_1", {
        "diary_content": "A generated interpretation of the session.",
        "tags": ["generator"],
    })

    metadata = storage.entries["diary_session_1"]["metadata"]
    assert metadata["memory_kind"] == "generated_session_summary"
    assert metadata["facts_only"] is False
    assert metadata["source"] == "memory_consolidation"


def test_thought_retrieval_rejects_generated_summary_and_preserves_fact_owner(monkeypatch):
    class FakeMemoryStorage:
        def __init__(self, **kwargs):
            pass

        def initialize(self):
            return True

        def query_similar(self, query, limit):
            assert limit >= 3
            return {
                "ids": [["diary_session_1", "memory_event_session_1_event_kyo_1"]],
                "documents": [[
                    "A generated conclusion about what Kyo meant.",
                    "I watched the generator fail.",
                ]],
                "metadatas": [[
                    {
                        "memory_kind": "generated_session_summary",
                        "facts_only": False,
                        "session_id": "session_1",
                        "source": "memory_consolidation",
                    },
                    {
                        "memory_kind": "episodic_event",
                        "facts_only": True,
                        "character_id": "nan0",
                        "session_id": "session_1",
                        "event_id": "event_kyo_1",
                        "source": "kyo_voice",
                        "source_actor_id": "Kyo",
                        "role": "user",
                    },
                ]],
                "distances": [[0.05, 0.12]],
            }

    monkeypatch.setattr(thought_engine, "MemoryStorage", FakeMemoryStorage)
    monkeypatch.setattr(
        thought_engine,
        "_memory_config",
        lambda: {"enabled": True, "chroma_path": "unused", "embedding_model": "local"},
    )

    memories = thought_engine._query_recent_memory("generator failed", limit=1)

    assert len(memories) == 1
    assert memories[0]["content"] == "I watched the generator fail."
    assert memories[0]["facts_only"] is True
    assert memories[0]["fact_type"] == "authored_event"
    assert memories[0]["generated_conclusion"] is False
    assert memories[0]["source"]["source_actor_id"] == "kyo"
    assert memories[0]["source"]["event_id"] == "event_kyo_1"
    assert memories[0]["source"]["memory_id"] == "memory_event_session_1_event_kyo_1"
    assert "conclusion" not in memories[0]
    assert "speech_packet" not in memories[0]
