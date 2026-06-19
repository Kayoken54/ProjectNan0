import asyncio
import time
import datetime

from typing import List, Dict
from pathlib import Path

from src.core.config import BrainConfig
from src.modules.nan0.identity_memory import normalize_actor_id
from src.modules.skills.base_skill import BaseSkill
from src.utils.logger import get_logger

logger = get_logger("bea.skills.memory")


class MemorySkill(BaseSkill):
    _shared_storage = None
    _storage_initialized = False
    _storage_class = None
    _generator_class = None

    def __init__(self, name: str, config: BrainConfig, brain):
        super().__init__(name, config, brain)

        self.memory_db_path = self.skill_config.get(
            "chroma_path",
            "data/memory_db"
        )

        self.embedding_mode = self.skill_config.get(
            "embedding_model",
            "local"
        )

        self.local_embedding_model = self.skill_config.get(
            "local_embedding_model",
            "all-MiniLM-L6-v2"
        )

        self.openai_key = getattr(config, "openai_key", None)

        Path(self.memory_db_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if MemorySkill._shared_storage is None:
            MemoryStorage = self._memory_storage_class()
            logger.info(
                "MemorySkill: Creating shared MemoryStorage singleton."
            )

            MemorySkill._shared_storage = MemoryStorage(
                db_path=self.memory_db_path,
                embedding_mode=self.embedding_mode,
                embedding_model=self.local_embedding_model,
                openai_key=self.openai_key
            )

        self.storage = MemorySkill._shared_storage
        self.generator = None


    @classmethod
    def _memory_storage_class(cls):
        if cls._storage_class is None:
            from src.modules.skills.memory.storage import MemoryStorage

            cls._storage_class = MemoryStorage
        return cls._storage_class

    @classmethod
    def _diary_generator_class(cls):
        if cls._generator_class is None:
            from src.modules.skills.memory.generator import DiaryGenerator

            cls._generator_class = DiaryGenerator
        return cls._generator_class

    def initialize(self):
        if not self.enabled:
            logger.info(
                "MemorySkill: Disabled, skipping initialization."
            )
            return

        logger.info(
            "MemorySkill: Lazy mode enabled; ChromaDB and diary generator will initialize on first memory use."
        )


    def _ensure_storage_initialized(self) -> bool:
        if MemorySkill._storage_initialized:
            return True

        logger.info(
            "MemorySkill: Lazy-initializing shared MemoryStorage."
        )
        if not self.storage.initialize():
            logger.error(
                "MemorySkill: Storage initialization failed."
            )
            self.skill_config["enabled"] = False
            return False

        MemorySkill._storage_initialized = True
        return True

    def _ensure_generator_initialized(self) -> bool:
        if self.generator:
            return True
        if not hasattr(self.context, "llm"):
            logger.error(
                "MemorySkill: Brain LLM not available."
            )
            return False

        DiaryGenerator = self._diary_generator_class()
        self.generator = DiaryGenerator(
            self.context.llm
        )
        return True

    async def start(self):
        if self.is_active:
            logger.info(
                "MemorySkill: Already active."
            )
            return

        await super().start()

        logger.info(
            "MemorySkill: Started."
        )

    async def stop(self):
        if not self.is_active:
            return

        await super().stop()

        logger.info(
            "MemorySkill: Stopped."
        )

    async def update(self):
        """
        Sacred Architecture Runtime Rule:

        MemorySkill must NEVER:
        - construct MemoryStorage
        - call storage.initialize()
        - recreate embeddings
        - reopen ChromaDB

        update() is intentionally passive.
        """
        return

    def process_previous_session(
        self,
        session_id: str,
        history: List[Dict]
    ):
        if not self.enabled:
            return

        if not self._ensure_storage_initialized():
            return

        if len(history) < 2:
            logger.warning(
                f"MemorySkill: Session {session_id} too short."
            )
            return

        if self.storage.entry_exists(
            f"diary_{session_id}"
        ):
            logger.info(
                f"MemorySkill: Session {session_id} diary already saved; checking episodic facts."
            )

        asyncio.create_task(
            self._process_session_async(
                session_id,
                history
            )
        )

    def _save_episodic_events(self, session_id: str, history: List[Dict]):
        """Persist raw authored turns with stable ownership and provenance.

        These records are the fact-only retrieval surface. Generated diary
        summaries remain a separate archive layer and are never relabeled as
        observed facts.
        """
        for index, message in enumerate(history):
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or "unknown").strip().lower()
            if role not in {"user", "assistant"}:
                continue

            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            nested = message.get("metadata")
            nested = nested if isinstance(nested, dict) else {}
            explicit_source = message.get("source") or nested.get("source")
            source = str(
                explicit_source
                or ("kyo_text" if role == "user" else "nan0_speech")
            ).strip().lower()
            explicit_actor = (
                message.get("source_actor_id")
                or nested.get("source_actor_id")
                or message.get("speaker")
                or nested.get("speaker")
            )
            if explicit_actor:
                source_actor_id = normalize_actor_id(str(explicit_actor), source)
            elif role == "user" and not explicit_source:
                source_actor_id = "kyo"
            elif role == "assistant":
                source_actor_id = "nan0"
            else:
                source_actor_id = normalize_actor_id(source, source)

            event_id = str(
                message.get("event_id")
                or nested.get("event_id")
                or message.get("id")
                or f"{session_id}_{index}"
            )
            entry_id = f"memory_event_{session_id}_{event_id}"
            if self.storage.entry_exists(entry_id):
                continue

            metadata = {
                "memory_kind": "episodic_event",
                "facts_only": True,
                "character_id": "nan0",
                "session_id": session_id,
                "event_id": event_id,
                "source": source,
                "source_actor_id": source_actor_id,
                "role": role,
                "timestamp": str(message.get("timestamp") or ""),
            }
            self.storage.add_entry(content.strip(), metadata, entry_id)

    async def _process_session_async(
        self,
        session_id: str,
        history: List[Dict]
    ):
        if not self._ensure_storage_initialized():
            return

        try:
            await asyncio.to_thread(self._save_episodic_events, session_id, history)

            if self.storage.entry_exists(
                f"diary_{session_id}"
            ):
                return

            if not self._ensure_generator_initialized():
                return

            diary_json = await self.generator.generate_diary(
                history
            )

            if not diary_json:
                return

            self._save_diary(
                session_id,
                diary_json
            )

        except Exception as e:
            logger.error(
                f"MemorySkill: Error processing session: {e}"
            )

    def _save_diary(
        self,
        session_id: str,
        diary_json: Dict
    ):
        diary_content = diary_json.get(
            "diary_content",
            ""
        )

        tags = diary_json.get(
            "tags",
            []
        )

        user_id = diary_json.get(
            "user_id",
            "owner"
        )

        if not diary_content:
            return

        timestamp = time.time()

        today_str = datetime.datetime.now().strftime(
            "%Y-%m-%d"
        )

        metadata = {
            "memory_kind": "generated_session_summary",
            "facts_only": False,
            "character_id": "nan0",
            "source": "memory_consolidation",
            "timestamp": timestamp,
            "date": today_str,
            "user_id": user_id,
            "tags": ",".join(tags),
            "session_id": session_id
        }

        self.storage.add_entry(
            diary_content,
            metadata,
            f"diary_{session_id}"
        )

        logger.info(
            f"MemorySkill: Saved Diary for {session_id}"
        )

    def retrieve_context(
        self,
        query: str,
        limit: int = 3
    ) -> str:
        if not self.enabled:
            return ""

        if not self._ensure_storage_initialized():
            return ""

        try:
            fetch_limit = limit * 3

            results = self.storage.query_similar(
                query,
                fetch_limit
            )

            if not results:
                return ""

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            scored_entries = []

            now = time.time()

            for i, doc in enumerate(docs):
                if not doc:
                    continue

                similarity = 1 - dists[i]

                timestamp = metas[i].get(
                    "timestamp",
                    0
                )

                age_seconds = now - timestamp
                age_days = age_seconds / 86400
                decay_rate = 0.1

                recency = 1 / (
                    1 + age_days * decay_rate
                )

                final_score = (
                    similarity * 0.7
                ) + (
                    recency * 0.3
                )

                scored_entries.append({
                    "doc": doc,
                    "date": metas[i].get(
                        "date",
                        "Unknown"
                    ),
                    "score": final_score
                })

            scored_entries.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            top_entries = scored_entries[:limit]

            context_str = "RELEVANT MEMORIES:\n"

            found = False

            for entry in top_entries:
                context_str += (
                    f"- [{entry['date']}]: "
                    f"{entry['doc']}\n"
                )
                found = True

            return context_str if found else ""

        except Exception as e:
            logger.error(
                f"MemorySkill: Error retrieving context: {e}"
            )

            return ""

    def save_current_session(self):
        if not self.enabled:
            return False

        if not hasattr(
            self.context,
            "history_manager"
        ):
            return False

        hm = self.context.history_manager

        session_id = hm.session_id
        history = hm.history

        if not session_id or not history:
            logger.warning(
                "MemorySkill: No active session."
            )
            return False

        logger.info(
            f"MemorySkill: Manual save for {session_id}"
        )

        self.process_previous_session(
            session_id,
            history
        )

        return True

    async def save_all_pending(self):
        if not self.enabled:
            return

        logger.info(
            "MemorySkill: Checking pending saves..."
        )

        if hasattr(
            self.context,
            "history_manager"
        ):
            hm = self.context.history_manager

            session_id = hm.session_id
            history = hm.history

            if (
                session_id
                and history
                and len(history) >= 2
            ):
                if not self.storage.entry_exists(
                    f"diary_{session_id}"
                ):
                    logger.info(
                        f"MemorySkill: Saving final session {session_id}"
                    )

                    await self._process_session_async(
                        session_id,
                        history
                    )

                else:
                    logger.info(
                        "MemorySkill: Session already saved."
                    )
