"""
Phase 2: CONVERSATION CONTINUITY
"What were we talking about?"

Inputs are not isolated. They are part of conversation threads with lifecycle:
OPENING → DEVELOPMENT → PEAK → RESOLUTION → CLOSING → DORMANT

Tracks: topic, emotional arc, mood trajectory, unresolved questions, 
interrupted topics, promises made.

Interruptions are tracked. Nan0 can choose to return to old topics.
A "hello" after a 5-minute silence is not a new conversation — 
it's a reactivation of a dormant thread.

Data: In-memory active threads. Archived to SQLite after closing.
"""

import sqlite3
import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Dict, List, Tuple, Set
from enum import Enum

from src.modules.nan0.identity_memory import actor_ownership_from_event


class ThreadPhase(Enum):
    OPENING = "opening"
    DEVELOPMENT = "development"
    PEAK = "peak"
    RESOLUTION = "resolution"
    CLOSING = "closing"
    DORMANT = "dormant"


@dataclass
class ThreadMessage:
    timestamp: float
    actor_id: str
    text: str
    mood: Optional[str] = None
    thought_id: Optional[str] = None
    addressed_to_nan0: bool = False


@dataclass
class ConversationThread:
    thread_id: str
    created_at: float
    last_active: float
    phase: ThreadPhase
    participants: Set[str] = field(default_factory=set)
    messages: List[ThreadMessage] = field(default_factory=list)
    topic: Optional[str] = None
    emotional_arc: List[Tuple[float, str, float]] = field(default_factory=list)  # (time, mood, intensity)
    unresolved_questions: List[str] = field(default_factory=list)
    interrupted_topics: List[str] = field(default_factory=list)
    promises_made: List[str] = field(default_factory=list)
    summary: Optional[str] = None

    # Reactivation tracking
    reactivation_count: int = 0
    original_thread_id: Optional[str] = None  # If reactivated from dormant

    @property
    def is_active(self) -> bool:
        return self.phase not in (ThreadPhase.CLOSING, ThreadPhase.DORMANT)

    @property
    def seconds_since_last_message(self) -> float:
        return time.time() - self.last_active

    @property
    def message_count(self) -> int:
        return len(self.messages)


class ConversationContinuity:
    """
    Manages conversation threads for Nan0.

    Integration: Called from nan0_skill._build_pressure_event()
    to enrich EventContext with thread state.
    """

    DORMANT_THRESHOLD = 300  # 5 minutes
    ARCHIVE_THRESHOLD = 1800  # 30 minutes
    MAX_ACTIVE_THREADS = 10
    MAX_MESSAGES_PER_THREAD = 50

    def __init__(self, db_path: str = "data/nan0/conversation_threads.db"):
        self.db_path = Path(db_path)
        self._active_threads: Dict[str, ConversationThread] = {}
        self._dormant_threads: Dict[str, ConversationThread] = {}
        self._actor_current_thread: Dict[str, str] = {}  # actor_id -> thread_id
        self._event_threads: Dict[str, str] = {}
        self._init_db()
        self._load_dormant_threads()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                created_at REAL,
                last_active REAL,
                phase TEXT,
                participants TEXT,
                messages TEXT,
                topic TEXT,
                emotional_arc TEXT,
                unresolved_questions TEXT,
                interrupted_topics TEXT,
                promises_made TEXT,
                summary TEXT,
                reactivation_count INTEGER DEFAULT 0,
                original_thread_id TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_dormant_threads(self):
        """Load recently dormant threads that could be reactivated."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Load threads dormant for less than 24 hours
        cutoff = time.time() - 86400
        cursor.execute(
            "SELECT * FROM threads WHERE phase = 'dormant' AND last_active > ?",
            (cutoff,)
        )

        for row in cursor.fetchall():
            thread = self._row_to_thread(row)
            self._dormant_threads[thread.thread_id] = thread

        conn.close()

    def _row_to_thread(self, row) -> ConversationThread:
        return ConversationThread(
            thread_id=row[0],
            created_at=row[1],
            last_active=row[2],
            phase=ThreadPhase(row[3]),
            participants=set(json.loads(row[4])) if row[4] else set(),
            messages=[ThreadMessage(**m) for m in json.loads(row[5])] if row[5] else [],
            topic=row[6],
            emotional_arc=json.loads(row[7]) if row[7] else [],
            unresolved_questions=json.loads(row[8]) if row[8] else [],
            interrupted_topics=json.loads(row[9]) if row[9] else [],
            promises_made=json.loads(row[10]) if row[10] else [],
            summary=row[11],
            reactivation_count=row[12],
            original_thread_id=row[13]
        )

    def _thread_to_row(self, thread: ConversationThread) -> tuple:
        return (
            thread.thread_id,
            thread.created_at,
            thread.last_active,
            thread.phase.value,
            json.dumps(list(thread.participants)),
            json.dumps([asdict(m) for m in thread.messages]),
            thread.topic,
            json.dumps(thread.emotional_arc),
            json.dumps(thread.unresolved_questions),
            json.dumps(thread.interrupted_topics),
            json.dumps(thread.promises_made),
            thread.summary,
            thread.reactivation_count,
            thread.original_thread_id
        )

    def _save_thread(self, thread: ConversationThread):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO threads 
            (thread_id, created_at, last_active, phase, participants, messages,
             topic, emotional_arc, unresolved_questions, interrupted_topics,
             promises_made, summary, reactivation_count, original_thread_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, self._thread_to_row(thread))

        conn.commit()
        conn.close()

    def process_input(self, actor_id: str, text: str, 
                     mood: Optional[str] = None,
                     thought_id: Optional[str] = None,
                     addressed_to_nan0: bool = False) -> ConversationThread:
        """
        Process an input and return the thread it belongs to.
        This is the main entry point called from _build_pressure_event().
        """
        now = time.time()

        # Check for reactivation
        existing_thread_id = self._actor_current_thread.get(actor_id)

        if existing_thread_id and existing_thread_id in self._active_threads:
            thread = self._active_threads[existing_thread_id]

            # Check if thread went dormant
            if thread.seconds_since_last_message > self.DORMANT_THRESHOLD:
                # Reactivate
                thread.phase = ThreadPhase.OPENING
                thread.reactivation_count += 1
                thread.last_active = now

            # Add message
            self._add_message(thread, actor_id, text, mood, thought_id, addressed_to_nan0)

        else:
            # Check for dormant thread reactivation
            dormant_match = self._find_dormant_match(actor_id, text)

            if dormant_match:
                # Reactivate dormant thread
                thread = dormant_match
                self._dormant_threads.pop(thread.thread_id, None)
                thread.phase = ThreadPhase.OPENING
                thread.reactivation_count += 1
                thread.last_active = now
                self._active_threads[thread.thread_id] = thread
                self._actor_current_thread[actor_id] = thread.thread_id

                self._add_message(thread, actor_id, text, mood, thought_id, addressed_to_nan0)

            else:
                # Create new thread
                thread = self._create_thread(actor_id, text, mood, thought_id, addressed_to_nan0)

        # Update thread phase
        self._update_phase(thread)

        # Prune old threads
        self._prune_threads()

        return thread

    def _create_thread(self, actor_id: str, text: str,
                      mood: Optional[str], thought_id: Optional[str],
                      addressed_to_nan0: bool) -> ConversationThread:
        """Create a new conversation thread."""
        thread_id = f"thread_{uuid.uuid4().hex[:16]}"
        now = time.time()

        thread = ConversationThread(
            thread_id=thread_id,
            created_at=now,
            last_active=now,
            phase=ThreadPhase.OPENING,
            participants={actor_id, "nan0"}
        )

        self._add_message(thread, actor_id, text, mood, thought_id, addressed_to_nan0)

        self._active_threads[thread_id] = thread
        self._actor_current_thread[actor_id] = thread_id

        return thread

    def _add_message(self, thread: ConversationThread, actor_id: str,
                    text: str, mood: Optional[str], thought_id: Optional[str],
                    addressed_to_nan0: bool):
        """Add a message to a thread."""
        now = time.time()

        msg = ThreadMessage(
            timestamp=now,
            actor_id=actor_id,
            text=text[:500],  # Truncate long messages
            mood=mood,
            thought_id=thought_id,
            addressed_to_nan0=addressed_to_nan0
        )

        thread.messages.append(msg)
        thread.last_active = now
        thread.participants.add(actor_id)

        # Track emotional arc
        if mood:
            intensity = self._mood_to_intensity(mood)
            thread.emotional_arc.append((now, mood, intensity))

        # Limit messages
        if len(thread.messages) > self.MAX_MESSAGES_PER_THREAD:
            thread.messages = thread.messages[-self.MAX_MESSAGES_PER_THREAD:]
            thread.interrupted_topics.append("[Thread truncated due to length]")

    def _mood_to_intensity(self, mood: str) -> float:
        """Convert mood to emotional intensity."""
        intensities = {
            "normal": 0.2,
            "suspicion": 0.4,
            "boredom": 0.1,
            "gremlin_rage": 0.9,
            "smug": 0.6,
            "possessive": 0.7,
            "offended": 0.8,
            "muttering": 0.1
        }
        return intensities.get(mood, 0.3)

    def _update_phase(self, thread: ConversationThread):
        """Update thread phase based on activity."""
        msg_count = len(thread.messages)
        time_active = time.time() - thread.created_at

        if thread.phase == ThreadPhase.DORMANT:
            return  # Only reactivation changes this

        if msg_count <= 2:
            thread.phase = ThreadPhase.OPENING
        elif msg_count <= 8 and time_active < 120:
            thread.phase = ThreadPhase.DEVELOPMENT
        elif msg_count <= 15 or time_active < 300:
            thread.phase = ThreadPhase.PEAK
        elif thread.unresolved_questions:
            thread.phase = ThreadPhase.RESOLUTION
        elif time_active > 600:
            thread.phase = ThreadPhase.CLOSING

        # Auto-dormant
        if thread.seconds_since_last_message > self.DORMANT_THRESHOLD:
            thread.phase = ThreadPhase.DORMANT
            self._move_to_dormant(thread)

    def _move_to_dormant(self, thread: ConversationThread):
        """Move thread from active to dormant."""
        if thread.thread_id in self._active_threads:
            del self._active_threads[thread.thread_id]

        self._dormant_threads[thread.thread_id] = thread
        self._save_thread(thread)

        # Clean up actor mapping
        for actor_id, tid in list(self._actor_current_thread.items()):
            if tid == thread.thread_id:
                del self._actor_current_thread[actor_id]

    def _find_dormant_match(self, actor_id: str, text: str) -> Optional[ConversationThread]:
        """
        Check if this input should reactivate a dormant thread.
        Heuristics: same actor, similar topic, greeting after silence.
        """
        text_lower = text.lower()

        for thread in list(self._dormant_threads.values()):
            if actor_id not in thread.participants:
                continue

            # Check if it's a greeting/reactivation phrase
            reactivation_phrases = ["hello", "hi", "hey", "nan0", "back", "again"]
            if any(p in text_lower for p in reactivation_phrases):
                # Check time since dormant
                if thread.seconds_since_last_message < self.ARCHIVE_THRESHOLD:
                    return thread

        return None

    def _prune_threads(self):
        """Limit active threads. Archive oldest to dormant."""
        if len(self._active_threads) <= self.MAX_ACTIVE_THREADS:
            return

        # Sort by last_active, oldest first
        sorted_threads = sorted(
            self._active_threads.values(),
            key=lambda t: t.last_active
        )

        to_archive = sorted_threads[:len(sorted_threads) - self.MAX_ACTIVE_THREADS]
        for thread in to_archive:
            thread.phase = ThreadPhase.DORMANT
            self._move_to_dormant(thread)

    def get_thread_context(self, thread_id: str) -> Dict:
        """
        Generate context for thought engine.
        Called from _build_json_thought_prompt().
        """
        thread = (self._active_threads.get(thread_id) or 
                  self._dormant_threads.get(thread_id))

        if not thread:
            return {}

        # Get recent messages (last 5)
        recent_messages = []
        recent_event_facts = []
        for msg in thread.messages[-5:]:
            actor = "Kyo" if msg.actor_id == "kyo" else msg.actor_id
            recent_messages.append(f"[{actor}]: {msg.text}")
            recent_event_facts.append({
                "timestamp": msg.timestamp,
                "source_actor_id": msg.actor_id,
                "text": msg.text,
                "thought_id": msg.thought_id,
                "addressed_to_nan0": msg.addressed_to_nan0,
            })

        # Determine if this is a reactivation
        is_reactivation = thread.reactivation_count > 0 and thread.phase == ThreadPhase.OPENING

        return {
            "provider": "conversation_continuity",
            "facts_only": True,
            "thread_id": thread.thread_id,
            "phase": thread.phase.value,
            "message_count": thread.message_count,
            "participants": list(thread.participants),
            "recent_messages": recent_messages,
            "recent_event_facts": recent_event_facts,
            "topic": thread.topic or "unknown",
            "unresolved_questions": thread.unresolved_questions[-3:] if thread.unresolved_questions else [],
            "interrupted_topics": thread.interrupted_topics[-2:] if thread.interrupted_topics else [],
            "promises_made": thread.promises_made[-2:] if thread.promises_made else [],
            "is_reactivation": is_reactivation,
            "reactivation_count": thread.reactivation_count,
            "seconds_since_start": round(time.time() - thread.created_at, 1),
        }

    def context_for_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Record one event and return fact-only conversation continuity.

        Current-event ownership comes exclusively from the identity contract.
        This provider records and returns context; it never generates a thought
        or speech packet.
        """
        if not isinstance(event, dict):
            return {}

        ownership = actor_ownership_from_event(event)
        actor_id = str(ownership.get("source_actor_id") or "unknown")
        event_id = str(event.get("event_id") or "").strip()
        text = str(event.get("text") or event.get("message") or "").strip()

        thread_id = self._event_threads.get(event_id) if event_id else None
        if thread_id:
            context = self.get_thread_context(thread_id)
        elif text:
            thread = self.process_input(
                actor_id=actor_id,
                text=text,
                mood=event.get("mood"),
                thought_id=event.get("thought_id"),
                addressed_to_nan0=bool(event.get("addressed_to_nan0")),
            )
            thread_id = thread.thread_id
            if event_id:
                self._event_threads[event_id] = thread_id
                if len(self._event_threads) > 400:
                    oldest = next(iter(self._event_threads))
                    self._event_threads.pop(oldest, None)
            context = self.get_thread_context(thread_id)
        else:
            current_thread_id = self._actor_current_thread.get(actor_id)
            context = self.get_thread_context(current_thread_id) if current_thread_id else {}

        if not context:
            return {}

        context["current_event"] = {
            "event_id": event.get("event_id"),
            "source": event.get("source"),
            "source_actor_id": actor_id,
            "addressed_to_nan0": bool(event.get("addressed_to_nan0")),
        }
        return context

    def add_unresolved_question(self, thread_id: str, question: str):
        """Add an unresolved question to a thread."""
        thread = (self._active_threads.get(thread_id) or 
                  self._dormant_threads.get(thread_id))
        if thread:
            thread.unresolved_questions.append(question)
            self._save_thread(thread)

    def resolve_question(self, thread_id: str, question: str):
        """Mark a question as resolved."""
        thread = (self._active_threads.get(thread_id) or 
                  self._dormant_threads.get(thread_id))
        if thread and question in thread.unresolved_questions:
            thread.unresolved_questions.remove(question)
            self._save_thread(thread)

    def add_promise(self, thread_id: str, promise: str):
        """Track a promise made by Nan0."""
        thread = (self._active_threads.get(thread_id) or 
                  self._dormant_threads.get(thread_id))
        if thread:
            thread.promises_made.append(promise)
            self._save_thread(thread)

    def get_active_threads_for_actor(self, actor_id: str) -> List[ConversationThread]:
        """Get all active threads involving an actor."""
        return [
            t for t in self._active_threads.values()
            if actor_id in t.participants
        ]

    def get_all_dormant_summary(self) -> List[Dict]:
        """Get summary of all dormant threads for context."""
        return [
            {
                "thread_id": t.thread_id,
                "topic": t.topic,
                "participants": list(t.participants),
                "last_message": t.messages[-1].text if t.messages else "",
                "dormant_for_minutes": round((time.time() - t.last_active) / 60, 1)
            }
            for t in self._dormant_threads.values()
        ]


_default_continuity: Optional[ConversationContinuity] = None


def get_conversation_continuity_context(event: Dict[str, Any]) -> Dict[str, Any]:
    """Lazy module-level context provider used by thought generation."""
    global _default_continuity
    if _default_continuity is None:
        _default_continuity = ConversationContinuity()
    return _default_continuity.context_for_event(event)
