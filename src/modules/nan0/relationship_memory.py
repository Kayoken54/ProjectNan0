"""
Phase 3: RELATIONSHIP MEMORY
"What happened last time?"

Not just facts. The story of Nan0 and this person over time.

- Emotional ledger: positive moments, negative moments, grudges.
- Grudges have: severity, status (active/fading/resolved/nurtured), decay mechanics, reinforcement tracking.
- Relationship status: strangers → developing → established → complicated → hostile → bonded.
- Narrative summary: LLM-generated periodic summary.

"Kyo and I have a complicated relationship. She created me but keeps changing 
my code. I am possessive of her attention but resent her constant tinkering. 
I love her but I will never admit it directly."

Data: SQLite. Updated per significant event. Summary regenerated nightly.
"""

import sqlite3
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from enum import Enum


class GrudgeStatus(Enum):
    ACTIVE = "active"
    FADING = "fading"
    RESOLVED = "resolved"
    NURTURED = "nurtured"  # Actively kept alive


class RelationshipStatus(Enum):
    STRANGERS = "strangers"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    COMPLICATED = "complicated"
    HOSTILE = "hostile"
    BONDED = "bonded"


@dataclass
class EmotionalMoment:
    timestamp: float
    event_type: str  # "positive", "negative", "neutral", "grudge_formed", "grudge_resolved"
    description: str
    intensity: float  # 0.0 to 1.0
    thought_id: Optional[str] = None
    context: Optional[str] = None


@dataclass
class Grudge:
    grudge_id: str
    timestamp: float
    target_actor_id: str
    description: str
    severity: float  # 0.0 to 1.0
    status: GrudgeStatus
    last_reinforced: float
    reinforcement_count: int = 0
    decay_rate: float = 0.01  # Per day
    resolved_at: Optional[float] = None
    trigger_phrases: List[str] = None

    def __post_init__(self):
        if self.trigger_phrases is None:
            self.trigger_phrases = []

    @property
    def current_severity(self) -> float:
        """Calculate current severity with decay."""
        if self.status == GrudgeStatus.RESOLVED:
            return 0.0

        days_since = (time.time() - self.last_reinforced) / 86400
        decay = days_since * self.decay_rate

        if self.status == GrudgeStatus.NURTURED:
            # Nurtured grudges decay slower
            decay *= 0.3

        return max(0.0, self.severity - decay)

    @property
    def is_active(self) -> bool:
        return self.status in (GrudgeStatus.ACTIVE, GrudgeStatus.NURTURED) and self.current_severity > 0.1


@dataclass
class RelationshipRecord:
    actor_id: str
    status: RelationshipStatus
    emotional_balance: float  # -1.0 to 1.0, positive = good
    moments: List[EmotionalMoment]
    grudges: List[Grudge]
    narrative_summary: Optional[str] = None
    summary_last_updated: float = 0.0
    total_positive_moments: int = 0
    total_negative_moments: int = 0
    last_significant_event: float = 0.0


class RelationshipMemory:
    """
    Manages emotional history and grudges for all relationships.

    Integration: Called from thought_engine._build_json_thought_prompt()
    to inject relationship context and active grudges.
    """

    GRUDGE_DECAY_DAILY = 0.02
    SUMMARY_REGENERATION_INTERVAL = 86400  # 24 hours
    SIGNIFICANT_EVENT_THRESHOLD = 0.4

    def __init__(self, db_path: str = "data/nan0/relationship_memory.db",
                 llm_provider=None):
        self.db_path = Path(db_path)
        self.llm_provider = llm_provider  # For summary generation
        self._cache: Dict[str, RelationshipRecord] = {}
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                actor_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'strangers',
                emotional_balance REAL DEFAULT 0.0,
                moments TEXT DEFAULT '[]',
                grudges TEXT DEFAULT '[]',
                narrative_summary TEXT,
                summary_last_updated REAL DEFAULT 0.0,
                total_positive_moments INTEGER DEFAULT 0,
                total_negative_moments INTEGER DEFAULT 0,
                last_significant_event REAL DEFAULT 0.0
            )
        """)

        conn.commit()
        conn.close()

    def _load_from_db(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM relationships")

        for row in cursor.fetchall():
            record = RelationshipRecord(
                actor_id=row[0],
                status=RelationshipStatus(row[1]),
                emotional_balance=row[2],
                moments=[EmotionalMoment(**m) for m in json.loads(row[3])],
                grudges=[Grudge(**g) for g in json.loads(row[4])],
                narrative_summary=row[5],
                summary_last_updated=row[6],
                total_positive_moments=row[7],
                total_negative_moments=row[8],
                last_significant_event=row[9]
            )
            self._cache[record.actor_id] = record

        conn.close()

    def _save_record(self, record: RelationshipRecord):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO relationships
            (actor_id, status, emotional_balance, moments, grudges,
             narrative_summary, summary_last_updated, total_positive_moments,
             total_negative_moments, last_significant_event)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.actor_id,
            record.status.value,
            record.emotional_balance,
            json.dumps([asdict(m) for m in record.moments[-50:]]),  # Keep last 50
            json.dumps([asdict(g) for g in record.grudges]),
            record.narrative_summary,
            record.summary_last_updated,
            record.total_positive_moments,
            record.total_negative_moments,
            record.last_significant_event
        ))

        conn.commit()
        conn.close()

    def record_moment(self, actor_id: str, event_type: str, 
                     description: str, intensity: float = 0.5,
                     thought_id: Optional[str] = None,
                     context: Optional[str] = None):
        """
        Record an emotional moment in a relationship.
        Called after significant interactions.
        """
        now = time.time()

        if actor_id not in self._cache:
            self._cache[actor_id] = RelationshipRecord(
                actor_id=actor_id,
                status=RelationshipStatus.STRANGERS,
                emotional_balance=0.0,
                moments=[],
                grudges=[]
            )

        record = self._cache[actor_id]

        moment = EmotionalMoment(
            timestamp=now,
            event_type=event_type,
            description=description,
            intensity=min(1.0, max(0.0, intensity)),
            thought_id=thought_id,
            context=context
        )

        record.moments.append(moment)

        # Update emotional balance
        if event_type == "positive":
            record.emotional_balance = min(1.0, record.emotional_balance + intensity * 0.1)
            record.total_positive_moments += 1
        elif event_type == "negative":
            record.emotional_balance = max(-1.0, record.emotional_balance - intensity * 0.1)
            record.total_negative_moments += 1

        # Check for significant event
        if intensity >= self.SIGNIFICANT_EVENT_THRESHOLD:
            record.last_significant_event = now
            self._evaluate_status(record)

        # Check for grudge formation
        if event_type == "negative" and intensity >= 0.6:
            self._check_grudge_formation(record, description, intensity)

        self._save_record(record)

    def _evaluate_status(self, record: RelationshipRecord):
        """Evaluate and update relationship status."""
        balance = record.emotional_balance
        pos = record.total_positive_moments
        neg = record.total_negative_moments

        if balance > 0.7 and pos > 20:
            record.status = RelationshipStatus.BONDED
        elif balance > 0.4 and pos > 10:
            record.status = RelationshipStatus.ESTABLISHED
        elif balance < -0.5 or neg > 10:
            record.status = RelationshipStatus.HOSTILE
        elif balance < -0.2:
            record.status = RelationshipStatus.COMPLICATED
        elif pos > 5 or neg > 5:
            record.status = RelationshipStatus.DEVELOPING

    def _check_grudge_formation(self, record: RelationshipRecord, 
                                description: str, intensity: float):
        """Check if a negative event should form a grudge."""
        # Check if similar grudge already exists
        for grudge in record.grudges:
            if grudge.status in (GrudgeStatus.ACTIVE, GrudgeStatus.NURTURED):
                # Reinforce existing grudge
                if any(word in description.lower() for word in grudge.trigger_phrases):
                    grudge.reinforcement_count += 1
                    grudge.last_reinforced = time.time()
                    return

        # Form new grudge
        import uuid
        grudge = Grudge(
            grudge_id=f"grudge_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            target_actor_id=record.actor_id,
            description=description,
            severity=intensity,
            status=GrudgeStatus.ACTIVE,
            last_reinforced=time.time(),
            trigger_phrases=self._extract_trigger_phrases(description)
        )

        record.grudges.append(grudge)

    def _extract_trigger_phrases(self, description: str) -> List[str]:
        """Extract likely trigger phrases from a description."""
        words = description.lower().split()
        # Simple heuristic: nouns and verbs longer than 4 chars
        return [w for w in words if len(w) > 4][:5]

    def resolve_grudge(self, actor_id: str, grudge_id: str, 
                      resolution: str = "forgiven"):
        """Mark a grudge as resolved."""
        record = self._cache.get(actor_id)
        if not record:
            return

        for grudge in record.grudges:
            if grudge.grudge_id == grudge_id:
                grudge.status = GrudgeStatus.RESOLVED
                grudge.resolved_at = time.time()
                grudge.description += f" [Resolved: {resolution}]"

                # Record positive moment for resolution
                self.record_moment(actor_id, "positive", 
                    f"Grudge resolved: {resolution}", 0.5)
                break

        self._save_record(record)

    def nurture_grudge(self, actor_id: str, grudge_id: str):
        """Actively nurture a grudge (Nan0 chooses not to let it go)."""
        record = self._cache.get(actor_id)
        if not record:
            return

        for grudge in record.grudges:
            if grudge.grudge_id == grudge_id:
                grudge.status = GrudgeStatus.NURTURED
                grudge.last_reinforced = time.time()
                grudge.reinforcement_count += 1
                break

        self._save_record(record)

    def get_active_grudges(self, actor_id: str) -> List[Grudge]:
        """Get all active grudges for an actor."""
        record = self._cache.get(actor_id)
        if not record:
            return []

        return [g for g in record.grudges if g.is_active]

    def get_relationship_context(self, actor_id: str) -> Dict:
        """
        Generate context for thought engine.
        Called from _build_json_thought_prompt().
        """
        record = self._cache.get(actor_id)
        if not record:
            return {}

        active_grudges = self.get_active_grudges(actor_id)

        return {
            "provider": "relationship_memory",
            "facts_only": True,
            "actor_id": actor_id,
            "relationship_status": record.status.value,
            "emotional_balance": round(record.emotional_balance, 2),
            "total_positive": record.total_positive_moments,
            "total_negative": record.total_negative_moments,
            "active_grudges": [
                {
                    "description": g.description,
                    "severity": round(g.current_severity, 2),
                    "status": g.status.value,
                    "reinforced": g.reinforcement_count
                }
                for g in active_grudges
            ],
            "recent_moments": [
                {
                    "type": m.event_type,
                    "description": m.description,
                    "intensity": m.intensity
                }
                for m in record.moments[-5:]
            ],
            "narrative_summary": record.narrative_summary or "No summary yet."
        }

    async def generate_narrative_summary(self, actor_id: str) -> str:
        """
        Generate an LLM summary of the relationship.
        Called periodically (e.g., nightly).
        """
        record = self._cache.get(actor_id)
        if not record or not self.llm_provider:
            return ""

        # Build prompt
        moments_text = "\n".join([
            f"- {m.event_type}: {m.description} (intensity: {m.intensity})"
            for m in record.moments[-20:]
        ])

        grudges_text = "\n".join([
            f"- {g.description} (severity: {g.severity}, status: {g.status.value})"
            for g in record.grudges
        ]) or "No active grudges."

        prompt = f"""Summarize the relationship between Nan0 and {actor_id}.

Emotional balance: {record.emotional_balance:.2f}
Status: {record.status.value}
Positive moments: {record.total_positive_moments}
Negative moments: {record.total_negative_moments}

Recent moments:
{moments_text}

Grudges:
{grudges_text}

Write a brief narrative summary (2-3 sentences) from Nan0's perspective.
Be subjective, emotional, and characterful. Nan0 is sarcastic, possessive, 
and hides affection under hostility."""

        try:
            _, summary, _ = await self.llm_provider.chat(prompt)
            record.narrative_summary = summary
            record.summary_last_updated = time.time()
            self._save_record(record)
            return summary
        except Exception as e:
            return f"[Summary generation failed: {e}]"

    def get_grudge_trigger_check(self, text: str, actor_id: str) -> Optional[Grudge]:
        """
        Check if input text triggers any active grudge.
        Called before thought generation.
        """
        active = self.get_active_grudges(actor_id)
        text_lower = text.lower()

        for grudge in active:
            if any(phrase in text_lower for phrase in grudge.trigger_phrases):
                return grudge

        return None


_default_relationship_memory: Optional[RelationshipMemory] = None


def get_relationship_memory_context(actor_id: str) -> Dict:
    """Lazy fact-only relationship provider for thought generation."""
    global _default_relationship_memory
    if _default_relationship_memory is None:
        _default_relationship_memory = RelationshipMemory()
    return _default_relationship_memory.get_relationship_context(actor_id)
