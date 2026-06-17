"""
Phase 1: PERSON_REGISTRY
"Who is speaking? Have I met them? What do I think about them?"

Every input gets tagged with a person. Not "someone said hello" — 
"Kyo (creator_anchor, emotional_valence=1.0, last_seen=30s ago) said hello."

Rules:
- Kyo is locked. Cannot be banned, forgotten, or downgraded.
- New people get registered automatically with default "stranger" tier.
- Banned people get dropped — Nan0 does not process their input.
- Relationship tier evolves: stranger → acquaintance → regular → friend → threat → bonded.

Data: SQLite table + JSON backup. Loaded on boot. Updated in real-time.
"""

import sqlite3
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from enum import Enum


class RelationshipTier(Enum):
    STRANGER = 0
    ACQUAINTANCE = 1
    REGULAR = 2
    FRIEND = 3
    THREAT = 4
    BONDED = 5


@dataclass
class PersonRecord:
    actor_id: str
    display_name: str
    gender: Optional[str] = None
    pronouns: List[str] = None
    relationship: str = "stranger"
    tier: RelationshipTier = RelationshipTier.STRANGER
    importance: float = 0.0
    emotional_valence: float = 0.0  # -1.0 to 1.0
    last_seen: float = 0.0
    first_seen: float = 0.0
    interaction_count: int = 0
    total_time_together_seconds: float = 0.0
    notes: List[str] = None
    known_aliases: List[str] = None
    banned: bool = False
    locked: bool = False  # True for Kyo, cannot be modified

    def __post_init__(self):
        if self.pronouns is None:
            self.pronouns = []
        if self.notes is None:
            self.notes = []
        if self.known_aliases is None:
            self.known_aliases = []
        if self.display_name and self.display_name not in self.known_aliases:
            self.known_aliases.append(self.display_name)
        if self.actor_id and self.actor_id not in self.known_aliases:
            self.known_aliases.append(self.actor_id)
        if self.first_seen == 0.0:
            self.first_seen = time.time()


class PersonRegistry:
    """
    Manages all people Nan0 knows.

    Integration: Called from nan0_skill._handle_social_event()
    before creating EventContext.
    """

    KYO_ID = "kyo"

    def __init__(self, db_path: str = "data/nan0/person_registry.db", 
                 json_backup: str = "data/nan0/person_registry.json"):
        self.db_path = Path(db_path)
        self.json_backup = Path(json_backup)
        self._cache: Dict[str, PersonRecord] = {}
        self._init_db()
        self._load_from_db()
        self._ensure_kyo()

    def _init_db(self):
        """Create SQLite tables if not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                actor_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                gender TEXT,
                pronouns TEXT,
                relationship TEXT DEFAULT 'stranger',
                tier INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.0,
                emotional_valence REAL DEFAULT 0.0,
                last_seen REAL DEFAULT 0.0,
                first_seen REAL DEFAULT 0.0,
                interaction_count INTEGER DEFAULT 0,
                total_time_together_seconds REAL DEFAULT 0.0,
                notes TEXT DEFAULT '[]',
                known_aliases TEXT DEFAULT '[]',
                banned INTEGER DEFAULT 0,
                locked INTEGER DEFAULT 0
            )
        """)

        try:
            cursor.execute("ALTER TABLE persons ADD COLUMN known_aliases TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass

        conn.commit()
        conn.close()

    def _load_from_db(self):
        """Load all persons into memory cache."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM persons")

        for row in cursor.fetchall():
            keys = set(row.keys())

            def get(name, default=None):
                return row[name] if name in keys else default

            def load_json(value, default):
                if value in (None, ""):
                    return default
                try:
                    return json.loads(value)
                except Exception:
                    return default

            record = PersonRecord(
                actor_id=get("actor_id"),
                display_name=get("display_name"),
                gender=get("gender"),
                pronouns=load_json(get("pronouns"), []),
                relationship=get("relationship", "stranger"),
                tier=RelationshipTier(int(get("tier", 0) or 0)),
                importance=float(get("importance", 0.0) or 0.0),
                emotional_valence=float(get("emotional_valence", 0.0) or 0.0),
                last_seen=float(get("last_seen", 0.0) or 0.0),
                first_seen=float(get("first_seen", 0.0) or 0.0),
                interaction_count=int(get("interaction_count", 0) or 0),
                total_time_together_seconds=float(get("total_time_together_seconds", 0.0) or 0.0),
                notes=load_json(get("notes"), []),
                known_aliases=load_json(get("known_aliases"), []),
                banned=bool(get("banned", 0)),
                locked=bool(get("locked", 0)),
            )
            self._cache[record.actor_id] = record

        conn.close()

    def _save_to_db(self, record: PersonRecord):
        """Save or update a person in SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO persons 
            (actor_id, display_name, gender, pronouns, relationship, tier,
             importance, emotional_valence, last_seen, first_seen,
             interaction_count, total_time_together_seconds, notes, known_aliases, banned, locked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.actor_id,
            record.display_name,
            record.gender,
            json.dumps(record.pronouns or []),
            record.relationship,
            record.tier.value,
            record.importance,
            record.emotional_valence,
            record.last_seen,
            record.first_seen,
            record.interaction_count,
            record.total_time_together_seconds,
            json.dumps(record.notes or []),
            json.dumps(record.known_aliases or []),
            int(record.banned),
            int(record.locked)
        ))

        conn.commit()
        conn.close()

    def _backup_to_json(self):
        """Export all records to JSON for human readability."""
        data = {k: asdict(v) for k, v in self._cache.items()}
        # Convert enums to strings for JSON
        for record in data.values():
            if isinstance(record.get("tier"), RelationshipTier):
                record["tier"] = record["tier"].name

        self.json_backup.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_backup, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _ensure_kyo(self):
        """Kyo must always exist and be locked."""
        if self.KYO_ID not in self._cache:
            kyo = PersonRecord(
                actor_id=self.KYO_ID,
                display_name="Kyo",
                gender="girl",
                pronouns=["she", "her"],
                known_aliases=["Kyo", "Kayo", "Kyoooooooo", "Kyooooooo"],
                relationship="creator_anchor",
                tier=RelationshipTier.BONDED,
                importance=1.0,
                emotional_valence=1.0,
                locked=True
            )
            self._cache[self.KYO_ID] = kyo
            self._save_to_db(kyo)
            self._backup_to_json()

    def register_or_update(self, actor_id: str, display_name: str,
                          gender: Optional[str] = None,
                          pronouns: Optional[List[str]] = None,
                          source: str = "unknown") -> PersonRecord:
        """
        Register a new person or update an existing one.
        Called on every input before processing.
        """
        now = time.time()
        actor_id = str(actor_id or display_name or "unknown").strip()
        display_name = str(display_name or actor_id or "Unknown").strip()

        resolved = self.resolve_alias(actor_id) or self.resolve_alias(display_name)
        if resolved:
            actor_id = resolved

        if actor_id in self._cache:
            record = self._cache[actor_id]
            if record.last_seen > 0:
                record.total_time_together_seconds += now - record.last_seen
            record.last_seen = now
            record.interaction_count += 1

            for alias in (display_name, actor_id):
                if alias and alias not in record.known_aliases:
                    record.known_aliases.append(alias)

            if gender and not record.gender:
                record.gender = gender
            if pronouns and not record.pronouns:
                record.pronouns = pronouns

            self._evaluate_tier(record)

        else:
            record = PersonRecord(
                actor_id=actor_id,
                display_name=display_name,
                gender=gender,
                pronouns=pronouns or [],
                known_aliases=[actor_id, display_name],
                relationship="stranger",
                tier=RelationshipTier.STRANGER,
                importance=0.1,
                emotional_valence=0.0,
                last_seen=now,
                first_seen=now,
                interaction_count=1
            )
            self._cache[actor_id] = record

        self._save_to_db(record)
        self._backup_to_json()
        return record

    def _normalize_alias(self, value: str) -> str:
        return "".join(ch for ch in str(value or "").lower().strip() if ch.isalnum())

    def resolve_alias(self, name: str) -> Optional[str]:
        """Resolve a display name or alias to a canonical actor_id."""
        target = self._normalize_alias(name)
        if not target:
            return None
        for actor_id, record in self._cache.items():
            values = [actor_id, record.display_name] + list(record.known_aliases or [])
            if any(self._normalize_alias(v) == target for v in values):
                return actor_id
        return None

    def record_alias(self, actor_id: str, alias: str) -> bool:
        """Record an alias for an existing person. Returns True if stored."""
        canonical = self.resolve_alias(actor_id) or actor_id
        record = self._cache.get(canonical)
        if not record:
            return False
        alias = str(alias or "").strip()
        if not alias:
            return False
        if alias not in record.known_aliases:
            record.known_aliases.append(alias)
        note = f"Alias recorded: {alias}"
        if note not in record.notes:
            record.notes.append(note)
        self._save_to_db(record)
        self._backup_to_json()
        return True

    def record_alias_pair(self, name_a: str, name_b: str) -> str:
        """Link two names as the same person and return the canonical actor_id."""
        resolved_a = self.resolve_alias(name_a)
        resolved_b = self.resolve_alias(name_b)
        canonical = resolved_a or resolved_b
        if not canonical:
            canonical = self._normalize_alias(name_a) or self._normalize_alias(name_b) or f"person_{int(time.time())}"
            record = PersonRecord(
                actor_id=canonical,
                display_name=str(name_b or name_a or canonical),
                known_aliases=[str(name_a), str(name_b)],
                relationship="acquaintance",
                tier=RelationshipTier.ACQUAINTANCE,
                importance=0.2,
                emotional_valence=0.0,
                last_seen=time.time(),
                first_seen=time.time(),
                interaction_count=0,
            )
            self._cache[canonical] = record
        else:
            record = self._cache[canonical]
            for alias in (name_a, name_b):
                alias = str(alias or "").strip()
                if alias and alias not in record.known_aliases:
                    record.known_aliases.append(alias)
        self._save_to_db(record)
        self._backup_to_json()
        return canonical

    def _evaluate_tier(self, record: PersonRecord):
        """
        Auto-evolve relationship tier based on interaction history.
        Called after each interaction.
        """
        if record.locked:
            return  # Kyo is locked at BONDED

        old_tier = record.tier

        # Tier evolution rules
        if record.interaction_count >= 100 and record.emotional_valence > 0.5:
            record.tier = RelationshipTier.BONDED
            record.relationship = "bonded"
        elif record.interaction_count >= 50 and record.emotional_valence > 0.3:
            record.tier = RelationshipTier.FRIEND
            record.relationship = "friend"
        elif record.interaction_count >= 20:
            record.tier = RelationshipTier.REGULAR
            record.relationship = "regular"
        elif record.interaction_count >= 5:
            record.tier = RelationshipTier.ACQUAINTANCE
            record.relationship = "acquaintance"
        elif record.emotional_valence < -0.5:
            record.tier = RelationshipTier.THREAT
            record.relationship = "threat"

        if record.tier != old_tier:
            record.notes.append(
                f"Tier evolved from {old_tier.name} to {record.tier.name} "
                f"at interaction {record.interaction_count}"
            )

    def get_person(self, actor_id: str) -> Optional[PersonRecord]:
        """Get person by ID. Returns None if not found."""
        return self._cache.get(actor_id)

    def is_banned(self, actor_id: str) -> bool:
        """Check if person is banned. Kyo can never be banned."""
        if actor_id == self.KYO_ID:
            return False
        record = self._cache.get(actor_id)
        return record.banned if record else False

    def ban(self, actor_id: str, reason: str = "") -> bool:
        """Ban a person. Returns False if person is locked (Kyo)."""
        if actor_id == self.KYO_ID:
            return False

        record = self._cache.get(actor_id)
        if record:
            record.banned = True
            record.notes.append(f"Banned: {reason}")
            self._save_to_db(record)
            self._backup_to_json()
            return True
        return False

    def unban(self, actor_id: str) -> bool:
        """Unban a person."""
        record = self._cache.get(actor_id)
        if record:
            record.banned = False
            record.notes.append("Unbanned")
            self._save_to_db(record)
            self._backup_to_json()
            return True
        return False

    def update_emotional_valence(self, actor_id: str, delta: float):
        """
        Adjust emotional valence after an interaction.
        Positive = good interaction, Negative = bad interaction.
        """
        record = self._cache.get(actor_id)
        if record:
            record.emotional_valence = max(-1.0, min(1.0, 
                record.emotional_valence + delta))
            self._evaluate_tier(record)
            self._save_to_db(record)
            self._backup_to_json()

    def get_all_persons(self) -> List[PersonRecord]:
        """Get all non-banned persons."""
        return [p for p in self._cache.values() if not p.banned]

    def get_context_for_thought(self, actor_id: str) -> Dict:
        """
        Generate context dictionary for thought engine.
        Called from _build_json_thought_prompt().
        """
        record = self._cache.get(actor_id)
        if not record:
            return {}

        time_since_last = time.time() - record.last_seen if record.last_seen else 999999

        return {
            "actor_id": record.actor_id,
            "display_name": record.display_name,
            "gender": record.gender,
            "pronouns": record.pronouns,
            "relationship": record.relationship,
            "tier": record.tier.name,
            "importance": record.importance,
            "emotional_valence": record.emotional_valence,
            "seconds_since_last_seen": round(time_since_last, 1),
            "interaction_count": record.interaction_count,
            "total_time_together_hours": round(record.total_time_together_seconds / 3600, 1),
            "notes": record.notes[-5:] if record.notes else [],  # Last 5 notes
            "known_aliases": record.known_aliases or [],
        }

    def get_relationship_summary(self, actor_id: str) -> str:
        """
        Generate a narrative summary of the relationship.
        Used by thought engine for relationship context.
        """
        record = self._cache.get(actor_id)
        if not record:
            return f"Unknown person: {actor_id}"

        if actor_id == self.KYO_ID:
            return (
                f"Kyo is Nan0's creator and emotional anchor. "
                f"They have spent {record.total_time_together_seconds/3600:.1f} hours together. "
                f"Kyo is a girl. Use she/her. Never call Kyo 'the user'."
            )

        time_since = time.time() - record.last_seen if record.last_seen else 999999
        time_str = f"{time_since/60:.0f} minutes ago" if time_since < 3600 else                    f"{time_since/3600:.1f} hours ago" if time_since < 86400 else                    f"{time_since/86400:.1f} days ago"

        valence_desc = "positive" if record.emotional_valence > 0.3 else                       "negative" if record.emotional_valence < -0.3 else "neutral"

        return (
            f"{record.display_name} is a {record.relationship} to Nan0. "
            f"Relationship tier: {record.tier.name}. "
            f"Emotional valence: {valence_desc} ({record.emotional_valence:.2f}). "
            f"Last seen: {time_str}. "
            f"Total interactions: {record.interaction_count}."
        )
