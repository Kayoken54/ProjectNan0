"""In-memory session continuity timeline for Nan0.

This layer stores recent meaningful runtime events and exposes structured
continuity context to cognition. It does not generate thoughts or speech.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional


MAX_SESSION_EVENTS = 20

_LOW_VALUE_SOURCES = {"", "unknown"}
_LOW_VALUE_TEXT = {"", "none", "null"}


@dataclass
class TimelineItem:
    timestamp: float
    event_type: str
    actor: str
    summary: str
    raw_ref: Dict[str, Any] = field(default_factory=dict)
    significance: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = round(float(self.timestamp), 3)
        return data


class SessionTimeline:
    """Bounded, in-memory session timeline.

    The timeline is intentionally local and non-speaking. It stores only facts
    already present in event/thought metadata and repeat counts derived from
    those facts.
    """

    def __init__(self, max_events: int = MAX_SESSION_EVENTS):
        self.max_events = max(1, int(max_events))
        self._items: List[TimelineItem] = []
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def add_event(self, event: Dict[str, Any]) -> Optional[TimelineItem]:
        item = self._item_from_event(event)
        if item is None:
            return None
        return self.add_item(item)

    def add_thought_packet(self, packet: Dict[str, Any]) -> Optional[TimelineItem]:
        if not isinstance(packet, dict) or not packet.get("thought_id"):
            return None

        summary = str(packet.get("private_text") or packet.get("thought_text") or "").strip()
        if not summary:
            return None

        raw_ref = {
            "thought_id": packet.get("thought_id"),
            "event_id": packet.get("event_id"),
            "source": packet.get("source"),
            "thought_type": packet.get("thought_type"),
        }
        tags = self._topic_tags(
            packet.get("thought_type"),
            packet.get("target_actor_id") or packet.get("target_actor"),
            packet.get("source"),
            packet.get("mood"),
            *(packet.get("memory_context") or [])[:2] if isinstance(packet.get("memory_context"), list) else [],
        )
        significance = self._float_or_none(packet.get("speakability"))
        return self.add_item(
            TimelineItem(
                timestamp=float(packet.get("created_at") or time.time()),
                event_type=str(packet.get("thought_type") or "thought"),
                actor=str(packet.get("target_actor_id") or packet.get("target_actor") or packet.get("source") or "unknown"),
                summary=summary[:280],
                raw_ref=raw_ref,
                significance=significance,
                tags=tags,
            )
        )

    def add_item(self, item: TimelineItem) -> TimelineItem:
        with self._lock:
            self._items.append(item)
            if len(self._items) > self.max_events:
                self._items = self._items[-self.max_events :]
            return item

    def recent_items(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [item.to_dict() for item in self._items]

    def continuity_context(self) -> Dict[str, Any]:
        items = self.recent_items()
        event_types = Counter(item.get("event_type") for item in items if item.get("event_type"))
        actors = Counter(item.get("actor") for item in items if item.get("actor") and item.get("actor") != "unknown")
        tags = Counter(tag for item in items for tag in item.get("tags", []) if tag)

        repeat_facts = []
        repeat_facts.extend(self._facts(event_types, "event_type"))
        repeat_facts.extend(self._facts(actors, "actor"))
        repeat_facts.extend(self._facts(tags, "topic"))

        return {
            "recent_event_count": len(items),
            "max_event_count": self.max_events,
            "recent_events": items,
            "repeat_counts": {
                "event_type": dict(event_types),
                "actor": dict(actors),
                "topic": dict(tags),
            },
            "repeat_facts": repeat_facts,
            "recent_topics": [name for name, _count in tags.most_common(8)],
        }

    def _item_from_event(self, event: Dict[str, Any]) -> Optional[TimelineItem]:
        if not isinstance(event, dict):
            return None

        source = str(event.get("source") or event.get("event_type") or "unknown").strip()
        text = str(event.get("summary") or event.get("text") or event.get("message") or "").strip()
        actor = str(event.get("source_actor_id") or event.get("actor") or event.get("speaker") or source or "unknown").strip()
        event_type = str(event.get("event_type") or event.get("thought_seed") or event.get("screen_state") or source or "event").strip()

        if source.lower() in _LOW_VALUE_SOURCES and text.lower() in _LOW_VALUE_TEXT:
            return None
        if event.get("priority") == "low" and not self._has_meaningful_signal(event, text):
            return None

        significance = self._float_or_none(
            event.get("significance")
            or event.get("relevance")
            or event.get("speakability")
            or event.get("pressure")
        )
        tags = self._topic_tags(event_type, actor, source, *(event.get("tags") or [] if isinstance(event.get("tags"), list) else []))

        raw_ref = {
            key: event.get(key)
            for key in ("event_id", "source", "priority", "thought_id", "screen_state", "game_ui_detected")
            if event.get(key) is not None
        }

        return TimelineItem(
            timestamp=float(event.get("timestamp") or event.get("time") or time.time()),
            event_type=event_type or "event",
            actor=actor or "unknown",
            summary=text[:280] if text else event_type[:280],
            raw_ref=raw_ref,
            significance=significance,
            tags=tags,
        )

    def _has_meaningful_signal(self, event: Dict[str, Any], text: str) -> bool:
        if event.get("addressed_to_nan0") or event.get("thought_id"):
            return True
        if event.get("source") in {"kyo", "discord", "social_pressure", "vision_pressure", "monologue"}:
            return True
        if any(event.get(key) for key in ("combat", "menu_open", "dark_scene", "major_change")):
            return True
        return len(text.split()) >= 3

    def _topic_tags(self, *values: Any) -> List[str]:
        tags: List[str] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                tags.extend(self._topic_tags(*value))
                continue
            text = str(value).strip().lower()
            if not text:
                continue
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text):
                if token in {"the", "and", "for", "with", "that", "this", "unknown", "source", "event"}:
                    continue
                tags.append(token[:40])
        seen = set()
        compact = []
        for tag in tags:
            if tag not in seen:
                compact.append(tag)
                seen.add(tag)
        return compact[:12]

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        try:
            return round(float(value), 3)
        except Exception:
            return None

    @staticmethod
    def _facts(counter: Counter, label: str) -> List[Dict[str, Any]]:
        return [
            {"kind": label, "value": value, "count": count}
            for value, count in counter.most_common()
            if value and count >= 2
        ]


_timeline = SessionTimeline()


def get_session_timeline() -> SessionTimeline:
    return _timeline


def record_session_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _timeline.add_event(event)
    return item.to_dict() if item else None


def record_thought_packet(packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = _timeline.add_thought_packet(packet)
    return item.to_dict() if item else None


def get_continuity_context() -> Dict[str, Any]:
    return _timeline.continuity_context()


def reset_session_timeline() -> None:
    _timeline.clear()
