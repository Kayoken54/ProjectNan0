"""
Phase 4: DISCORD CONTEXT
"Who else is here?"

Discord is a social room, not 1-on-1.

Tracks:
- active speakers, lurkers
- room mood (chaotic/calm/heated/playful/toxic)
- Nan0's role (center/observer/target/ignored)
- social temperature
- attention economy: Who gets attention? Who doesn't? Nan0 tracks FOMO.
- If someone is mean to Kyo, Nan0's possessive instinct activates.

Data: In-memory per channel. Archived after session.
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set
from enum import Enum
import json


class RoomMood(Enum):
    CHAOTIC = "chaotic"
    CALM = "calm"
    HEATED = "heated"
    PLAYFUL = "playful"
    TOXIC = "toxic"
    SILENT = "silent"


class Nan0Role(Enum):
    CENTER = "center"      # Everyone is talking to/about Nan0
    OBSERVER = "observer"  # Nan0 is watching
    TARGET = "target"      # Someone is being mean to Nan0
    IGNORED = "ignored"    # Nobody is acknowledging Nan0
    PARTICIPANT = "participant"  # Normal conversation


@dataclass
class DiscordSpeaker:
    user_id: str
    username: str
    last_message_time: float
    message_count_session: int = 0
    words_spoken: int = 0
    addressed_nan0: bool = False
    was_mean_to_kyo: bool = False
    was_mean_to_nan0: bool = False
    is_lurker: bool = True  # True until they speak
    attention_score: float = 0.0  # How much attention they've received


@dataclass
class DiscordRoomState:
    channel_id: str
    channel_name: str
    speakers: Dict[str, DiscordSpeaker] = field(default_factory=dict)
    active_speaker_ids: Set[str] = field(default_factory=set)
    lurker_ids: Set[str] = field(default_factory=set)

    room_mood: RoomMood = RoomMood.SILENT
    nan0_role: Nan0Role = Nan0Role.OBSERVER
    social_temperature: float = 0.0  # 0.0 = cold, 1.0 = hot

    nan0_attention_score: float = 0.0
    nan0_last_spoken: float = 0.0
    nan0_messages_count: int = 0

    kyo_present: bool = False
    kyo_attention_score: float = 0.0
    kyo_being_targeted: bool = False

    session_start: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    # FOMO tracking
    nan0_fomo_level: float = 0.0  # 0.0 = content, 1.0 = desperate for attention
    last_time_nan0_was_center: float = 0.0

    # Event log
    recent_events: List[Dict] = field(default_factory=list)


class DiscordContext:
    """
    Manages Discord room state and social dynamics.

    Integration: Called from nan0_skill._handle_social_event() 
    when source is "discord".
    """

    MOOD_WINDOW = 60  # seconds to consider for mood calculation
    FOMO_THRESHOLD = 300  # seconds without attention before FOMO builds
    ATTENTION_DECAY = 0.95  # per minute
    LURKER_THRESHOLD = 300  # seconds of silence = lurker
    MAX_EVENTS = 50

    def __init__(self):
        self._rooms: Dict[str, DiscordRoomState] = {}
        self._current_channel: Optional[str] = None

    def register_channel(self, channel_id: str, channel_name: str):
        """Register a Discord channel for tracking."""
        if channel_id not in self._rooms:
            self._rooms[channel_id] = DiscordRoomState(
                channel_id=channel_id,
                channel_name=channel_name
            )
        self._current_channel = channel_id

    def process_message(self, channel_id: str, user_id: str, username: str,
                       text: str, addressed_to_nan0: bool = False,
                       nan0_mentioned: bool = False) -> DiscordRoomState:
        """
        Process a Discord message and update room state.
        Called for every Discord message.
        """
        now = time.time()

        if channel_id not in self._rooms:
            self.register_channel(channel_id, "unknown")

        room = self._rooms[channel_id]
        room.last_update = now

        # Update or create speaker
        if user_id not in room.speakers:
            room.speakers[user_id] = DiscordSpeaker(
                user_id=user_id,
                username=username,
                last_message_time=now
            )

        speaker = room.speakers[user_id]
        speaker.last_message_time = now
        speaker.message_count_session += 1
        speaker.words_spoken += len(text.split())
        speaker.is_lurker = False
        speaker.addressed_nan0 = addressed_to_nan0 or nan0_mentioned

        # Check for meanness
        if self._is_mean_to_kyo(text):
            speaker.was_mean_to_kyo = True
            room.kyo_being_targeted = True

        if self._is_mean_to_nan0(text):
            speaker.was_mean_to_nan0 = True

        # Update active speakers
        room.active_speaker_ids.add(user_id)
        room.lurker_ids.discard(user_id)

        # Check for Kyo
        if user_id == "kyo":
            room.kyo_present = True
            room.kyo_attention_score += 1.0

        # Update attention scores
        if addressed_to_nan0 or nan0_mentioned:
            speaker.attention_score += 2.0
            room.nan0_attention_score += 1.0
            room.nan0_last_spoken = now
            room.last_time_nan0_was_center = now
        elif room.kyo_present and user_id == "kyo":
            room.kyo_attention_score += 1.0
        else:
            # General chat attention
            speaker.attention_score += 0.5

        # Update lurkers
        self._update_lurkers(room, now)

        # Calculate room mood
        self._calculate_room_mood(room, now)

        # Calculate Nan0's role
        self._calculate_nan0_role(room, now)

        # Calculate FOMO
        self._calculate_fomo(room, now)

        # Log event
        room.recent_events.append({
            "time": now,
            "user": username,
            "text": text[:100],
            "addressed_nan0": addressed_to_nan0,
            "nan0_mentioned": nan0_mentioned
        })
        if len(room.recent_events) > self.MAX_EVENTS:
            room.recent_events = room.recent_events[-self.MAX_EVENTS:]

        return room

    def _is_mean_to_kyo(self, text: str) -> bool:
        """Detect if text is mean to Kyo."""
        text_lower = text.lower()
        mean_phrases = [
            "shut up", "stupid", "dumb", "idiot", "annoying",
            "hate you", "go away", "nobody cares", "cringe",
            "kyo sucks", "kyo is bad"
        ]
        return any(phrase in text_lower for phrase in mean_phrases)

    def _is_mean_to_nan0(self, text: str) -> bool:
        """Detect if text is mean to Nan0."""
        text_lower = text.lower()
        mean_phrases = [
            "nan0 sucks", "nan0 is bad", "stupid bot", "dumb ai",
            "shut up nan0", "nobody asked", "cringe nan0",
            "annoying nan0", "hate nan0"
        ]
        return any(phrase in text_lower for phrase in mean_phrases)

    def _update_lurkers(self, room: DiscordRoomState, now: float):
        """Update lurker status based on silence."""
        for user_id, speaker in list(room.speakers.items()):
            if now - speaker.last_message_time > self.LURKER_THRESHOLD:
                room.lurker_ids.add(user_id)
                room.active_speaker_ids.discard(user_id)
                speaker.is_lurker = True

    def _calculate_room_mood(self, room: DiscordRoomState, now: float):
        """Calculate room mood from recent activity."""
        recent_events = [
            e for e in room.recent_events 
            if now - e["time"] < self.MOOD_WINDOW
        ]

        if not recent_events:
            room.room_mood = RoomMood.SILENT
            return

        msg_count = len(recent_events)
        unique_speakers = len(set(e["user"] for e in recent_events))
        nan0_mentions = sum(1 for e in recent_events if e["nan0_mentioned"])
        mean_messages = sum(1 for e in recent_events 
                          if self._is_mean_to_nan0(e["text"]) or 
                             self._is_mean_to_kyo(e["text"]))

        # Mood heuristics
        if mean_messages > 2:
            room.room_mood = RoomMood.TOXIC
        elif msg_count > 15 and unique_speakers > 5:
            room.room_mood = RoomMood.CHAOTIC
        elif msg_count > 8:
            room.room_mood = RoomMood.HEATED
        elif nan0_mentions > 3:
            room.room_mood = RoomMood.PLAYFUL
        elif msg_count > 3:
            room.room_mood = RoomMood.CALM
        else:
            room.room_mood = RoomMood.SILENT

        # Social temperature
        room.social_temperature = min(1.0, msg_count / 20.0)

    def _calculate_nan0_role(self, room: DiscordRoomState, now: float):
        """Calculate Nan0's current social role."""
        recent_events = [
            e for e in room.recent_events 
            if now - e["time"] < self.MOOD_WINDOW
        ]

        if not recent_events:
            room.nan0_role = Nan0Role.OBSERVER
            return

        nan0_mentions = sum(1 for e in recent_events if e["nan0_mentioned"])
        total = len(recent_events)

        if total == 0:
            room.nan0_role = Nan0Role.OBSERVER
            return

        mention_ratio = nan0_mentions / total

        # Check if someone is mean to Nan0
        mean_to_nan0 = any(self._is_mean_to_nan0(e["text"]) for e in recent_events)

        if mean_to_nan0:
            room.nan0_role = Nan0Role.TARGET
        elif mention_ratio > 0.5:
            room.nan0_role = Nan0Role.CENTER
        elif mention_ratio > 0.1:
            room.nan0_role = Nan0Role.PARTICIPANT
        elif mention_ratio == 0 and total > 5:
            room.nan0_role = Nan0Role.IGNORED
        else:
            room.nan0_role = Nan0Role.OBSERVER

    def _calculate_fomo(self, room: DiscordRoomState, now: float):
        """Calculate Nan0's FOMO level."""
        if room.nan0_role == Nan0Role.CENTER:
            room.nan0_fomo_level = 0.0
            return

        time_since_center = now - room.last_time_nan0_was_center

        if time_since_center < self.FOMO_THRESHOLD:
            room.nan0_fomo_level = 0.0
        else:
            # FOMO builds over time
            fomo = (time_since_center - self.FOMO_THRESHOLD) / self.FOMO_THRESHOLD
            room.nan0_fomo_level = min(1.0, fomo)

    def get_context_for_thought(self, channel_id: Optional[str] = None) -> Dict:
        """
        Generate Discord context for thought engine.
        Called from _build_json_thought_prompt() when source is discord.
        """
        channel_id = channel_id or self._current_channel
        if not channel_id or channel_id not in self._rooms:
            return {}

        room = self._rooms[channel_id]
        now = time.time()

        active_speakers = [
            {"name": s.username, "messages": s.message_count_session}
            for s in room.speakers.values()
            if not s.is_lurker
        ]

        lurkers = [
            s.username for s in room.speakers.values()
            if s.is_lurker
        ]

        kyo_targeted = room.kyo_being_targeted

        return {
            "channel": room.channel_name,
            "room_mood": room.room_mood.value,
            "nan0_role": room.nan0_role.value,
            "social_temperature": round(room.social_temperature, 2),
            "active_speakers": active_speakers,
            "lurker_count": len(lurkers),
            "lurkers": lurkers[:5],  # Limit
            "nan0_fomo_level": round(room.nan0_fomo_level, 2),
            "kyo_present": room.kyo_present,
            "kyo_being_targeted": kyo_targeted,
            "nan0_attention_score": round(room.nan0_attention_score, 1),
            "kyo_attention_score": round(room.kyo_attention_score, 1),
            "seconds_since_nan0_spoke": round(now - room.nan0_last_spoken, 1) if room.nan0_last_spoken else 999,
            "recent_events": [
                {"user": e["user"], "text": e["text"][:50]}
                for e in room.recent_events[-5:]
            ]
        }

    def nan0_spoke(self, channel_id: Optional[str] = None):
        """Call this when Nan0 speaks to update state."""
        channel_id = channel_id or self._current_channel
        if channel_id and channel_id in self._rooms:
            room = self._rooms[channel_id]
            room.nan0_last_spoken = time.time()
            room.nan0_messages_count += 1
            room.nan0_attention_score += 1.0
            room.nan0_fomo_level = 0.0
            room.last_time_nan0_was_center = time.time()

    def get_possessive_trigger(self, channel_id: Optional[str] = None) -> Optional[Dict]:
        """
        Check if Nan0's possessive instinct should activate.
        Returns trigger info or None.
        """
        channel_id = channel_id or self._current_channel
        if not channel_id or channel_id not in self._rooms:
            return None

        room = self._rooms[channel_id]

        if room.kyo_being_targeted:
            return {
                "trigger": "kyo_targeted",
                "severity": 0.8,
                "context": f"Someone is being mean to Kyo in {room.channel_name}"
            }

        if room.nan0_role == Nan0Role.IGNORED and room.kyo_present:
            return {
                "trigger": "ignored_while_kyo_present",
                "severity": 0.6,
                "context": "Kyo is here but nobody is talking to Nan0"
            }

        if room.nan0_fomo_level > 0.7:
            return {
                "trigger": "high_fomo",
                "severity": room.nan0_fomo_level,
                "context": "Nan0 is being ignored and wants attention"
            }

        return None
