"""
Phase 7: VISION EXPANSION
"What am I seeing? What does it mean?"

Current vision: screen capture → OCR → basic events.
Vision Expansion: screen → semantic understanding → environmental context → emotional interpretation.

Tracks:
- room lighting, time of day
- Kyo's posture (inferred from mouse movement patterns)
- how long she's been playing
- pattern deviations
- "Kyo has been playing for 4 hours on a Tuesday. This is unusual. 
   She might be avoiding something."

Data: In-memory per session. Archived to vision log.
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import json


class RoomLighting(Enum):
    BRIGHT = "bright"
    NORMAL = "normal"
    DIM = "dim"
    DARK = "dark"
    BACKLIT = "backlit"


class TimeOfDay(Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    DAWN = "dawn"


class ActivityPattern(Enum):
    NORMAL = "normal"
    EXTENDED = "extended"
    UNUSUAL_TIME = "unusual_time"
    BINGE = "binge"
    AVOIDANCE = "avoidance"
    FOCUSED = "focused"
    SCATTERED = "scattered"


@dataclass
class MousePattern:
    """Inferred from mouse movement data."""
    avg_speed: float = 0.0
    speed_variance: float = 0.0
    click_frequency: float = 0.0  # clicks per minute
    pause_frequency: float = 0.0    # long pauses per minute
    movement_jitter: float = 0.0   # erratic movement indicator

    # Inferred states
    inferred_posture: Optional[str] = None  # "leaning_forward", "relaxed", "tense"
    inferred_alertness: Optional[str] = None  # "focused", "tired", "distracted"
    inferred_emotional_state: Optional[str] = None  # "calm", "agitated", "excited"


@dataclass
class ScreenPattern:
    """Patterns detected from screen state over time."""
    app_switches_per_hour: float = 0.0
    tab_switches_per_hour: float = 0.0
    window_title_changes: List[Tuple[float, str]] = field(default_factory=list)
    dominant_app: Optional[str] = None
    task_switching_score: float = 0.0  # 0 = focused, 1 = scattered


@dataclass
class EnvironmentalContext:
    """Complete environmental understanding."""
    timestamp: float = field(default_factory=time.time)

    # Lighting
    screen_brightness_avg: float = 0.0
    screen_brightness_trend: str = "stable"  # "rising", "falling", "stable"
    room_lighting: RoomLighting = RoomLighting.NORMAL

    # Time
    time_of_day: TimeOfDay = TimeOfDay.AFTERNOON
    day_of_week: str = ""
    is_weekend: bool = False
    is_late_night: bool = False

    # Activity duration
    current_activity: Optional[str] = None
    activity_duration_seconds: float = 0.0
    session_total_duration_seconds: float = 0.0

    # Patterns
    mouse_pattern: MousePattern = field(default_factory=MousePattern)
    screen_pattern: ScreenPattern = field(default_factory=ScreenPattern)

    # Anomalies
    pattern_deviations: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0  # 0.0 = normal, 1.0 = highly unusual

    # Inferred emotional context
    inferred_kyo_state: Optional[str] = None
    inferred_situation: Optional[str] = None
    confidence: float = 0.0


class VisionExpansion:
    """
    Expands vision understanding beyond raw screen data.

    Integration: Called from nan0_vision_skill.py L3 interpretation.
    Enriches vision context with environmental understanding.
    """

    BRIGHTNESS_THRESHOLDS = {
        "dark": 20,
        "dim": 50,
        "normal": 150,
        "bright": 255
    }

    LATE_NIGHT_HOUR = 1  # 1 AM
    EARLY_MORNING_HOUR = 6  # 6 AM
    EXTENDED_SESSION_HOURS = 3
    BINGE_SESSION_HOURS = 6

    def __init__(self):
        self._history: List[EnvironmentalContext] = []
        self._current_context: Optional[EnvironmentalContext] = None
        self._session_start: float = time.time()
        self._activity_start: Optional[float] = None
        self._last_mouse_data: Optional[Dict] = None

        # Baseline patterns (learned over time)
        self._typical_play_times: List[int] = []  # hours of day
        self._typical_session_length: float = 0.0
        self._typical_brightness_range: Tuple[float, float] = (30, 200)

    def process_vision_state(self, vision_data: Dict) -> EnvironmentalContext:
        """
        Process vision stack state and create expanded environmental context.
        Called every vision tick.
        """
        now = time.time()

        context = EnvironmentalContext(timestamp=now)

        # Extract brightness
        layer1 = vision_data.get("layer1_reflex", {})
        brightness = layer1.get("brightness", 100)
        context.screen_brightness_avg = brightness

        # Determine room lighting
        context.room_lighting = self._classify_lighting(brightness)

        # Determine time
        context.time_of_day = self._classify_time_of_day()
        context.day_of_week = time.strftime("%A")
        context.is_weekend = time.strftime("%w") in ("0", "6")
        context.is_late_night = self._is_late_night()

        # Activity tracking
        layer2 = vision_data.get("layer2_semantic", {})
        activity = layer2.get("activity", "unknown")

        if self._current_context and self._current_context.current_activity == activity:
            context.activity_duration_seconds = self._current_context.activity_duration_seconds + (now - self._current_context.timestamp)
        else:
            context.activity_duration_seconds = 0
            if self._current_context:
                self._analyze_activity_transition(self._current_context.current_activity, activity)

        context.current_activity = activity
        context.session_total_duration_seconds = now - self._session_start

        # Mouse patterns (would need actual mouse tracking data)
        context.mouse_pattern = self._analyze_mouse_patterns(vision_data)

        # Screen patterns
        context.screen_pattern = self._analyze_screen_patterns(vision_data)

        # Detect anomalies
        context.pattern_deviations = self._detect_anomalies(context)
        context.anomaly_score = len(context.pattern_deviations) * 0.2

        # Infer emotional context
        context.inferred_kyo_state, context.inferred_situation, context.confidence =             self._infer_emotional_context(context, vision_data)

        # Store
        self._current_context = context
        self._history.append(context)

        # Prune history
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        return context

    def _classify_lighting(self, brightness: float) -> RoomLighting:
        if brightness < self.BRIGHTNESS_THRESHOLDS["dark"]:
            return RoomLighting.DARK
        elif brightness < self.BRIGHTNESS_THRESHOLDS["dim"]:
            return RoomLighting.DIM
        elif brightness > self.BRIGHTNESS_THRESHOLDS["bright"]:
            return RoomLighting.BRIGHT
        else:
            return RoomLighting.NORMAL

    def _classify_time_of_day(self) -> TimeOfDay:
        hour = int(time.strftime("%H"))
        if 5 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        elif 21 <= hour or hour < 5:
            return TimeOfDay.NIGHT
        return TimeOfDay.DAWN

    def _is_late_night(self) -> bool:
        hour = int(time.strftime("%H"))
        return hour < self.EARLY_MORNING_HOUR or hour >= 23

    def _analyze_mouse_patterns(self, vision_data: Dict) -> MousePattern:
        """Analyze mouse patterns from vision data or external tracker."""
        pattern = MousePattern()

        # Would integrate with actual mouse tracking
        # For now, infer from screen state changes
        layer1 = vision_data.get("layer1_reflex", {})

        if layer1.get("motion_intensity", 0) > 0.5:
            pattern.inferred_posture = "leaning_forward"
            pattern.inferred_alertness = "focused"
        elif layer1.get("motion_intensity", 0) < 0.1:
            pattern.inferred_posture = "relaxed"
            pattern.inferred_alertness = "tired"
        else:
            pattern.inferred_posture = "normal"
            pattern.inferred_alertness = "normal"

        # Infer emotional state from activity
        layer2 = vision_data.get("layer2_semantic", {})
        tone = layer2.get("tone", "neutral")

        if tone == "stressful":
            pattern.inferred_emotional_state = "agitated"
        elif tone == "calm":
            pattern.inferred_emotional_state = "calm"
        elif tone == "excited":
            pattern.inferred_emotional_state = "excited"
        else:
            pattern.inferred_emotional_state = "neutral"

        return pattern

    def _analyze_screen_patterns(self, vision_data: Dict) -> ScreenPattern:
        """Analyze screen usage patterns."""
        pattern = ScreenPattern()

        # Track app/window changes
        # Would need actual window title tracking
        # Placeholder based on game detection
        game_ui = vision_data.get("layer1_reflex", {}).get("game_ui_detected", "unknown")

        if game_ui != "unknown":
            pattern.dominant_app = game_ui

        return pattern

    def _detect_anomalies(self, context: EnvironmentalContext) -> List[str]:
        """Detect deviations from typical patterns."""
        deviations = []

        # Late night gaming
        if context.is_late_night and context.current_activity and "game" in context.current_activity.lower():
            deviations.append("playing_late_night")

        # Extended session
        hours = context.session_total_duration_seconds / 3600
        if hours > self.EXTENDED_SESSION_HOURS:
            if hours > self.BINGE_SESSION_HOURS:
                deviations.append("binge_session")
            else:
                deviations.append("extended_session")

        # Unusual time
        current_hour = int(time.strftime("%H"))
        if self._typical_play_times and current_hour not in self._typical_play_times:
            deviations.append("unusual_play_time")

        # Dark room + extended session
        if context.room_lighting in (RoomLighting.DARK, RoomLighting.DIM) and hours > 2:
            deviations.append("dark_room_extended")

        # Weekend anomaly
        if not context.is_weekend and hours > 4:
            deviations.append("long_session_weekday")

        # Low brightness during day
        if context.time_of_day in (TimeOfDay.MORNING, TimeOfDay.AFTERNOON) and context.screen_brightness_avg < 50:
            deviations.append("dark_during_day")

        return deviations

    def _infer_emotional_context(self, context: EnvironmentalContext, 
                                  vision_data: Dict) -> Tuple[Optional[str], Optional[str], float]:
        """Infer Kyo's emotional state from environmental cues."""

        # Build inference from multiple signals
        signals = []
        confidence = 0.0

        # Signal: Late night + extended session
        if context.is_late_night and context.session_total_duration_seconds > 4 * 3600:
            signals.append("avoidance_or_obsession")
            confidence += 0.3

        # Signal: Dark room + alone
        if context.room_lighting == RoomLighting.DARK and len(context.screen_pattern.window_title_changes) < 2:
            signals.append("isolated_or_focused")
            confidence += 0.2

        # Signal: High activity stress
        layer2 = vision_data.get("layer2_semantic", {})
        if layer2.get("tone") == "stressful":
            signals.append("stressed_or_challenged")
            confidence += 0.25

        # Signal: Unusual time
        if "unusual_play_time" in context.pattern_deviations:
            signals.append("disrupted_schedule")
            confidence += 0.15

        # Signal: Binge session
        if "binge_session" in context.pattern_deviations:
            signals.append("escapism_or_hyperfocus")
            confidence += 0.2

        # Determine primary inference
        if not signals:
            return "normal", "typical session", 0.1

        # Map signals to narrative
        if "avoidance_or_obsession" in signals and "escapism_or_hyperfocus" in signals:
            return "avoidant", "might be avoiding something difficult", min(0.8, confidence)

        if "stressed_or_challenged" in signals:
            return "stressed", "pushing through difficulty", min(0.7, confidence)

        if "isolated_or_focused" in signals:
            return "focused", "deep in concentration", min(0.6, confidence)

        if "disrupted_schedule" in signals:
            return "disrupted", "routine is off", min(0.5, confidence)

        return "uncertain", "mixed signals", min(0.4, confidence)

    def _analyze_activity_transition(self, old_activity: Optional[str], new_activity: str):
        """Analyze what an activity transition means."""
        # Could trigger specific thoughts about why Kyo switched activities
        pass

    def get_context_for_thought(self) -> Dict:
        """
        Generate vision expansion context for thought engine.
        Called from _build_json_thought_prompt().
        """
        if not self._current_context:
            return {}

        ctx = self._current_context

        return {
            "environment": {
                "lighting": ctx.room_lighting.value,
                "time_of_day": ctx.time_of_day.value,
                "day_of_week": ctx.day_of_week,
                "is_weekend": ctx.is_weekend,
                "is_late_night": ctx.is_late_night
            },
            "activity": {
                "current": ctx.current_activity,
                "duration_minutes": round(ctx.activity_duration_seconds / 60, 1),
                "session_duration_hours": round(ctx.session_total_duration_seconds / 3600, 1)
            },
            "patterns": {
                "mouse_posture": ctx.mouse_pattern.inferred_posture,
                "alertness": ctx.mouse_pattern.inferred_alertness,
                "emotional_state": ctx.mouse_pattern.inferred_emotional_state,
                "task_switching": round(ctx.screen_pattern.task_switching_score, 2)
            },
            "anomalies": {
                "deviations": ctx.pattern_deviations,
                "anomaly_score": round(ctx.anomaly_score, 2)
            },
            "inferred": {
                "kyo_state": ctx.inferred_kyo_state,
                "situation": ctx.inferred_situation,
                "confidence": round(ctx.confidence, 2)
            }
        }

    def get_deep_vision_prompt(self) -> Optional[str]:
        """
        Generate a prompt for Deep Vision (Phase 8) based on current context.
        Returns None if no significant inference.
        """
        if not self._current_context:
            return None

        ctx = self._current_context

        if ctx.anomaly_score < 0.3 and ctx.confidence < 0.5:
            return None

        # Build deep vision prompt
        parts = []

        if ctx.inferred_kyo_state and ctx.confidence > 0.4:
            parts.append(f"Kyo seems {ctx.inferred_kyo_state}.")

        if ctx.inferred_situation and ctx.confidence > 0.4:
            parts.append(f"Situation: {ctx.inferred_situation}")

        if ctx.pattern_deviations:
            parts.append(f"Anomalies detected: {', '.join(ctx.pattern_deviations)}")

        if ctx.is_late_night and ctx.session_total_duration_seconds > 4 * 3600:
            parts.append("It's late. Kyo has been here for hours.")

        if ctx.room_lighting in (RoomLighting.DARK, RoomLighting.DIM):
            parts.append("The room is dark.")

        if not parts:
            return None

        return " ".join(parts)
