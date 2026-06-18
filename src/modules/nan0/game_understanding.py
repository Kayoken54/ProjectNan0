"""
Phase 6: GAME UNDERSTANDING
"What does this game MEAN to Kyo?"

Not mechanics. Narrative. Emotional significance. Patterns.

Detects narrative arcs:
- comeback story
- domination
- struggle
- learning

Tracks Kyo's tells:
- "gets quiet when frustrated"
- "laughs when nervous"
- "overextends when confident"
- "switches class when tilted"

Commentary strategy adapts:
- hype | roast | coach | narrator | silent

Data: SQLite per game. Updated per session. Pattern detection over time.
"""

import sqlite3
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum


class NarrativeArc(Enum):
    COMEBACK = "comeback"           # Losing then winning
    DOMINATION = "domination"       # Consistently winning
    STRUGGLE = "struggle"           # Consistently losing but trying
    LEARNING = "learning"           # Improving over session
    TILT = "tilt"                   # Getting progressively worse
    CHAOS = "chaos"                 # Unpredictable, wild swings
    GRIND = "grind"                 # Repetitive, mechanical
    STORY = "story"                 # Memorable sequence of events
    NONE = "none"


class CommentaryStrategy(Enum):
    HYPE = "hype"           # Encourage, celebrate
    ROAST = "roast"         # Mock playfully
    COACH = "coach"         # Give advice
    NARRATOR = "narrator"   # Describe what's happening
    SILENT = "silent"       # Say nothing, just watch
    EMPATHY = "empathy"     # Comfort, understand
    TERRITORIAL = "territorial"  # Protect Kyo from others


class KyoTells(Enum):
    QUIET_WHEN_FRUSTRATED = "quiet_when_frustrated"
    LAUGHS_WHEN_NERVOUS = "laughs_when_nervous"
    OVEREXTENDS_CONFIDENT = "overextends_when_confident"
    SWITCHES_CLASS_TILTED = "switches_class_when_tilted"
    PLAYS_AGGRESSIVE_TIRED = "plays_aggressive_when_tired"
    DEFENSIVE_WHEN_SAD = "defensive_when_sad"
    CHATTY_WHEN_HAPPY = "chatty_when_happy"
    SILENT_WHEN_FOCUSING = "silent_when_focusing"


@dataclass
class GameSession:
    session_id: str
    game_type: str
    start_time: float
    end_time: Optional[float] = None

    # Performance metrics
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    score: int = 0
    win: bool = False

    # Temporal patterns
    performance_over_time: List[Tuple[float, float]] = field(default_factory=list)  # (time, performance_score)
    mood_over_time: List[Tuple[float, str]] = field(default_factory=list)  # (time, mood)

    # Detected tells
    tells_observed: List[Tuple[float, KyoTells, str]] = field(default_factory=list)  # (time, tell, context)

    # Narrative
    narrative_arc: NarrativeArc = NarrativeArc.NONE
    arc_confidence: float = 0.0

    # Commentary strategy used
    primary_strategy: CommentaryStrategy = CommentaryStrategy.NARRATOR
    strategy_switches: List[Tuple[float, CommentaryStrategy, str]] = field(default_factory=list)

    # Key moments
    key_moments: List[Dict] = field(default_factory=list)

    # Nan0's commentary
    nan0_comments: List[Dict] = field(default_factory=list)


@dataclass
class KyoProfile:
    """Persistent profile of Kyo's gaming behavior."""
    total_sessions: int = 0
    total_hours: float = 0.0

    # Tell frequencies
    tell_frequency: Dict[str, int] = field(default_factory=dict)
    tell_confidence: Dict[str, float] = field(default_factory=dict)

    # Preferred strategies
    strategy_effectiveness: Dict[str, List[float]] = field(default_factory=dict)

    # Emotional patterns
    typical_mood_trajectory: List[str] = field(default_factory=list)

    # Game-specific
    game_preferences: Dict[str, int] = field(default_factory=dict)  # game -> hours
    game_skill_estimate: Dict[str, float] = field(default_factory=dict)  # game -> skill 0-1

    # Time patterns
    typical_play_times: List[int] = field(default_factory=list)  # hours of day
    session_length_avg: float = 0.0

    last_updated: float = field(default_factory=time.time)


class GameUnderstanding:
    """
    Understands games narratively and tracks Kyo's behavioral patterns.

    Integration: Called after game sessions to update understanding.
    Feeds into thought_engine for commentary strategy selection.
    """

    TELL_DETECTION_THRESHOLD = 3  # Observations before confident
    PERFORMANCE_WINDOW = 300  # seconds for rolling performance

    def __init__(self, db_path: str = "data/nan0/game_understanding.db"):
        self.db_path = Path(db_path)
        self._current_session: Optional[GameSession] = None
        self._kyo_profile: KyoProfile = KyoProfile()
        self._init_db()
        self._load_profile()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                game_type TEXT,
                start_time REAL,
                end_time REAL,
                kills INTEGER,
                deaths INTEGER,
                assists INTEGER,
                score INTEGER,
                win INTEGER,
                performance_over_time TEXT,
                mood_over_time TEXT,
                tells_observed TEXT,
                narrative_arc TEXT,
                arc_confidence REAL,
                primary_strategy TEXT,
                strategy_switches TEXT,
                key_moments TEXT,
                nan0_comments TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kyo_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                profile TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_profile(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT profile FROM kyo_profile WHERE id = 1")
        row = cursor.fetchone()

        if row and row[0]:
            data = json.loads(row[0])
            self._kyo_profile = KyoProfile(**data)

        conn.close()

    def _save_profile(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO kyo_profile (id, profile)
            VALUES (1, ?)
        """, (json.dumps(self._kyo_profile.__dict__),))

        conn.commit()
        conn.close()

    def start_session(self, game_type: str) -> GameSession:
        """Start tracking a new game session."""
        import uuid
        session = GameSession(
            session_id=f"game_{uuid.uuid4().hex[:12]}",
            game_type=game_type,
            start_time=time.time()
        )
        self._current_session = session
        return session

    def record_performance(self, timestamp: float, metric: str, value: float):
        """Record a performance metric at a point in time."""
        if not self._current_session:
            return

        # Normalize to 0-1 performance score
        perf_score = self._normalize_performance(metric, value)
        self._current_session.performance_over_time.append((timestamp, perf_score))

    def record_mood(self, timestamp: float, mood: str):
        """Record Kyo's apparent mood at a point in time."""
        if not self._current_session:
            return

        self._current_session.mood_over_time.append((timestamp, mood))

    def detect_tell(self, timestamp: float, tell: KyoTells, context: str):
        """Record observation of a Kyo tell."""
        if not self._current_session:
            return

        self._current_session.tells_observed.append((timestamp, tell, context))

        # Update profile
        tell_key = tell.value
        self._kyo_profile.tell_frequency[tell_key] =             self._kyo_profile.tell_frequency.get(tell_key, 0) + 1

    def record_commentary(self, timestamp: float, strategy: CommentaryStrategy,
                         comment: str, context: str):
        """Record Nan0's commentary and strategy."""
        if not self._current_session:
            return

        self._current_session.nan0_comments.append({
            "time": timestamp,
            "strategy": strategy.value,
            "comment": comment,
            "context": context
        })

        # Track strategy switch
        if (not self._current_session.strategy_switches or 
            self._current_session.strategy_switches[-1][1] != strategy):
            self._current_session.strategy_switches.append((timestamp, strategy, context))

    def end_session(self, win: bool = False, final_score: int = 0):
        """End current session and analyze."""
        if not self._current_session:
            return

        session = self._current_session
        session.end_time = time.time()
        session.win = win
        session.score = final_score

        # Analyze narrative arc
        session.narrative_arc, session.arc_confidence =             self._detect_narrative_arc(session)

        # Determine primary strategy
        if session.strategy_switches:
            strategy_counts = {}
            for _, strategy, _ in session.strategy_switches:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            session.primary_strategy = max(strategy_counts, key=strategy_counts.get)

        # Update Kyo profile
        self._update_kyo_profile(session)

        # Save session
        self._save_session(session)
        self._save_profile()

        self._current_session = None
        return session

    def _normalize_performance(self, metric: str, value: float) -> float:
        """Normalize performance metric to 0-1."""
        # Game-specific normalization
        if metric == "kd_ratio":
            return min(1.0, value / 5.0)  # Cap at 5.0 K/D
        elif metric == "score_per_min":
            return min(1.0, value / 1000.0)
        elif metric == "accuracy":
            return value  # Already 0-1
        elif metric == "win_probability":
            return value
        return 0.5

    def _detect_narrative_arc(self, session: GameSession) -> Tuple[NarrativeArc, float]:
        """Detect narrative arc from session data."""
        if len(session.performance_over_time) < 3:
            return NarrativeArc.NONE, 0.0

        # Get performance trend
        performances = [p[1] for p in session.performance_over_time]

        # Calculate trend
        first_third = sum(performances[:len(performances)//3]) / max(1, len(performances)//3)
        last_third = sum(performances[-len(performances)//3:]) / max(1, len(performances)//3)

        # Detect tells
        has_tilt = any(t[1] == KyoTells.SWITCHES_CLASS_TILTED 
                      for t in session.tells_observed)
        has_struggle = any(t[1] == KyoTells.QUIET_WHEN_FRUSTRATED 
                          for t in session.tells_observed)

        # Arc detection
        if has_tilt and last_third < first_third:
            return NarrativeArc.TILT, 0.8
        elif first_third < 0.3 and last_third > 0.7:
            return NarrativeArc.COMEBACK, 0.9
        elif first_third > 0.7 and last_third > 0.7:
            return NarrativeArc.DOMINATION, 0.8
        elif first_third < 0.3 and last_third < 0.3 and has_struggle:
            return NarrativeArc.STRUGGLE, 0.7
        elif first_third < 0.4 and last_third > 0.5:
            return NarrativeArc.LEARNING, 0.6
        elif max(performances) - min(performances) > 0.5:
            return NarrativeArc.CHAOS, 0.6

        return NarrativeArc.GRIND, 0.4

    def _update_kyo_profile(self, session: GameSession):
        """Update persistent Kyo profile from session."""
        kp = self._kyo_profile

        kp.total_sessions += 1
        if session.end_time:
            kp.total_hours += (session.end_time - session.start_time) / 3600

        # Game preferences
        kp.game_preferences[session.game_type] =             kp.game_preferences.get(session.game_type, 0) + 1

        # Calculate tell confidence
        for tell_key, count in kp.tell_frequency.items():
            kp.tell_confidence[tell_key] = min(1.0, count / self.TELL_DETECTION_THRESHOLD)

        # Update session length average
        if session.end_time:
            session_length = (session.end_time - session.start_time) / 3600
            if kp.session_length_avg == 0:
                kp.session_length_avg = session_length
            else:
                kp.session_length_avg = (kp.session_length_avg * (kp.total_sessions - 1) + 
                                        session_length) / kp.total_sessions

        # Time patterns
        hour = int(time.strftime("%H", time.localtime(session.start_time)))
        if hour not in kp.typical_play_times:
            kp.typical_play_times.append(hour)

        kp.last_updated = time.time()

    def _save_session(self, session: GameSession):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO sessions
            (session_id, game_type, start_time, end_time, kills, deaths,
             assists, score, win, performance_over_time, mood_over_time,
             tells_observed, narrative_arc, arc_confidence, primary_strategy,
             strategy_switches, key_moments, nan0_comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.game_type,
            session.start_time,
            session.end_time,
            session.kills,
            session.deaths,
            session.assists,
            session.score,
            int(session.win),
            json.dumps(session.performance_over_time),
            json.dumps(session.mood_over_time),
            json.dumps([(t[0], t[1].value, t[2]) for t in session.tells_observed]),
            session.narrative_arc.value,
            session.arc_confidence,
            session.primary_strategy.value,
            json.dumps([(t[0], t[1].value, t[2]) for t in session.strategy_switches]),
            json.dumps(session.key_moments),
            json.dumps(session.nan0_comments)
        ))

        conn.commit()
        conn.close()

    def get_commentary_strategy(self, game_state: Dict) -> CommentaryStrategy:
        """
        Determine best commentary strategy for current moment.
        Called before generating game commentary.
        """
        if not self._current_session:
            return CommentaryStrategy.NARRATOR

        session = self._current_session

        # High intensity moments -> HYPE or TERRITORIAL
        if game_state.get("is_high_intensity"):
            if game_state.get("player", {}).get("danger_level") == "critical":
                return CommentaryStrategy.HYPE
            return CommentaryStrategy.TERRITORIAL

        # Detected tilt -> EMPATHY or SILENT
        if any(t[1] == KyoTells.SWITCHES_CLASS_TILTED 
               for t in session.tells_observed[-3:]):
            return CommentaryStrategy.EMPATHY

        # Kyo doing well -> ROAST (playful)
        if session.narrative_arc == NarrativeArc.DOMINATION:
            return CommentaryStrategy.ROAST

        # Learning -> COACH
        if session.narrative_arc == NarrativeArc.LEARNING:
            return CommentaryStrategy.COACH

        # Struggle -> EMPATHY
        if session.narrative_arc == NarrativeArc.STRUGGLE:
            return CommentaryStrategy.EMPATHY

        # Default
        return CommentaryStrategy.NARRATOR

    def get_context_for_thought(self) -> Dict:
        """
        Generate game understanding context for thought engine.
        """
        if not self._current_session:
            return {}

        session = self._current_session
        kp = self._kyo_profile

        # Get confident tells
        confident_tells = [
            {"tell": tell, "confidence": conf}
            for tell, conf in kp.tell_confidence.items()
            if conf >= 0.5
        ]

        return {
            "current_session": {
                "game": session.game_type,
                "duration_minutes": round((time.time() - session.start_time) / 60, 1) if not session.end_time else 
                                   round((session.end_time - session.start_time) / 60, 1),
                "narrative_arc": session.narrative_arc.value,
                "arc_confidence": round(session.arc_confidence, 2),
                "primary_strategy": session.primary_strategy.value,
                "tells_observed": [t[1].value for t in session.tells_observed[-5:]]
            },
            "kyo_profile": {
                "total_sessions": kp.total_sessions,
                "total_hours": round(kp.total_hours, 1),
                "confident_tells": confident_tells,
                "typical_play_times": kp.typical_play_times,
                "avg_session_hours": round(kp.session_length_avg, 1)
            }
        }

    def get_narrative_summary(self) -> str:
        """Get a narrative summary of current session for thought engine."""
        if not self._current_session:
            return "No active game session."

        session = self._current_session

        arc_desc = {
            NarrativeArc.COMEBACK: "a comeback story",
            NarrativeArc.DOMINATION: "a domination",
            NarrativeArc.STRUGGLE: "a struggle",
            NarrativeArc.LEARNING: "a learning session",
            NarrativeArc.TILT: "a tilt spiral",
            NarrativeArc.CHAOS: "chaos",
            NarrativeArc.GRIND: "a grind",
            NarrativeArc.STORY: "something memorable",
            NarrativeArc.NONE: "an unremarkable session"
        }

        return (
            f"This session is shaping up as {arc_desc.get(session.narrative_arc, 'unknown')}. "
            f"Kyo has shown {len(session.tells_observed)} behavioral tells. "
            f"Nan0's commentary strategy: {session.primary_strategy.value}."
        )
