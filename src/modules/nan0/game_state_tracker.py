"""
Phase 5: GAME STATE TRACKING
"What is Kyo playing? What is happening?"

Not "Kyo is playing TF2." 
"Kyo is Soldier, 23 health, on payload, enemy Spy behind her, about to die."

Game-specific analyzers:
- TF2: class, health, killfeed, objective
- Minecraft: biome, inventory, mobs, time

Tracks: player status, environment danger, recent events, match status.

Data: In-memory during gameplay. Archived to session log.
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import json


class GameType(Enum):
    UNKNOWN = "unknown"
    TF2 = "tf2"
    MINECRAFT = "minecraft"
    OTHER = "other"


class TF2Class(Enum):
    SCOUT = "scout"
    SOLDIER = "soldier"
    PYRO = "pyro"
    DEMOMAN = "demoman"
    HEAVY = "heavy"
    ENGINEER = "engineer"
    MEDIC = "medic"
    SNIPER = "sniper"
    SPY = "spy"
    UNKNOWN = "unknown"


class DangerLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class PlayerStatus:
    health: int = 100
    max_health: int = 100
    ammo: int = 0
    class_type: Optional[str] = None
    team: Optional[str] = None
    position: Optional[Tuple[float, float, float]] = None
    is_alive: bool = True
    is_ubered: bool = False
    killstreak: int = 0
    deaths: int = 0
    kills: int = 0

    @property
    def health_percent(self) -> float:
        return self.health / self.max_health if self.max_health > 0 else 0

    @property
    def danger_level(self) -> DangerLevel:
        if self.health_percent < 0.2:
            return DangerLevel.CRITICAL
        elif self.health_percent < 0.5:
            return DangerLevel.DANGER
        elif self.health_percent < 0.75:
            return DangerLevel.CAUTION
        return DangerLevel.SAFE


@dataclass
class GameEvent:
    timestamp: float
    event_type: str  # "kill", "death", "objective", "chat", "class_change", "item_pickup"
    description: str
    actor: Optional[str] = None
    target: Optional[str] = None
    importance: float = 0.5  # 0.0 to 1.0


@dataclass
class GameState:
    game_type: GameType = GameType.UNKNOWN
    game_detected: bool = False
    player_status: PlayerStatus = field(default_factory=PlayerStatus)

    # Environment
    map_name: Optional[str] = None
    game_mode: Optional[str] = None
    round_time: float = 0.0
    team_score: Tuple[int, int] = (0, 0)

    # TF2 specific
    objective_type: Optional[str] = None  # "payload", "cp", "koth", "ctf"
    objective_progress: float = 0.0  # 0.0 to 1.0
    is_overtime: bool = False

    # Minecraft specific
    biome: Optional[str] = None
    time_of_day: Optional[str] = None  # "day", "night", "dawn", "dusk"
    inventory_key_items: List[str] = field(default_factory=list)
    nearby_mobs: List[str] = field(default_factory=list)
    depth: Optional[int] = None
    dimension: Optional[str] = None

    # Events
    recent_events: List[GameEvent] = field(default_factory=list)
    killfeed: List[GameEvent] = field(default_factory=list)

    # Session tracking
    session_start: float = field(default_factory=time.time)
    session_duration: float = 0.0
    peak_danger: DangerLevel = DangerLevel.SAFE
    total_deaths: int = 0
    total_kills: int = 0

    @property
    def is_high_intensity(self) -> bool:
        """Check if game is in high-intensity moment."""
        if self.player_status.danger_level in (DangerLevel.DANGER, DangerLevel.CRITICAL):
            return True
        if self.is_overtime:
            return True
        if len([e for e in self.recent_events[-10:] 
                if e.event_type in ("kill", "death")]) > 3:
            return True
        return False


class GameStateTracker:
    """
    Tracks real-time game state from vision/screen data.

    Integration: Called from nan0_vision_skill.py when game is detected.
    Enriches _build_pressure_event() with game context.
    """

    MAX_EVENTS = 100
    DANGER_HISTORY_WINDOW = 30  # seconds

    def __init__(self):
        self._current_state: Optional[GameState] = None
        self._session_log: List[Dict] = []
        self._last_update: float = 0.0

    def detect_game(self, vision_data: Dict) -> GameType:
        """
        Detect which game is being played from vision data.
        Called from vision skill L2 semantic analysis.
        """
        # Heuristics from vision data
        game_ui = vision_data.get("game_ui_detected", "unknown")

        if game_ui in ("tf2", "team_fortress"):
            return GameType.TF2
        elif game_ui in ("minecraft", "mc"):
            return GameType.MINECRAFT

        # Additional heuristics from OCR
        ocr_text = " ".join(vision_data.get("ocr_keywords", []))
        if any(word in ocr_text for word in ["payload", "control point", "scout", "medic"]):
            return GameType.TF2
        elif any(word in ocr_text for word in ["minecraft", "inventory", "crafting", "biome"]):
            return GameType.MINECRAFT

        return GameType.UNKNOWN

    def update_from_vision(self, vision_data: Dict):
        """
        Update game state from vision stack data.
        Called every vision tick when game is detected.
        """
        game_type = self.detect_game(vision_data)

        if game_type == GameType.UNKNOWN:
            if self._current_state and self._current_state.game_detected:
                # Game ended
                self._archive_session()
                self._current_state = None
            return

        # Initialize or update state
        if not self._current_state or self._current_state.game_type != game_type:
            self._current_state = GameState(
                game_type=game_type,
                game_detected=True,
                session_start=time.time()
            )

        state = self._current_state
        now = time.time()
        state.session_duration = now - state.session_start

        # Update based on game type
        if game_type == GameType.TF2:
            self._update_tf2(state, vision_data)
        elif game_type == GameType.MINECRAFT:
            self._update_minecraft(state, vision_data)

        # Track peak danger
        if state.player_status.danger_level.value > state.peak_danger.value:
            state.peak_danger = state.player_status.danger_level

        self._last_update = now

    def _update_tf2(self, state: GameState, vision_data: Dict):
        """Update TF2-specific state from vision."""
        layer1 = vision_data.get("layer1_reflex", {})
        layer2 = vision_data.get("layer2_semantic", {})

        # Detect class from UI
        state.player_status.class_type = self._detect_tf2_class(vision_data)

        # Detect health from HUD
        state.player_status.health = self._detect_health(vision_data)
        state.player_status.max_health = self._detect_max_health(vision_data)

        # Detect game mode
        state.objective_type = layer2.get("game_mode", state.objective_type)

        # Detect overtime
        state.is_overtime = self._detect_overtime(vision_data)

        # Add events from killfeed detection
        new_events = self._detect_tf2_events(vision_data)
        for event in new_events:
            state.recent_events.append(event)
            if event.event_type in ("kill", "death"):
                state.killfeed.append(event)
                if event.event_type == "kill":
                    state.total_kills += 1
                    state.player_status.kills += 1
                elif event.event_type == "death":
                    state.total_deaths += 1
                    state.player_status.deaths += 1
                    state.player_status.killstreak = 0

        # Prune old events
        self._prune_events(state)

    def _update_minecraft(self, state: GameState, vision_data: Dict):
        """Update Minecraft-specific state from vision."""
        layer1 = vision_data.get("layer1_reflex", {})
        layer2 = vision_data.get("layer2_semantic", {})

        # Detect biome from screen
        state.biome = layer2.get("biome", state.biome)

        # Detect time
        brightness = layer1.get("brightness", 50)
        if brightness > 80:
            state.time_of_day = "day"
        elif brightness < 20:
            state.time_of_day = "night"
        elif brightness > 50:
            state.time_of_day = "dawn" if state.time_of_day == "night" else "dusk"

        # Detect dimension
        if layer2.get("dimension"):
            state.dimension = layer2["dimension"]

        # Detect depth
        state.depth = layer2.get("depth", state.depth)

        # Detect inventory (from UI detection)
        state.inventory_key_items = self._detect_mc_inventory(vision_data)

        # Detect nearby mobs
        yolo_objects = vision_data.get("latest_yolo", {}).get("objects", [])
        state.nearby_mobs = [obj["class"] for obj in yolo_objects 
                            if obj.get("class") in ["zombie", "skeleton", "creeper", 
                                                     "spider", "enderman", "pig", "cow"]]

        # Add events
        new_events = self._detect_mc_events(vision_data)
        state.recent_events.extend(new_events)
        self._prune_events(state)

    def _detect_tf2_class(self, vision_data: Dict) -> Optional[str]:
        """Detect TF2 class from HUD/screen."""
        ocr = vision_data.get("latest_ocr", {}).get("text", "")
        class_indicators = {
            "scout": ["scattergun", "bonk", "mad milk"],
            "soldier": ["rocket launcher", "shotgun", "banner"],
            "pyro": ["flamethrower", "degreaser", "airblast"],
            "demoman": ["grenade launcher", "stickybomb", "shield"],
            "heavy": ["minigun", "sandvich", "brass beast"],
            "engineer": ["wrench", "sentry", "dispenser"],
            "medic": ["medi gun", "syringe gun", "ubercharge"],
            "sniper": ["sniper rifle", "jarate", "bushwacka"],
            "spy": ["revolver", "sapper", "knife", "disguise"]
        }

        ocr_lower = ocr.lower()
        for class_name, indicators in class_indicators.items():
            if any(ind in ocr_lower for ind in indicators):
                return class_name

        return None

    def _detect_health(self, vision_data: Dict) -> int:
        """Detect player health from HUD."""
        # Placeholder: would use OCR on health bar region
        # For now, infer from semantic
        layer2 = vision_data.get("layer2_semantic", {})
        if "low health" in layer2.get("activity", "").lower():
            return 25
        elif "critical" in layer2.get("tone", "").lower():
            return 15
        return 100  # Default

    def _detect_max_health(self, vision_data: Dict) -> int:
        """Detect max health (varies by class)."""
        class_type = self._detect_tf2_class(vision_data)
        max_healths = {
            "scout": 125, "soldier": 200, "pyro": 175,
            "demoman": 175, "heavy": 300, "engineer": 125,
            "medic": 150, "sniper": 125, "spy": 125
        }
        return max_healths.get(class_type, 100)

    def _detect_overtime(self, vision_data: Dict) -> bool:
        """Detect overtime from screen."""
        ocr = vision_data.get("latest_ocr", {}).get("text", "")
        return "OVERTIME" in ocr.upper() or "overtime" in ocr.lower()

    def _detect_tf2_events(self, vision_data: Dict) -> List[GameEvent]:
        """Detect game events from vision changes."""
        events = []
        now = time.time()

        # Detect kills from killfeed (would need OCR on killfeed region)
        # Placeholder logic
        layer1 = vision_data.get("layer1_reflex", {})
        if layer1.get("major_change") and layer1.get("combat"):
            events.append(GameEvent(
                timestamp=now,
                event_type="combat",
                description="Combat detected",
                importance=0.6
            ))

        return events

    def _detect_mc_inventory(self, vision_data: Dict) -> List[str]:
        """Detect key inventory items from screen."""
        # Would use OCR on inventory screen
        # Placeholder
        return []

    def _detect_mc_events(self, vision_data: Dict) -> List[GameEvent]:
        """Detect Minecraft events."""
        events = []
        now = time.time()

        layer1 = vision_data.get("layer1_reflex", {})
        if layer1.get("menu_open"):
            events.append(GameEvent(
                timestamp=now,
                event_type="menu_open",
                description="Inventory/menu opened",
                importance=0.3
            ))

        return events

    def _prune_events(self, state: GameState):
        """Keep only recent events."""
        if len(state.recent_events) > self.MAX_EVENTS:
            state.recent_events = state.recent_events[-self.MAX_EVENTS:]
        if len(state.killfeed) > 20:
            state.killfeed = state.killfeed[-20:]

    def _archive_session(self):
        """Archive completed game session."""
        if not self._current_state:
            return

        session_data = {
            "game_type": self._current_state.game_type.value,
            "session_start": self._current_state.session_start,
            "session_duration": self._current_state.session_duration,
            "total_kills": self._current_state.total_kills,
            "total_deaths": self._current_state.total_deaths,
            "peak_danger": self._current_state.peak_danger.value,
            "events_count": len(self._current_state.recent_events)
        }

        self._session_log.append(session_data)

        # Save to file
        log_path = Path("data/nan0/game_sessions.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(session_data) + "\n")

    def get_context_for_thought(self) -> Dict:
        """
        Generate game context for thought engine.
        Called from _build_pressure_event() when game is active.
        """
        if not self._current_state or not self._current_state.game_detected:
            return {}

        state = self._current_state
        ps = state.player_status

        context = {
            "game_type": state.game_type.value,
            "game_detected": True,
            "session_duration_minutes": round(state.session_duration / 60, 1),
            "is_high_intensity": state.is_high_intensity,
            "player": {
                "health": ps.health,
                "health_percent": round(ps.health_percent * 100, 1),
                "danger_level": ps.danger_level.value,
                "class": ps.class_type,
                "is_alive": ps.is_alive,
                "killstreak": ps.killstreak,
                "kills": ps.kills,
                "deaths": ps.deaths
            },
            "environment": {
                "map": state.map_name,
                "mode": state.game_mode,
                "overtime": state.is_overtime
            },
            "recent_events": [
                {"type": e.event_type, "desc": e.description, "importance": e.importance}
                for e in state.recent_events[-5:]
            ]
        }

        # Minecraft-specific
        if state.game_type == GameType.MINECRAFT:
            context["minecraft"] = {
                "biome": state.biome,
                "time": state.time_of_day,
                "dimension": state.dimension,
                "depth": state.depth,
                "nearby_mobs": state.nearby_mobs[:5],
                "inventory": state.inventory_key_items[:5]
            }

        # TF2-specific
        if state.game_type == GameType.TF2:
            context["tf2"] = {
                "objective_type": state.objective_type,
                "objective_progress": state.objective_progress,
                "team_score": state.team_score,
                "overtime": state.is_overtime
            }

        return context

    def get_urgency_score(self) -> float:
        """
        Calculate urgency for Nan0 to comment.
        0.0 = no need, 1.0 = critical moment.
        """
        if not self._current_state:
            return 0.0

        state = self._current_state

        if state.player_status.danger_level == DangerLevel.CRITICAL:
            return 0.9
        elif state.player_status.danger_level == DangerLevel.DANGER:
            return 0.7
        elif state.is_overtime:
            return 0.8
        elif state.player_status.killstreak >= 5:
            return 0.6
        elif state.is_high_intensity:
            return 0.5

        return 0.0
