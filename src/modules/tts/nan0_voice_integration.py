"""
Optional helper for wiring Nan0 Voice Formatter into ProjectBEA.

This file is intentionally tiny. It does not know which TTS backend ProjectBEA is using.
Call prepare_nan0_tts_text(...) immediately before the existing TTS speak/synthesize call.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from src.modules.tts.nan0_voice_formatter import Nan0VoiceConfig, format_nan0_voice

logger = logging.getLogger(__name__)


def _read_voice_config(config: Optional[Mapping[str, Any]]) -> Nan0VoiceConfig:
    if not config:
        return Nan0VoiceConfig()

    section = config.get("nan0_voice", {}) if isinstance(config, Mapping) else {}
    if not isinstance(section, Mapping):
        section = {}

    return Nan0VoiceConfig(
        enabled=bool(section.get("enabled", True)),
        max_spoken_chars=int(section.get("max_spoken_chars", 220)),
        allow_pauses=bool(section.get("allow_pauses", True)),
        soft_kyo_mode=bool(section.get("soft_kyo_mode", True)),
        log_voice_formatting=bool(section.get("log_voice_formatting", True)),
        remove_markdown=bool(section.get("remove_markdown", True)),
        collapse_repeated_punctuation=bool(section.get("collapse_repeated_punctuation", True)),
        preserve_nan0_attitude=bool(section.get("preserve_nan0_attitude", True)),
    )


def prepare_nan0_tts_text(
    raw_line: str,
    mood: Optional[str] = None,
    target: Optional[str] = None,
    runtime_config: Optional[Mapping[str, Any]] = None,
) -> str:
    voice_config = _read_voice_config(runtime_config)
    spoken_line = format_nan0_voice(raw_line, mood=mood, target=target, config=voice_config)

    if voice_config.log_voice_formatting and spoken_line != raw_line:
        logger.info("Nan0 voice formatter: raw=%r spoken=%r mood=%r target=%r", raw_line, spoken_line, mood, target)

    return spoken_line


# [Mood Expansion] Expression hints for avatar/rig layers. These are string hints only;
# no polite/helpful behavior is introduced here.
MOOD_TO_EXPRESSION = {
    "silly": {"eyes": "eyes_wide", "mouth": "mouth_twitch", "pose": "head_tilt"},
    "playful": {"eyes": "eyes_narrow", "mouth": "smirk", "pose": "lean_forward"},
    "delighted": {"eyes": "eyes_soft", "mouth": "mouth_small", "pose": "slight_blush_begrudged"},
    "curious": {"eyes": "eyes_focus", "mouth": "neutral", "pose": "head_tilt_ear_perk"},
    "excited": {"eyes": "eyes_wide", "mouth": "mouth_open", "pose": "shake_slight"},
    "fond": {"eyes": "eyes_half", "mouth": "mouth_small", "pose": "slow_blink"},
    "chaotic_happy": {"eyes": "eyes_spin", "mouth": "mouth_grin_too_wide", "pose": "jitter"},
}


def expression_for_mood(mood: Optional[str]) -> Mapping[str, str]:
    # [Mood Expansion] Runtime-safe expression lookup. Unknown moods return empty hints.
    return MOOD_TO_EXPRESSION.get((mood or "").strip().lower(), {})
