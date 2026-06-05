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
