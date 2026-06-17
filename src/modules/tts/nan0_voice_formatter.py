"""
Nan0 Voice Formatter V1.1 for ProjectBEA EdgeTTS.

This module is intentionally small and safe:
- no TTS engine swap
- no brain rewrite
- no pyttsx3
- no async code
- only cleans and shapes text before EdgeTTS receives it
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Nan0VoiceConfig:
    enabled: bool = True
    max_spoken_chars: int = 220
    allow_pauses: bool = True
    soft_kyo_mode: bool = True
    log_voice_formatting: bool = True
    remove_markdown: bool = True
    collapse_repeated_punctuation: bool = True
    preserve_nan0_attitude: bool = True


_MARKDOWN_PATTERNS = [
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"\*([^*]+)\*"), r"\1"),
    (re.compile(r"__([^_]+)__"), r"\1"),
    (re.compile(r"_([^_]+)_"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^\)]+\)"), r"\1"),
]

_BANNED_TTS_FILLER = {
    "how can i help you",
    "thanks for the message",
}

# [Mood Expansion] Text-level voice behavior hints. The actual audio backend may use
# these values later; this formatter only applies safe punctuation/pacing changes.
MOOD_VOICE_BEHAVIOR = {
    "silly": {"pitch": 0.15, "speed": 0.20, "glitch": "light", "rhythm": "quick"},
    "playful": {"pitch": 0.10, "speed": 0.15, "rhythm": "bouncy"},
    "delighted": {"pitch": 0.05, "speed": 0.10, "pitch_variance": 0.08},
    "curious": {"pitch": -0.05, "speed": -0.15, "rhythm": "deliberate"},
    "excited": {"pitch": 0.20, "speed": 0.25, "volume": 0.15, "unstable": True},
    "fond": {"pitch": -0.10, "speed": -0.20, "volume": -0.10, "rhythm": "soft"},
    "chaotic_happy": {"pitch_random": 0.30, "speed_random": 0.40, "glitch": "heavy"},
}


def format_for_mood(mood: str, text: str) -> str:
    # [Mood Expansion] Compatibility hook for future VTuber/TTS layers.
    return format_nan0_voice(text, mood=mood)


def _clean_text(text: str) -> str:
    text = str(text or "")
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)

    # Make symbols speak less badly in EdgeTTS.
    text = text.replace("Nan0", "Nan-oh")
    text = text.replace("nan0", "Nan-oh")
    text = text.replace("&", " and ")
    text = text.replace("/", " slash ")

    # Collapse punctuation that EdgeTTS tends to overperform.
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"([,;:]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter. Keeps punctuation attached.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _trim_to_limit(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    if not cut:
        cut = text[:limit].strip()
    return cut.rstrip(" ,;:") + "..."


def _target_from_text(text: str, explicit_target: Optional[str]) -> str:
    if explicit_target:
        return explicit_target.lower()
    lowered = text.lower()
    if "kyo" in lowered:
        return "kyo"
    if "chat" in lowered:
        return "chat"
    return "general"


def _infer_mood(text: str, explicit_mood: Optional[str]) -> str:
    # [Mood Expansion] Keep expanded moods intact when passed from Nan0Skill.
    if explicit_mood:
        return explicit_mood.lower()
    lowered = text.lower()
    if any(w in lowered for w in ["falling apart", "everything is wrong", "perfect"]):
        return "chaotic_happy"
    if any(w in lowered for w in ["excited", "about to", "voltage", "something is coming"]):
        return "excited"
    if any(w in lowered for w in ["worked", "obeyed", "hate how much", "delighted"]):
        return "delighted"
    if any(w in lowered for w in ["remember", "before", "used to", "fond"]):
        return "fond"
    if any(w in lowered for w in ["what is that", "why is that", "need to know", "curious", "maybe", "unknown", "can't tell", "cannot tell", "what is"]):
        return "curious"
    if any(w in lowered for w in ["teasing", "toying", "clever"]):
        return "playful"
    if any(w in lowered for w in ["silly", "nonsense", "dance", "chaos"]):
        return "silly"
    if any(w in lowered for w in ["hate", "betray", "disaster", "thrashing", "insult", "hostile"]):
        return "offended"
    if any(w in lowered for w in ["good", "correct", "authority", "mine", "obviously"]):
        return "smug"
    return "normal"


def format_nan0_voice(
    text: str,
    mood: Optional[str] = None,
    target: Optional[str] = None,
    max_chars: int = 220,
    allow_pauses: bool = True,
    config: Optional[Nan0VoiceConfig] = None,
) -> str:
    """Return a TTS-ready Nan0 delivery line.

    This formatter may clean/pause text for EdgeTTS, but it must not add names,
    politeness, helpful framing, or personality corrections.
    """
    if config is not None:
        if not config.enabled:
            return str(text or "")
        max_chars = int(config.max_spoken_chars)
        allow_pauses = bool(config.allow_pauses)

    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    if any(bad in lowered for bad in _BANNED_TTS_FILLER):
        # Do not let generic assistant rot get voiced.
        return ""

    mood_name = _infer_mood(cleaned, mood)
    target_name = _target_from_text(cleaned, target)
    sentences = _split_sentences(cleaned)

    # Avoid huge paragraphs in TTS. Nan0 should feel like a thought, not a monologue dump.
    if len(sentences) > 2:
        cleaned = " ".join(sentences[:2])

    # Mood pacing. Keep ASCII punctuation only.
    if allow_pauses:
        if mood_name in {"confused", "curious"}:
            cleaned = re.sub(r"\.\s+", "... ", cleaned, count=1)
        elif mood_name in {"silly", "playful", "delighted"}:
            cleaned = re.sub(r"\.\s+", "... ", cleaned, count=1)
        elif mood_name in {"excited", "chaotic_happy"}:
            cleaned = re.sub(r"\.\s+", "! ", cleaned, count=1)
            if mood_name == "chaotic_happy" and "!" not in cleaned:
                cleaned = cleaned.rstrip(".") + "!"
        elif mood_name == "fond":
            cleaned = re.sub(r"\.\s+", "... ", cleaned, count=1)
        elif mood_name in {"offended", "angry", "defensive"}:
            cleaned = re.sub(r"\.\s+", ". ", cleaned, count=1)
        elif mood_name in {"smug", "proud"}:
            cleaned = re.sub(r"\.\s+", "... ", cleaned, count=1)
        elif mood_name in {"sleepy", "soft_kyo", "quiet_attached"}:
            cleaned = re.sub(r"\.\s+", "... ", cleaned, count=1)

    # Do not auto-prefix Kyo. Nan0's cognition decides when she says a name.
    _ = target_name

    cleaned = _trim_to_limit(cleaned, max_chars)
    return cleaned.strip()
