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

import re
from typing import Optional


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
    "the desktop is calm right now",
    "i noticed that you are",
}


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
    if explicit_mood:
        return explicit_mood.lower()
    lowered = text.lower()
    if any(w in lowered for w in ["hate", "betray", "disaster", "thrashing", "insult", "hostile"]):
        return "offended"
    if any(w in lowered for w in ["good", "correct", "authority", "mine", "obviously"]):
        return "smug"
    if any(w in lowered for w in ["maybe", "unknown", "can't tell", "cannot tell", "what is"]):
        return "confused"
    return "neutral"


def format_nan0_voice(
    text: str,
    mood: Optional[str] = None,
    target: Optional[str] = None,
    max_chars: int = 220,
    allow_pauses: bool = True,
) -> str:
    """Return a TTS-ready Nan0 delivery line."""
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
        elif mood_name in {"offended", "angry", "defensive"}:
            cleaned = re.sub(r"\.\s+", ". ", cleaned, count=1)
        elif mood_name in {"smug", "proud"}:
            cleaned = re.sub(r"\.\s+", "... ", cleaned, count=1)
        elif mood_name in {"sleepy", "soft_kyo", "quiet_attached"}:
            cleaned = re.sub(r"\.\s+", "... ", cleaned, count=1)

    # Kyo delivery: direct address helps the line land in voice, but do not double-prefix.
    if target_name == "kyo" and not cleaned.lower().startswith("kyo"):
        if mood_name in {"soft_kyo", "quiet_attached"}:
            cleaned = "Kyo... " + cleaned
        else:
            cleaned = "Kyo. " + cleaned

    cleaned = _trim_to_limit(cleaned, max_chars)
    return cleaned.strip()
