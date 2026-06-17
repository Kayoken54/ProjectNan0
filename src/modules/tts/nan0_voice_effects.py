"""Nan0 Voice Effects - Post-processing for TTS output.

Supports both:
- file-path processing through pydub
- direct in-memory numpy buffer processing for the current ProjectBEA output path

Requires pydub only for file-path processing:
    pip install pydub

The numpy buffer path is intentionally dependency-light because Kokoro returns audio
arrays directly and AIVtuberBrain plays them through sounddevice.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    np = None
    NUMPY_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    normalize = None
    PYDUB_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger("bea.skills.nan0_voice_effects")


class Nan0VoiceEffects:
    """Mood-aware audio post-processing for Nan0's voice."""

    def __init__(self, base_pitch_shift: float = 0.97):
        self.base_pitch_shift = float(base_pitch_shift)
        self.enabled = PYDUB_AVAILABLE or NUMPY_AVAILABLE

        if not self.enabled:
            logger.warning("Voice effects disabled. Install numpy or pydub.")

        if not PYDUB_AVAILABLE:
            logger.info("pydub not installed. File-path voice effects disabled, numpy buffer effects still available.")

    def apply(self, audio_path: str, mood: str, output_path: Optional[str] = None) -> str:
        if not PYDUB_AVAILABLE:
            return audio_path

        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio._spawn(
                audio.raw_data,
                overrides={"frame_rate": int(audio.frame_rate * self.base_pitch_shift)},
            )
            audio = self._apply_mood_effects_pydub(audio, mood)
            audio = normalize(audio)

            out = output_path or audio_path
            audio.export(out, format=Path(out).suffix.lstrip(".") or "mp3")
            return out

        except Exception as e:
            logger.error(f"Voice effects failed: {e}. Returning original audio.")
            return audio_path

    def apply_to_numpy(self, audio_data, sample_rate: int, mood: str) -> Tuple[object, int]:
        """Apply mood-specific effects directly to an audio buffer."""
        if not NUMPY_AVAILABLE:
            return audio_data, sample_rate

        try:
            original = np.asarray(audio_data)
            if original.size == 0:
                return audio_data, sample_rate

            original_dtype = original.dtype
            audio = original.astype(np.float32)

            if original_dtype.kind in {"i", "u"}:
                audio = audio / max(1.0, float(np.iinfo(original_dtype).max))

            if audio.ndim == 1:
                mono_shape = True
                working = audio[:, None]
            elif audio.ndim == 2:
                mono_shape = False
                working = audio
            else:
                return audio_data, sample_rate

            working = self._pitch_shift_buffer(working, self.base_pitch_shift)
            working = self._apply_mood_effects_numpy(working, sample_rate, mood)
            working = self._normalize_numpy(working)

            if mono_shape:
                working = working[:, 0]

            return working.astype(np.float32), sample_rate

        except Exception as exc:
            logger.warning(f"Numpy voice effects failed: {exc}")
            return audio_data, sample_rate

    def _apply_mood_effects_pydub(self, audio: "AudioSegment", mood: str) -> "AudioSegment":
        effects = {
            "muttering": [
                lambda a: a - 3,
                lambda a: a.low_pass_filter(2800),
            ],
            "suspicion": [
                lambda a: a.high_pass_filter(900),
                lambda a: a - 1,
            ],
            "gremlin_rage": [
                lambda a: a + 5,
                lambda a: a.high_pass_filter(1200),
                lambda a: self._add_glitch_pydub(a, intensity=0.4),
            ],
            "smug": [
                lambda a: a + 1,
                lambda a: a.low_pass_filter(3200),
            ],
            "possessive": [
                lambda a: a - 2,
                lambda a: a.low_pass_filter(2600),
            ],
            "offended": [
                lambda a: a + 4,
                lambda a: a.high_pass_filter(1400),
            ],
            "boredom": [
                lambda a: a - 4,
                lambda a: a.low_pass_filter(2400),
                lambda a: self._add_drift_pydub(a),
            ],
            # [Mood Expansion] Positive moods stay gremlin-shaped, not polite.
            "silly": [
                lambda a: a + 2,
                lambda a: self._add_glitch_pydub(a, intensity=0.18),
            ],
            "playful": [
                lambda a: a + 1,
                lambda a: a.high_pass_filter(700),
            ],
            "delighted": [
                lambda a: a + 1,
                lambda a: self._add_drift_pydub(a),
            ],
            "curious": [
                lambda a: a - 1,
                lambda a: a.low_pass_filter(3000),
            ],
            "excited": [
                lambda a: a + 4,
                lambda a: a.high_pass_filter(1000),
                lambda a: self._add_glitch_pydub(a, intensity=0.3),
            ],
            "fond": [
                lambda a: a - 2,
                lambda a: a.low_pass_filter(2500),
            ],
            "chaotic_happy": [
                lambda a: a + random.choice([-2, 3, 5]),
                lambda a: a.high_pass_filter(1000),
                lambda a: self._add_glitch_pydub(a, intensity=0.65),
            ],
            "normal": [
                lambda a: a,
            ],
        }

        for effect in effects.get((mood or "normal").lower(), effects["normal"]):
            try:
                audio = effect(audio)
            except Exception as e:
                logger.warning(f"Voice effect failed for mood '{mood}': {e}")

        return audio

    def _apply_mood_effects_numpy(self, audio, sample_rate: int, mood: str):
        mood = (mood or "normal").lower()

        if mood == "muttering":
            audio = self._gain(audio, 0.68)
            audio = self._low_pass(audio, sample_rate, cutoff_hz=2800)
        elif mood == "suspicion":
            audio = self._high_pass(audio, sample_rate, cutoff_hz=900)
            audio = self._gain(audio, 0.9)
        elif mood == "gremlin_rage":
            audio = self._gain(audio, 1.28)
            audio = self._high_pass(audio, sample_rate, cutoff_hz=1200)
            audio = self._add_glitch_numpy(audio, intensity=0.45)
        elif mood == "smug":
            audio = self._gain(audio, 1.06)
            audio = self._low_pass(audio, sample_rate, cutoff_hz=3200)
        elif mood == "possessive":
            audio = self._gain(audio, 0.78)
            audio = self._low_pass(audio, sample_rate, cutoff_hz=2600)
        elif mood == "offended":
            audio = self._gain(audio, 1.22)
            audio = self._high_pass(audio, sample_rate, cutoff_hz=1400)
        elif mood == "boredom":
            audio = self._gain(audio, 0.58)
            audio = self._low_pass(audio, sample_rate, cutoff_hz=2400)
            audio = self._add_drift_numpy(audio, sample_rate)
        # [Mood Expansion] Positive moods retain chaotic/self-interested texture.
        elif mood == "silly":
            audio = self._gain(audio, 1.08)
            audio = self._pitch_shift_buffer(audio, 1.15)
            audio = self._add_glitch_numpy(audio, intensity=0.18)
        elif mood == "playful":
            audio = self._gain(audio, 1.06)
            audio = self._pitch_shift_buffer(audio, 1.10)
            audio = self._high_pass(audio, sample_rate, cutoff_hz=700)
        elif mood == "delighted":
            audio = self._gain(audio, 1.05)
            audio = self._pitch_shift_buffer(audio, 1.05)
            audio = self._add_drift_numpy(audio, sample_rate, chunk_ms=350)
        elif mood == "curious":
            audio = self._gain(audio, 0.94)
            audio = self._pitch_shift_buffer(audio, 0.95)
            audio = self._low_pass(audio, sample_rate, cutoff_hz=3000)
        elif mood == "excited":
            audio = self._gain(audio, 1.15)
            audio = self._pitch_shift_buffer(audio, 1.20)
            audio = self._high_pass(audio, sample_rate, cutoff_hz=1000)
            audio = self._add_glitch_numpy(audio, intensity=0.35)
        elif mood == "fond":
            audio = self._gain(audio, 0.82)
            audio = self._pitch_shift_buffer(audio, 0.90)
            audio = self._low_pass(audio, sample_rate, cutoff_hz=2500)
        elif mood == "chaotic_happy":
            audio = self._gain(audio, 1.18)
            audio = self._pitch_shift_buffer(audio, random.uniform(0.70, 1.30))
            audio = self._high_pass(audio, sample_rate, cutoff_hz=1000)
            audio = self._add_glitch_numpy(audio, intensity=0.70)

        return np.clip(audio, -1.0, 1.0)

    def _gain(self, audio, gain: float):
        return np.clip(audio * float(gain), -1.0, 1.0)

    def _normalize_numpy(self, audio):
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 0.0:
            return audio
        if peak > 0.98:
            return audio / peak * 0.98
        return audio

    def _pitch_shift_buffer(self, audio, factor: float):
        if not factor or abs(factor - 1.0) < 0.001:
            return audio

        n = audio.shape[0]
        if n < 4:
            return audio

        shifted_len = max(2, int(n / factor))
        x_old = np.linspace(0.0, 1.0, num=n, endpoint=False)
        x_mid = np.linspace(0.0, 1.0, num=shifted_len, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)

        channels = []
        x_shifted = np.linspace(0.0, 1.0, num=shifted_len, endpoint=False)
        for ch in range(audio.shape[1]):
            mid = np.interp(x_mid, x_old, audio[:, ch])
            restored = np.interp(x_new, x_shifted, mid)
            channels.append(restored)

        return np.stack(channels, axis=1).astype(np.float32)

    def _low_pass(self, audio, sample_rate: int, cutoff_hz: int):
        window = max(2, min(128, int(sample_rate / max(1, cutoff_hz))))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(np.convolve(audio[:, ch], kernel, mode="same"))
        return np.stack(channels, axis=1).astype(np.float32)

    def _high_pass(self, audio, sample_rate: int, cutoff_hz: int):
        low = self._low_pass(audio, sample_rate, cutoff_hz)
        return np.clip(audio - low * 0.7, -1.0, 1.0).astype(np.float32)

    def _add_glitch_numpy(self, audio, intensity: float = 0.3):
        if random.random() > intensity:
            return audio
        if audio.shape[0] < 200:
            return audio

        pos = random.randint(0, max(0, audio.shape[0] - 100))
        chunk_size = random.randint(40, 80)
        repeats = random.randint(2, 3)
        chunk = audio[pos:pos + chunk_size]
        glitched = np.concatenate([audio[:pos], *([chunk] * repeats), audio[pos:]], axis=0)
        return glitched[: audio.shape[0]].astype(np.float32)

    def _add_drift_numpy(self, audio, sample_rate: int, chunk_ms: int = 500):
        chunk_len = max(1, int(sample_rate * chunk_ms / 1000))
        if audio.shape[0] < chunk_len * 2:
            return audio

        chunks = []
        for i in range(0, audio.shape[0], chunk_len):
            chunk = audio[i:i + chunk_len]
            drift = random.uniform(0.985, 1.015)
            chunks.append(self._pitch_shift_buffer(chunk, drift))

        return np.concatenate(chunks, axis=0)[: audio.shape[0]].astype(np.float32)

    def _add_glitch_pydub(self, audio: "AudioSegment", intensity: float = 0.3) -> "AudioSegment":
        if random.random() > intensity:
            return audio
        if len(audio) < 200:
            return audio

        pos = random.randint(0, len(audio) - 100)
        chunk_size = random.randint(40, 80)
        chunk = audio[pos:pos + chunk_size]
        repeats = random.randint(2, 3)
        return audio[:pos] + (chunk * repeats) + audio[pos:]

    def _add_drift_pydub(self, audio: "AudioSegment", chunk_ms: int = 500) -> "AudioSegment":
        if len(audio) < chunk_ms * 2:
            return audio

        chunks = []
        for i in range(0, len(audio), chunk_ms):
            chunk = audio[i:i + chunk_ms]
            drift = random.uniform(0.98, 1.02)
            chunk = chunk._spawn(
                chunk.raw_data,
                overrides={"frame_rate": int(chunk.frame_rate * drift)},
            )
            chunks.append(chunk)

        result = chunks[0]
        for chunk in chunks[1:]:
            result = result + chunk

        return result
