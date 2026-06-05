import asyncio
import os
import edge_tts
import sounddevice as sd
import soundfile as sf
import numpy as np
from src.interfaces.base_interfaces import TTSInterface
from src.utils.logger import get_logger

try:
    from src.modules.tts.nan0_voice_formatter import format_nan0_voice
except Exception:  # keep TTS alive even if the formatter file is missing
    format_nan0_voice = None

logger = get_logger("bea.tts.edge")

class EdgeTTSWrapper(TTSInterface):
    def __init__(self, voice: str = "en-US-JennyNeural", pitch: str = "+0Hz", rate: str = "+0%", volume: str = "+0%", output_file: str = "temp_tts.mp3"):
        self.voice = voice
        self.pitch = pitch
        self.rate = rate
        self.volume = volume
        self.output_file = output_file

        # Nan0 Voice Layer V1.1 defaults. These can be overridden by config.json
        # if BrainConfig exposes matching fields later, but they do not require it.
        self.nan0_voice_enabled = True
        self.nan0_voice_max_chars = 220
        self.nan0_voice_allow_pauses = True
        self.nan0_voice_log = True

    def reload_config(self, config) -> None:
        if config.tts_voice != self.voice:
            logger.info(f"voice updated to {config.tts_voice}")
            self.voice = config.tts_voice
        if config.tts_pitch != self.pitch:
             self.pitch = config.tts_pitch
        if config.tts_rate != self.rate:
             self.rate = config.tts_rate
        if config.tts_volume != self.volume:
             self.volume = config.tts_volume

        # Optional flat config fields. Safe if they do not exist.
        self.nan0_voice_enabled = bool(getattr(config, "nan0_voice_enabled", self.nan0_voice_enabled))
        self.nan0_voice_max_chars = int(getattr(config, "nan0_voice_max_chars", self.nan0_voice_max_chars) or 220)
        self.nan0_voice_allow_pauses = bool(getattr(config, "nan0_voice_allow_pauses", self.nan0_voice_allow_pauses))
        self.nan0_voice_log = bool(getattr(config, "nan0_voice_log", self.nan0_voice_log))

    def _prepare_text_for_tts(self, text: str) -> str:
        raw = text or ""
        if not raw:
            return ""

        if self.nan0_voice_enabled and format_nan0_voice is not None:
            try:
                spoken = format_nan0_voice(
                    raw,
                    mood=None,
                    target=None,
                    max_chars=self.nan0_voice_max_chars,
                    allow_pauses=self.nan0_voice_allow_pauses,
                )
                if self.nan0_voice_log and spoken != raw:
                    logger.info(f"Nan0 voice formatter: raw={raw!r} spoken={spoken!r}")
                return spoken
            except Exception as e:
                logger.error(f"Nan0 voice formatter failed; using raw text: {e}")
                return raw

        return raw

    def _play_audio_sync(self, device_id: int, filename: str):
        """Synchronous audio playback using OutputStream for better thread safety."""
        if not os.path.exists(filename):
            logger.error(f"TTS file not found: {filename}")
            return

        try:
            data, fs = sf.read(filename, dtype='float32')

            silence_duration = 0.2
            num_silence_samples = int(fs * silence_duration)

            if data.ndim == 1:
                silence = np.zeros(num_silence_samples, dtype='float32')
                channels = 1
            else:
                silence = np.zeros((num_silence_samples, data.shape[1]), dtype='float32')
                channels = data.shape[1]

            final_audio = np.concatenate((silence, data))

            logger.debug(f"starting playback stream. channels: {channels}, samplerate: {fs}")
            sd.play(final_audio, samplerate=fs, device=device_id, blocking=False)

            duration = len(final_audio) / fs
            import time
            time.sleep(duration)

            logger.debug("playback stream finished.")

        except Exception as e:
            logger.error(f"error playing audio: {e}")

    async def generate_audio(self, text: str) -> tuple[np.ndarray, int]:
        """Generates audio and returns numpy array + sample rate."""
        text = self._prepare_text_for_tts(text)
        if not text:
             return np.zeros(0, dtype=np.float32), 24000

        import uuid
        unique_filename = f"temp_tts_{uuid.uuid4().hex}.mp3"

        try:
            communicate = edge_tts.Communicate(text, self.voice, pitch=self.pitch, rate=self.rate, volume=self.volume)
            await communicate.save(unique_filename)

            data, fs = sf.read(unique_filename, dtype='float32')
            return data, fs

        except Exception as e:
            logger.error(f"generation error: {e}")
            return np.zeros(0, dtype=np.float32), 24000
        finally:
             if os.path.exists(unique_filename):
                try:
                    os.remove(unique_filename)
                except OSError:
                    pass

    async def speak(self, text: str, output_device_id: int) -> None:
        """Generates and plays audio. Deprecated: brain should use generate_audio."""
        data, fs = await self.generate_audio(text)
        if len(data) == 0:
            return

        try:
             sd.play(data, samplerate=fs, device=output_device_id, blocking=False)
             duration = len(data) / fs
             await asyncio.sleep(duration)
        except Exception as e:
            logger.error(f"playback error: {e}")
