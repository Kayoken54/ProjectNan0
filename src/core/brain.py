import asyncio
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from src.interfaces.base_interfaces import LLMInterface, TTSInterface, OBSInterface, STTInterface
from src.core.config import BrainConfig
from src.core.resources import load_avatar_resources, resolve_mood_paths
from src.utils.history_manager import HistoryManager
from src.modules.skills.skill_manager import SkillManager
from src.core.events import EventManager, EventCategory
from src.utils.logger import get_logger
import datetime
from src.utils.nan0_output_normalizer import normalize_mood_message

logger = get_logger("bea.brain")

class AIVtuberBrain:
    def __init__(
        self, 
        config: BrainConfig, 
        llm: LLMInterface, 
        tts: TTSInterface,
        stt: STTInterface,
        obs: OBSInterface
    ):
        self.config = config
        self.llm = llm
        self.tts = tts
        self.stt = stt
        self.obs = obs
        self.png_map = {}
        self.system_prompt = ""
        self.history_manager = HistoryManager()
        self.is_speaking = False
        self.current_typing_task: Optional[asyncio.Task] = None
        self.current_speech_task: Optional[asyncio.Task] = None
        self.audio_lock = asyncio.Lock() 
        self.current_audio_buffer = None
        self.playback_start_time = 0
        self.playback_sample_rate = 24000
        self.resume_buffer = None
        
        # event manager
        self.event_manager = EventManager()
        
        # skills
        self.skill_manager = SkillManager(config, self)

        # discord voice aggregation
        self.interaction_buffer = [] 
        self.buffer_lock = asyncio.Lock()
        self.flush_task = None
        self.BUFFER_WINDOW = 0.3
        self.pending_transcripts = []
        self.transcript_buffer_lock = asyncio.Lock()


    @property
    def memory_skill(self):
        return self.skill_manager.skills.get("memory")


    def _obs_enabled(self) -> bool:
        """Return whether OBS output should be touched at runtime."""
        if hasattr(self.config, "obs_enabled"):
            return bool(getattr(self.config, "obs_enabled"))
        obs_section = getattr(self.config, "obs", None)
        if isinstance(obs_section, dict) and "enabled" in obs_section:
            return bool(obs_section.get("enabled"))
        return True

    def _get_nan0_skill(self):
        try:
            return self.skill_manager.skills.get("nan0")
        except Exception:
            return None

    def _apply_nan0_voice_effects(self, audio_data, sample_rate: int, mood: str):
        """Apply Nan0 voice effects directly to generated TTS buffers."""
        nan0_skill = self._get_nan0_skill()
        if nan0_skill and hasattr(nan0_skill, "apply_voice_effects_to_audio_buffer"):
            try:
                return nan0_skill.apply_voice_effects_to_audio_buffer(audio_data, sample_rate, mood)
            except Exception as exc:
                logger.warning(f"Nan0 voice effects failed, using raw TTS buffer: {exc}")
        return audio_data, sample_rate


    def initialize(self):
        """Loads resources and connects to services."""
        logger.info("Initializing Brain...")

        self.png_map = load_avatar_resources(self.config.avatar_map)
        if not self.png_map:
            logger.warning("No avatar resources loaded from avatar_map.")

        sys_path = Path(self.config.system_prompt_path)
        try:
            if sys_path.exists():
                self.system_prompt = sys_path.read_text(encoding="utf-8")
                logger.info(f"Loaded system prompt from {sys_path}")
            else:
                logger.warning(f"System prompt file not found: {sys_path}")
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")


        self._obs_connect()

        self.history_manager.create_session()
        logger.info(f"Brain Initialized. Session ID: {self.history_manager.session_id}")

        self.skill_manager.initialize()

    def reload_configuration(self):
        """
        Hot-reloads configuration for all components.
        Called after config.json is updated via API.
        """
        logger.info("Hot Reloading Configuration")
        
        sys_path = Path(self.config.system_prompt_path)
        try:
            if sys_path.exists():
                new_prompt = sys_path.read_text(encoding="utf-8")
                if new_prompt != self.system_prompt:
                    self.system_prompt = new_prompt
                    logger.info("Updated System Prompt.")
        except Exception as e:
            logger.error(f"Error reloading system prompt: {e}")

        self.llm.reload_config(self.config)
        self.tts.reload_config(self.config)
        if self._obs_enabled():
            self.obs.reload_config(self.config)
        if self.stt:
            self.stt.reload_config(self.config)
        
        self.skill_manager.reload_config()
        
        logger.info("Hot Reload Complete")

    def _obs_connect(self):
        if not self._obs_enabled():
            logger.info("OBS disabled by config. Skipping OBS WebSocket connection.")
            return
        if hasattr(self.obs, 'source_name'):
             self.obs.source_name = self.config.obs_avatar_source
        self.obs.connect()

    def list_sessions(self):
        return self.history_manager.list_sessions()

    def load_session(self, session_id):
        if self.history_manager.load_session(session_id):
            logger.info(f"Loaded session: {session_id}")
            return True
        return False

    def create_new_session(self):
        prev_session_id = self.history_manager.session_id
        prev_history = self.history_manager.history
        
        self.history_manager.create_session()
        logger.info(f"Created new session: {self.history_manager.session_id}")
        
        memory_skill = self.memory_skill
        if prev_session_id and prev_history and memory_skill:
             memory_skill.process_previous_session(prev_session_id, prev_history)
             
        return self.history_manager.session_id


    def _normalize_mood(self, mood: str) -> str:
        allowed = {
            "normal",
            "suspicion",
            "boredom",
            "gremlin_rage",
            "smug",
            "possessive",
            "offended",
            "muttering",
            "angry",
            "bored",
            "shock",
            "love",
            "cry",
            "ew",
            "neutral",
        }
        if not mood:
            return "normal"
        mood = str(mood).strip().lower()
        if mood in allowed:
            if mood == "angry":
                return "gremlin_rage"
            if mood == "bored":
                return "boredom"
            if mood == "neutral":
                return "normal"
            return mood
        mapping = {
            "curiosity": "suspicion",
            "curious": "suspicion",
            "friendliness": "normal",
            "friendly": "normal",
            "annoyed": "offended",
            "rage": "gremlin_rage",
            "anger": "gremlin_rage",
            "quiet": "muttering",
            "fixation": "suspicion",
            "existential": "muttering",
        }
        return mapping.get(mood, "normal")

    def _try_parse_json_obj(self, value):
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _compress_internal_thought_to_line(self, thought: dict, user_text: str = "") -> str:
        """
        V7 speech boundary:
        Thought Generation -> Internal Thought Object -> Speech Compression Layer -> Output Line.

        The LLM may produce Nan0's private thought packet. That object must never leak
        directly into chat/voice. This method compresses the internal packet into one
        speakable line while preserving the feeling that speech came from a thought.
        """
        if not thought:
            return "...the thought got lost in the cable somewhere. Annoying."

        if isinstance(thought.get("message"), str) and thought.get("message", "").strip():
            candidate = thought["message"].strip()
        else:
            candidate = str(thought.get("thought_text", "")).strip()

        if not candidate:
            return "...something moved in my head and then refused to identify itself."

        # Prevent direct echo of Kyo's input from becoming speech.
        if user_text and candidate.strip().lower() == user_text.strip().lower():
            target = str(thought.get("target_actor", "kyo")).lower()
            if target == "kyo":
                return "Kyo said hello. ...I noticed. Obviously."
            return "I heard that. The room has been informed."

        # Remove accidental JSON-ish wrapping fragments from malformed responses.
        candidate = candidate.replace("\\n", " ").replace("\n", " ").strip()
        if candidate.startswith("{") or candidate.endswith("}"):
            candidate = candidate.strip("{} ")

        # Internal thoughts can be a little longer, spoken lines should stay tight.
        max_len = 180
        if len(candidate) > max_len:
            candidate = candidate[:max_len].rsplit(" ", 1)[0] + "..."

        return candidate

    def _normalize_llm_output(self, mood: str, message: str, metadata: dict = None, user_text: str = ""):
        """Convert any raw Nan0 thought packet into safe speakable output."""
        metadata = metadata or {}
        thought_obj = None

        # Prefer explicit thought packet in metadata, then parse message if needed.
        for key in ("thought", "thought_packet", "internal_thought"):
            parsed = self._try_parse_json_obj(metadata.get(key))
            if parsed:
                thought_obj = parsed
                break

        if thought_obj is None:
            parsed_message = self._try_parse_json_obj(message)
            if parsed_message and (
                "thought_text" in parsed_message
                or "speech_pressure" in parsed_message
                or "primary_emotion" in parsed_message
            ):
                thought_obj = parsed_message

        if thought_obj:
            logger.info("Speech pipeline: Thought Generation -> Internal Thought Object -> Speech Compression Layer -> Output Line")
            compressed = self._compress_internal_thought_to_line(thought_obj, user_text=user_text)
            mood = self._normalize_mood(thought_obj.get("primary_emotion") or thought_obj.get("mood") or mood)
            metadata = dict(metadata)
            metadata["internal_thought"] = thought_obj
            metadata["speech_pipeline"] = "thought->internal_thought_object->speech_compression->output_line"
            return mood, compressed, metadata

        return self._normalize_mood(mood), str(message or "...").strip(), metadata

    async def _play_audio(self, audio_data, sample_rate, device_id):
        """
        Internal helper to play audio via sounddevice with tracking.
        """
        import sounddevice as sd
        import time
        import numpy as np

        if len(audio_data) == 0:
            return

        # Windows audio hardening for EdgeTTS + sounddevice.
        # Some devices reject EdgeTTS native sample rates. Nan0 normalizes to
        # 48 kHz stereo before playback, then falls back to the default device
        # instead of crashing the whole runtime.
        target_sample_rate = int(getattr(self.config, "audio_output_sample_rate", 48000) or 48000)

        audio_data = np.asarray(audio_data)
        if audio_data.size == 0:
            return

        if audio_data.dtype.kind in {"i", "u"}:
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / max_val
        else:
            audio_data = audio_data.astype(np.float32)

        if audio_data.ndim == 1:
            audio_data = np.column_stack([audio_data, audio_data])
        elif audio_data.ndim == 2 and audio_data.shape[1] == 1:
            audio_data = np.repeat(audio_data, 2, axis=1)
        elif audio_data.ndim == 2 and audio_data.shape[1] > 2:
            audio_data = audio_data[:, :2]

        if sample_rate != target_sample_rate:
            duration_seconds = len(audio_data) / float(sample_rate)
            old_x = np.linspace(0.0, duration_seconds, num=len(audio_data), endpoint=False)
            new_len = max(1, int(duration_seconds * target_sample_rate))
            new_x = np.linspace(0.0, duration_seconds, num=new_len, endpoint=False)
            channels = []
            for ch in range(audio_data.shape[1]):
                channels.append(np.interp(new_x, old_x, audio_data[:, ch]))
            audio_data = np.stack(channels, axis=1).astype(np.float32)
            sample_rate = target_sample_rate

        self.current_audio_buffer = audio_data
        self.playback_sample_rate = sample_rate
        self.playback_start_time = time.time()

        if device_id == "" or str(device_id).lower() == "none":
            device_id = None

        try:
            sd.play(audio_data, samplerate=sample_rate, device=device_id, blocking=False)
        except Exception as e:
            logger.error(f"Audio playback failed on device {device_id}: {e}. Trying default output device.")
            try:
                sd.play(audio_data, samplerate=sample_rate, device=None, blocking=False)
            except Exception as e2:
                logger.error(f"Audio playback failed on default output too: {e2}. Skipping audio instead of crashing.")
                return

        duration = len(audio_data) / float(sample_rate)
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            sd.stop()
            raise

    async def perform_output_task(self, mood: str, message: str):
        """Background task to handle audio/visual output."""
        import sounddevice as sd
        self.is_speaking = True
        try:
            logger.info(f"Mood: {mood}")
            logger.info(f"Message: {message}")

            if self.current_typing_task and not self.current_typing_task.done():
                logger.info("Interrupting previous typing task...")
                self.current_typing_task.cancel()
            
            if self.current_speech_task and not self.current_speech_task.done():
                logger.info("Interrupting previous speech task...")
                self.current_speech_task.cancel()

            if self._obs_enabled() and self.config.obs_text_source:
                self.obs.set_text("", self.config.obs_text_source)

            try:
                idle_path, talking_path = resolve_mood_paths(self.png_map, mood)
            except KeyError:
                 logger.warning(f"Could not resolve mood {mood}, checking 'normal'...")
                 if "normal" in self.png_map:
                     idle_path, talking_path = self.png_map["normal"]
                 else:
                     idle_path = talking_path = Path("placeholder.png") 

            if self._obs_enabled():
                if self.config.obs_source_type == "media":
                    self.obs.set_media(talking_path)
                else:
                    self.obs.set_image(talking_path)


            async with self.audio_lock:
                self.current_typing_task = None
                if self._obs_enabled() and self.config.obs_text_source:
                    self.current_typing_task = asyncio.create_task(
                        self.obs.type_text(
                            text=message,
                            source_name=self.config.obs_text_source,
                            line_width=self.config.text_line_width,
                            max_lines=self.config.text_lines,
                            base_font_size=self.config.text_font_size,
                            min_font_size=self.config.text_min_font_size,
                            font_step=self.config.text_font_step,
                            typing_delay=self.config.typing_delay,
                            min_page_duration=self.config.text_min_duration
                        )
                    )

                self.event_manager.publish(EventCategory.OUTPUT, "tts", f"Speaking: {message[:50]}...", metadata={"device_id": self.config.audio_device_id})

                if message:
                    audio_data, fs = await self.tts.generate_audio(message)
                    audio_data, fs = self._apply_nan0_voice_effects(audio_data, fs, mood)
                    self.current_speech_task = asyncio.create_task(
                        self._play_audio(audio_data, fs, self.config.audio_device_id)
                    )

                    font_used = self.config.text_font_size
                    try:
                        if self.current_typing_task:
                            results = await asyncio.gather(self.current_typing_task, self.current_speech_task)
                            font_used = results[0]
                        else:
                            await self.current_speech_task
                    except asyncio.CancelledError:
                        logger.info("Output tasks cancelled (Interruption).")
                        pass

            if self._obs_enabled() and self.config.obs_text_source:
                 self.obs.set_text("", self.config.obs_text_source, font_size=font_used)
            
            if self._obs_enabled():
                if self.config.obs_source_type == "media":
                    self.obs.set_media(idle_path)
                else:
                    self.obs.set_image(idle_path)
        finally:
            self.is_speaking = False

    async def interrupt(self):
        """
        Stops current speech/typing immediately.
        Intended for 'Barge-in' functionality.
        """
        logger.info("Interruption Signal Received!")
        import sounddevice as sd
        import time
        
        if self.is_speaking and self.current_audio_buffer is not None:
            try:
                elapsed = time.time() - self.playback_start_time
                consumed_samples = int(elapsed * self.playback_sample_rate)
                total_samples = len(self.current_audio_buffer)
                
                if consumed_samples < total_samples:
                    remaining = self.current_audio_buffer[consumed_samples:]
                    if len(remaining) > (0.5 * self.playback_sample_rate):
                        self.resume_buffer = remaining
                        logger.info(f"Buffered {len(remaining)/self.playback_sample_rate:.2f}s for resume.")
                    else:
                        self.resume_buffer = None
                else:
                    self.resume_buffer = None
            except Exception as e:
                logger.error(f"Error calculating resume buffer: {e}")
                self.resume_buffer = None
        
        try:
            sd.stop()
        except Exception as e:
             logger.error(f"Error stopping sounddevice: {e}")
        
        if self.current_speech_task and not self.current_speech_task.done():
            self.current_speech_task.cancel()
        
        if self.current_typing_task and not self.current_typing_task.done():
            self.current_typing_task.cancel()
            
        if self._obs_enabled() and self.config.obs_text_source:
             self.obs.set_text("", self.config.obs_text_source)
        
        self.history_manager.add_message("system", "[Interrupted by User]")
        
        self.is_speaking = False
        return "Interrupted"

    async def _resume_speech(self):
        """
        Resumes speech from the resume_buffer.
        """
        if self.resume_buffer is None:
            logger.info("No resume buffer found.")
            return

        logger.info("Resuming speech...")
        self.is_speaking = True
        try:
             async with self.audio_lock:

                 if self._obs_enabled() and "normal" in self.png_map:
                     _, talking_path = self.png_map["normal"]
                     if self.config.obs_source_type == "media":
                        self.obs.set_media(talking_path)
                     else:
                        self.obs.set_image(talking_path)

                 self.current_speech_task = asyncio.create_task(
                     self._play_audio(self.resume_buffer, self.playback_sample_rate, self.config.audio_device_id)
                 )
                 try:
                    await self.current_speech_task
                 except asyncio.CancelledError:
                    pass
                 
                 self.resume_buffer = None
        finally:
            self.is_speaking = False
            if self._obs_enabled() and "normal" in self.png_map:
                 idle_path, _ = self.png_map["normal"]
                 if self.config.obs_source_type == "media":
                    self.obs.set_media(idle_path)
                 else:
                    self.obs.set_image(idle_path)

    def _is_backchannel(self, text: str) -> bool:
        """
        Heuristic to decide if input is just a backchannel (resume signal).
        """
        text = text.strip().lower()
        if len(text) > 30: 
            return False
            
        backchannels = {
            "ok", "okay", "k", "kk", 
            "yes", "yeah", "yep", "yup", "sì", "si", "certo",
            "mh", "mm", "mmm", "mhm", "uh-huh",
            "go on", "continue", "procedi", "continua", "vai avanti"
        }
        
        if text in backchannels:
            return True
            
        return False

    async def generate_response(self, user_text: str, system_prompt: str = None) -> Tuple[str, str]:
        """Generates the response but does NOT play it."""
        
        if self.resume_buffer is not None and self._is_backchannel(user_text):
            logger.info(f"Backchannel detected ('{user_text}'). Resuming...")
            self.history_manager.add_message("user", user_text)
            await self._resume_speech()
            return "neutral", "[RESUMED]"


        self.event_manager.publish(EventCategory.INPUT, "user", user_text)
        history = self.history_manager.get_recent_history()
        self.history_manager.add_message("user", user_text)
        
        final_prompt = system_prompt if system_prompt else self.system_prompt
        
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        final_prompt = f"CURRENT DATE: {today_str}\n\n{final_prompt}"

        # --- MEMORY INJECTION (RAG) ---
        if self.memory_skill and self.memory_skill.enabled:
             # 2. retrieve context
             context = self.memory_skill.retrieve_context(user_text)
             
             # 3. inject context at the end
             memory_section = f"\n\n[LONG TERM MEMORY]\n{context}\n"
             final_prompt += memory_section

        
        mood, message, metadata = self.llm.chat(user_text, system_prompt=final_prompt, history=history)
        mood, message, metadata = self._normalize_llm_output(mood, message, metadata, user_text=user_text)
        
        # save with metadata
        if "mood" in metadata:
            del metadata["mood"]
        self.history_manager.add_message("assistant", message, mood=mood, **metadata)
        
        self.event_manager.publish(EventCategory.OUTPUT, "llm", message, metadata={"mood": mood})
        return mood, message

    async def generate_audio_response(self, audio_path: str) -> Tuple[str, str, str]:
        import os
        filename = os.path.basename(audio_path)       
        history = self.history_manager.get_recent_history()
        transcript = ""
        
        if self.stt:
             transcript = self.stt.transcribe(audio_path)
             if transcript:
                  logger.info(f"Audio Transcript: '{transcript}'")
                  if self.resume_buffer is not None and self._is_backchannel(transcript):
                       logger.info("Audio Backchannel detected. Resuming...")
                       self.history_manager.add_message("user", transcript)
                       await self._resume_speech()
                       return "neutral", "[RESUMED]", transcript
        
        # prepare context (same as generate_response)
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        final_prompt = f"CURRENT DATE: {today_str}\n\n{self.system_prompt}"
        
        if self.memory_skill and self.memory_skill.enabled:
             # use transcript for retrieval if available, else generic
             query = transcript if transcript else "Audio Message"
             context = self.memory_skill.retrieve_context(query) 
             memory_section = f"\n\n[LONG TERM MEMORY]\n{context}\n"
             final_prompt += memory_section

        # save user turn to history (mirrors generate_response behaviour)
        user_content = transcript if transcript else "[Audio Message]"
        self.event_manager.publish(EventCategory.INPUT, "user", user_content)
        self.history_manager.add_message("user", user_content)

        # call LLM
        
        if transcript:
             mood, message, metadata = self.llm.chat(transcript, system_prompt=final_prompt, history=history)
             mood, message, metadata = self._normalize_llm_output(mood, message, metadata, user_text=transcript)
        else:
             mood, message, metadata = self.llm.chat_audio(audio_path, system_prompt=final_prompt, history=history)
             transcript = "[Audio Message]"
             mood, message, metadata = self._normalize_llm_output(mood, message, metadata, user_text=transcript)
        
        if "mood" in metadata:
            del metadata["mood"]
        self.history_manager.add_message("assistant", message, mood=mood, **metadata)
        
        return mood, message, transcript

    # deprecated single-call methods kept for compatibility if needed, but updated to use new flow
    async def process_text_input(self, user_text: str):
        nan0 = None
        try:
            nan0 = self.skill_manager.skills.get("nan0")
        except Exception:
            nan0 = None
        if nan0 and getattr(nan0, "is_active", False) and hasattr(nan0, "handle_external_message"):
            return await nan0.handle_external_message(user_text, actor="kyo", source="kyo")

        mood, message = await self.generate_response(user_text)
        mood, message = normalize_mood_message(mood, message, target_actor="kyo")
        await self.perform_output_task(mood, message)
        return mood, message

    async def process_audio_input(self, audio_path: str):
        transcript = ""
        if self.stt:
            try:
                transcript = self.stt.transcribe(audio_path) or ""
            except Exception as e:
                logger.error(f"STT failed: {e}")
        if transcript:
            nan0 = None
            try:
                nan0 = self.skill_manager.skills.get("nan0")
            except Exception:
                nan0 = None
            if nan0 and getattr(nan0, "is_active", False) and hasattr(nan0, "handle_external_message"):
                mood, message = await nan0.handle_external_message(transcript, actor="kyo", source="kyo_mic")
                return mood, message

        mood, message, _ = await self.generate_audio_response(audio_path)
        mood, message = normalize_mood_message(mood, message, target_actor="kyo")
        await self.perform_output_task(mood, message)
        return mood, message

    async def process_discord_interaction(self, audio_path: str, username: str) -> Tuple[str, str, str, bytes]:
        """
        BUFFERED pipeline for Discord Voice.
        Aggregates simultaneous speakers into one LLM context.
        """
        # 1. transcribe immediately
        transcript = ""
        if self.stt:
            transcript = self.stt.transcribe(audio_path)
            logger.info(f"Transcript from {username}: '{transcript}'")
        
        if not transcript:
             transcript = "[Unintelligible]"

        is_backchannel = self._is_backchannel(transcript)

        # 2. add to buffer
        future = asyncio.Future()
        
        async with self.buffer_lock:
            self.interaction_buffer.append({
                "future": future,
                "username": username,
                "transcript": transcript,
                "is_backchannel": is_backchannel
            })
            
            # start flush timer if not running
            if not self.flush_task:
                 self.flush_task = asyncio.create_task(self._schedule_flush())

        # 3. wait for flush result
        try:
            return await future
        except Exception as e:
            logger.error(f"Error waiting for flush: {e}")
            return "error", "", "", b""

    async def _schedule_flush(self):
        """Waits for window then flushes."""
        await asyncio.sleep(self.BUFFER_WINDOW)
        async with self.buffer_lock:
            await self._flush_buffer()
            self.flush_task = None

    async def _flush_buffer(self):
        """
        Combines all buffered inputs and calls LLM once.
        Includes any pending_transcripts accumulated while Bea was speaking.
        """
        if not self.interaction_buffer:
            return

        items = self.interaction_buffer[:]
        self.interaction_buffer.clear()
        
        logger.info(f"Flushing {len(items)} items...")

        all_backchannel = all(item['is_backchannel'] for item in items)
        
        if all_backchannel:
             logger.info("All inputs are backchannels. Resuming.")
             for item in items:
                 if not item['future'].done():
                     item['future'].set_result(("resume", "[RESUMED]", item['transcript'], b""))
             return

        buffered_context = ""
        async with self.transcript_buffer_lock:
            if self.pending_transcripts:
                buffered_context = "\n".join(self.pending_transcripts)
                logger.info(f"Draining {len(self.pending_transcripts)} buffered transcript(s)")
                self.pending_transcripts.clear()

        combined_text = ""
        full_transcript_log = ""
        
        if buffered_context:
            combined_text += f"[While you were talking, you overheard:]\n{buffered_context}\n\n[Then they said:]\n"
        
        for item in items:
            combined_text += f"[{item['username']}]: {item['transcript']}\n"
            full_transcript_log += f"{item['username']}: {item['transcript']} | "

        logger.info(f"Combined Context:\n{combined_text.strip()}")

        mood, message = await self.generate_response(combined_text.strip())
        
        audio_data, sample_rate = await self.tts.generate_audio(message)
        audio_data, sample_rate = self._apply_nan0_voice_effects(audio_data, sample_rate, mood)
        
        import io
        import soundfile as sf
        byte_io = io.BytesIO()
        sf.write(byte_io, audio_data, sample_rate, format='WAV')
        audio_bytes = byte_io.getvalue()
        
        asyncio.create_task(self._perform_visual_only_task(mood, message, len(audio_data)/sample_rate))

        leader = items[0]
        
        if not leader['future'].done():
             leader['future'].set_result(("success", message, full_transcript_log, audio_bytes))
             
        # resolve followers (empty audio)
        for item in items[1:]:
             if not item['future'].done():
                 item['future'].set_result(("success", "(Merged)", item['transcript'], b""))

    async def _perform_visual_only_task(self, mood: str, message: str, duration: float):
        """
        Updates OBS visuals/text without playing local audio.
        """
        self.is_speaking = True
        try:
             # resolve imagse
            try:
                _, talking_path = resolve_mood_paths(self.png_map, mood)
            except KeyError:
                 _, talking_path = self.png_map.get("normal", (Path("placeholder.png"), Path("placeholder.png")))

            # START ANIMATOIN
            if self._obs_enabled():
                if self.config.obs_source_type == "media":
                    self.obs.set_media(talking_path)
                else:
                    self.obs.set_image(talking_path)

            if self._obs_enabled() and self.config.obs_text_source:
                 await self.obs.type_text(
                    text=message,
                    source_name=self.config.obs_text_source,
                    line_width=self.config.text_line_width,
                    max_lines=self.config.text_lines,
                    base_font_size=self.config.text_font_size,
                    min_font_size=self.config.text_min_font_size,
                    font_step=self.config.text_font_step,
                    typing_delay=self.config.typing_delay,
                    min_page_duration=self.config.text_min_duration
                )
            else:
                await asyncio.sleep(duration)
                
            # END ANIMATION
            if self._obs_enabled() and "normal" in self.png_map:
                 idle_path, _ = self.png_map["normal"]
                 if self.config.obs_source_type == "media":
                    self.obs.set_media(idle_path)
                 else:
                    self.obs.set_image(idle_path)

            if self.config.obs_text_source:
                 self.obs.set_text("", self.config.obs_text_source)
                 
        finally:
            self.is_speaking = False

    async def run_loop(self):
        logger.info("Starting interactive loop. Type 'exit' to quit.")
        logger.info("To send audio, type 'audio:path/to/file.wav'")
        
        while True:
            user_text = await asyncio.to_thread(input, "You > ")
            user_text = user_text.strip()
            
            if user_text.lower() in ("exit", "quit"):
                break
            
            if not user_text:
                continue
            
            if user_text.lower().startswith("audio:"):
                audio_path = user_text[6:].strip()
                logger.info(f"I will process audio from: {audio_path}")
                await self.process_audio_input(audio_path)
            else:
                await self.process_text_input(user_text)

    async def start_skills(self):
        """Starts the background skill manager loop."""
        await self.skill_manager.start()

    def shutdown(self):
        if self._obs_enabled():
            self.obs.disconnect()
        try:
             loop = asyncio.get_event_loop()
             if loop.is_running():
                 pass
        except:
            pass
