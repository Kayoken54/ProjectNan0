import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("bea.config")
CONFIG_FILE = "config.json"


@dataclass
class BrainConfig:
    language: str = "en"
    system_prompt_path: str = "data/prompts/nan0_persona.txt"
    llm_provider: str = "ollama"

    # local ollama
    ollama_model: str = "qwen2.5:1.5b"
    ollama_timeout: float = 25.0
    ollama_host: str = "http://localhost:11434"

    # cloud providers remain optional fallbacks only
    gemini_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = "gemini-2-flash"

    glm_key: Optional[str] = field(default_factory=lambda: os.getenv("GLM_API_KEY"))
    glm_model: str = "glm-4.7"

    openai_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = "gpt-4o-mini"

    groq_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = "llama-3.1-8b-instant"

    obs_text_source: Optional[str] = "Nan0Text"
    obs_avatar_source: str = "Nan0Avatar"
    obs_source_type: str = "image"
    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""
    audio_device_id: int = 67
    audio_output_sample_rate: int = 48000
    audio_device_hint: str = "HyperX"
    audio_hostapi_hint: str = "MME"
    stt_input_device_id: int = -1
    stt_input_device_hint: str = "HyperX"
    stt_hostapi_hint: str = "MME"

    tts_provider: str = "edge"
    tts_voice: str = "en-US-AriaNeural"
    tts_pitch: str = "+0Hz"
    tts_rate: str = "+10%"
    tts_volume: str = "+0%"

    orpheus_key: Optional[str] = field(default_factory=lambda: os.getenv("ORPHEUS_API_KEY"))
    orpheus_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("ORPHEUS_ENDPOINT", ""))
    orpheus_voice: str = "zoe"

    kokoro_model: str = "kokoro-v0_19.onnx"
    kokoro_voices_file: str = "voices.bin"
    kokoro_voice: str = "af_bella"
    kokoro_speed: float = 1.1
    kokoro_lang: str = "en-us"

    avatar_map: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "normal": {"idle": "nan0_idle.png", "talking": "nan0_talking.png"},
        "suspicion": {"idle": "nan0_suspicion.png", "talking": "nan0_suspicion_talk.png"},
        "boredom": {"idle": "nan0_bored.png", "talking": "nan0_bored_talk.png"},
        "gremlin_rage": {"idle": "nan0_angry.png", "talking": "nan0_angry_talk.png"},
        "smug": {"idle": "nan0_smug.png", "talking": "nan0_smug_talk.png"},
        "possessive": {"idle": "nan0_possessive.png", "talking": "nan0_possessive_talk.png"},
        "offended": {"idle": "nan0_offended.png", "talking": "nan0_offended_talk.png"},
        "muttering": {"idle": "nan0_mutter.png", "talking": "nan0_mutter_talk.png"},
        # [Mood Expansion] Avatar aliases until dedicated PNGs exist.
        "silly": {"idle": "nan0_smug.png", "talking": "nan0_smug_talk.png"},
        "playful": {"idle": "nan0_smug.png", "talking": "nan0_smug_talk.png"},
        "delighted": {"idle": "nan0_smug.png", "talking": "nan0_smug_talk.png"},
        "curious": {"idle": "nan0_suspicion.png", "talking": "nan0_suspicion_talk.png"},
        "excited": {"idle": "nan0_angry.png", "talking": "nan0_angry_talk.png"},
        "fond": {"idle": "nan0_possessive.png", "talking": "nan0_possessive_talk.png"},
        "chaotic_happy": {"idle": "nan0_angry.png", "talking": "nan0_angry_talk.png"},
    })

    png_dir: str = "data/pngs/nan0"

    text_line_width: int = 45
    text_lines: Optional[int] = 3
    text_font_size: int = 65
    text_min_font_size: int = 45
    text_font_step: int = 2
    typing_delay: float = 0.04
    text_min_duration: float = 2.0

    skills: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "nan0": {
            "enabled": True,
            "fast_model": "qwen2.5:1.5b",
            "deep_model": "qwen2.5:1.5b",
            "latency_budget": 25.0,
            "deep_interval": 45,
            "kyo_voice_inbox": "data/input/kyo_voice_inbox.jsonl",
            "discord_inbox": "data/input/discord_voice_inbox.jsonl",
            "perception_debug_enabled": True,
            "perception_debug_path": "data/nan0/perception_debug.jsonl",
            "persona_path": "data/prompts/nan0_persona.txt",
            "enable_body": False,
            "fallback_to_template": False,
        },
        "nan0_vision": {
            "enabled": True,
            "interval": 30,
            "model": "moondream",
            "monitor": 3,
            "confidence": 0.45,
        },
        "monologue": {
            "enabled": False,
            "interval_seconds": 45,
            "chunk_pause_seconds": 3.0,
            "prompt_path": "data/prompts/nan0_monologue.txt",
        },
        "memory": {
            "enabled": True,
            "chroma_path": "data/memory_db",
            "embedding_model": "local",
            "local_embedding_model": "all-MiniLM-L6-v2",
        },
        "minecraft": {
            "enabled": False,
            "server_url": "ws://localhost:8080",
            "max_history_events": 20,
            "debug_mode": False,
            "auto_chat_thoughts": False,
            "auto_speak_thoughts": False,
            "mc_openai_model": "gpt-4o-mini",
            "mc_openai_key": "",
            "system_prompt_path": "data/prompts/minecraft.txt",
        },
        "discord": {
            "enabled": False,
            "token": "",
            "target_channel": "",
            "api_port": 3030,
            "interrupt_threshold_ms": 3000,
        },
    })

    # Local-first default: no STT provider until explicitly enabled.
    stt_provider: str = "none"
    stt_model: str = ""

    SECRET_KEYS = [
        "gemini_key",
        "glm_key",
        "openai_key",
        "groq_key",
        "orpheus_key",
        "orpheus_endpoint",
        "mc_openai_key",
    ]

    def __post_init__(self):
        self.load_from_file()

    def load_from_file(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "obs_image_source" in data and "obs_avatar_source" not in data:
                data["obs_avatar_source"] = data.pop("obs_image_source")

            for key, value in data.items():
                if not hasattr(self, key):
                    continue

                if key in self.SECRET_KEYS:
                    current_val = getattr(self, key, None)
                    if current_val:
                        continue
                    if value is None or value == "":
                        continue

                if key == "skills":
                    current_skills = self.skills
                    for skill_name, skill_val in value.items():
                        if skill_name in current_skills:
                            if skill_name == "minecraft" and "mc_openai_key" in skill_val:
                                mc_key = skill_val["mc_openai_key"]
                                if mc_key is None or mc_key == "":
                                    del skill_val["mc_openai_key"]
                            current_skills[skill_name].update(skill_val)
                        else:
                            current_skills[skill_name] = skill_val
                else:
                    setattr(self, key, value)

        except Exception as e:
            logger.error(f"Error loading config.json: {e}")

    def save_to_file(self):
        data = asdict(self)
        for secret in self.SECRET_KEYS:
            if secret in data:
                del data[secret]

        if "skills" in data and "minecraft" in data["skills"]:
            if "mc_openai_key" in data["skills"]["minecraft"]:
                del data["skills"]["minecraft"]["mc_openai_key"]

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Configuration saved to {CONFIG_FILE} (secrets excluded)")
        except Exception as e:
            logger.error(f"Error saving config.json: {e}")
