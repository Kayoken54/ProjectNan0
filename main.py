import asyncio
import argparse
import os
import sys
import io
import faulthandler
from pathlib import Path
import warnings


def _force_utf8_stdio() -> None:
    """
    Windows hardening:
    Some local TTS/model tooling may emit UTF-8 text while the Windows console
    is still using a legacy codepage. Force stdout/stderr to UTF-8 so Kokoro
    and other local components do not crash the boot path with charmap errors.
    """
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
    except Exception:
        pass


_force_utf8_stdio()

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

warnings.filterwarnings(
    "ignore",
    message='Field name ".*" shadows an attribute in parent "Operation"',
    category=UserWarning,
    module="pydantic",
)

faulthandler.enable()

from dotenv import load_dotenv

try:
    load_dotenv(encoding="utf-8")
except UnicodeDecodeError:
    print("WARNING: .env is not UTF-8. Ignoring .env for local-first boot. Rename or resave it as UTF-8.")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import get_logger
from src.core.config import BrainConfig
from src.core.brain import AIVtuberBrain
from src.modules.llm.gemini_llm import GeminiLLM
from src.modules.llm.glm_llm import GLM47LLM
from src.modules.llm.openai_llm import OpenAILLM
from src.modules.llm.groq_llm import GroqLLM
from src.modules.llm.ollama_provider import OllamaLLM
from src.modules.obs.obs_websocket import OBSController

logger = get_logger("bea")


def parse_args():
    parser = argparse.ArgumentParser(description="ProjectBEA - AI Vtuber Engine")
    parser.add_argument("--web", action="store_true", help="Start Web Interface (FastAPI + React)")
    parser.add_argument("--system-file", default=None, help="Path to system prompt file")
    parser.add_argument("--png-dir", default=None, help="Directory for avatar PNGs")

    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "glm", "openai", "groq", "ollama"],
        default=None,
        help="LLM Provider to use",
    )
    parser.add_argument("--ollama-model", default=None, help="Ollama model")
    parser.add_argument("--ollama-timeout", type=float, default=None, help="Ollama timeout seconds")
    parser.add_argument("--ollama-host", default=None, help="Ollama host URL")

    parser.add_argument("--gemini-key", default=None, help="Google GenAI API Key")
    parser.add_argument("--gemini-model", default=None, help="Gemini Model")
    parser.add_argument("--glm-key", default=None, help="GLM API Key")
    parser.add_argument("--glm-model", default=None, help="GLM Model")
    parser.add_argument("--openai-key", default=None, help="OpenAI API Key")
    parser.add_argument("--openai-model", default=None, help="OpenAI Model")
    parser.add_argument("--groq-key", default=None, help="Groq API Key")
    parser.add_argument("--groq-model", default=None, help="Groq Model")

    parser.add_argument("--stt-provider", choices=["groq", "none"], default=None, help="STT Provider")
    parser.add_argument("--stt-model", default=None, help="STT Model")

    parser.add_argument("--obs-host", default=None, help="OBS WebSocket host")
    parser.add_argument("--obs-port", type=int, default=None, help="OBS WebSocket port")
    parser.add_argument("--obs-password", default=None, help="OBS WebSocket password")
    parser.add_argument("--obs-avatar-source", default=None, required=False, help="OBS Source Name for Avatar")
    parser.add_argument("--obs-source-type", choices=["image", "media"], default=None, help="OBS Source Type")
    parser.add_argument("--obs-text-source", default=None, help="OBS Source Name for Text Bubble")

    parser.add_argument("--tts-provider", choices=["edge", "coqui", "orpheus", "kokoro"], default=None, help="TTS Provider")
    parser.add_argument("--tts-voice", default=None, help="TTS Voice")
    parser.add_argument("--orpheus-key", default=None, help="Orpheus API Key")
    parser.add_argument("--orpheus-endpoint", default=None, help="Orpheus Endpoint")
    parser.add_argument("--orpheus-voice", default=None, help="Orpheus Voice")
    parser.add_argument("--kokoro-file", default=None, help="Kokoro Model File")
    parser.add_argument("--kokoro-voices", default=None, help="Kokoro Voices File")
    parser.add_argument("--device-id", type=int, default=None, help="Audio Output Device ID")
    parser.add_argument("--typing-delay", type=float, default=None, help="Typing animation delay")

    return parser.parse_args()


async def main():
    args = parse_args()
    config = BrainConfig()

    cli_overrides = {
        "system_prompt_path": args.system_file,
        "png_dir": args.png_dir,
        "llm_provider": args.llm_provider,
        "ollama_model": args.ollama_model,
        "ollama_timeout": args.ollama_timeout,
        "ollama_host": args.ollama_host,
        "gemini_key": args.gemini_key,
        "gemini_model": args.gemini_model,
        "glm_key": args.glm_key,
        "glm_model": args.glm_model,
        "openai_key": args.openai_key,
        "openai_model": args.openai_model,
        "groq_key": args.groq_key,
        "groq_model": args.groq_model,
        "stt_provider": args.stt_provider,
        "stt_model": args.stt_model,
        "obs_host": args.obs_host,
        "obs_port": args.obs_port,
        "obs_password": args.obs_password,
        "obs_avatar_source": args.obs_avatar_source,
        "obs_source_type": args.obs_source_type,
        "obs_text_source": args.obs_text_source,
        "tts_provider": args.tts_provider,
        "tts_voice": args.tts_voice,
        "orpheus_key": args.orpheus_key,
        "orpheus_endpoint": args.orpheus_endpoint,
        "orpheus_voice": args.orpheus_voice,
        "kokoro_model": args.kokoro_file,
        "kokoro_voices_file": args.kokoro_voices,
        "audio_device_id": args.device_id,
        "typing_delay": args.typing_delay,
    }

    for field_name, value in cli_overrides.items():
        if value is not None:
            setattr(config, field_name, value)
            logger.info(f"CLI override: {field_name} = {value}")

    if config.stt_provider == "groq":
        if not config.groq_key:
            logger.warning("GROQ_API_KEY missing. STT disabled for local-first boot.")
            stt = None
        else:
            from src.modules.STT.groq_stt import GroqSTT

            stt = GroqSTT(config)
    else:
        stt = None

    if config.llm_provider == "ollama":
        llm = OllamaLLM(
            model_name=config.ollama_model,
            timeout=config.ollama_timeout,
            host=config.ollama_host,
            stt_interface=stt,
        )
    elif config.llm_provider == "gemini":
        if not config.gemini_key:
            logger.error("GEMINI_API_KEY is missing via env, config, or CLI.")
            return
        llm = GeminiLLM(api_key=config.gemini_key, model_name=config.gemini_model)
    elif config.llm_provider == "glm":
        if not config.glm_key:
            logger.error("GLM_API_KEY is missing via env, config, or CLI.")
            return
        llm = GLM47LLM(api_key=config.glm_key, model_name=config.glm_model, stt_interface=stt)
    elif config.llm_provider == "openai":
        if not config.openai_key:
            logger.error("OPENAI_API_KEY is missing via env, config, or CLI.")
            return
        llm = OpenAILLM(api_key=config.openai_key, model_name=config.openai_model, stt_interface=stt)
    elif config.llm_provider == "groq":
        if not config.groq_key:
            logger.error("GROQ_API_KEY is missing via env, config, or CLI.")
            return
        llm = GroqLLM(api_key=config.groq_key, model_name=config.groq_model, stt_interface=stt)
    else:
        logger.error(f"Unknown LLM provider: {config.llm_provider}")
        return

    if config.tts_provider == "orpheus":
        from src.modules.tts.orpheus_tts_wrapper import OrpheusTTSWrapper

        tts = OrpheusTTSWrapper(
            api_key=config.orpheus_key,
            endpoint_url=config.orpheus_endpoint,
            voice=config.orpheus_voice,
        )
    elif config.tts_provider == "kokoro":
        from src.modules.tts.kokoro_tts_wrapper import KokoroTTSWrapper

        tts = KokoroTTSWrapper(
            model_path=config.kokoro_model,
            voices_path=config.kokoro_voices_file,
            voice=config.kokoro_voice,
            speed=config.kokoro_speed,
            lang=config.kokoro_lang,
        )
    else:
        from src.modules.tts.edge_tts_wrapper import EdgeTTSWrapper

        tts = EdgeTTSWrapper(
            voice=config.tts_voice,
            pitch=config.tts_pitch,
            rate=config.tts_rate,
            volume=config.tts_volume,
        )

    obs = OBSController(
        host=config.obs_host,
        port=config.obs_port,
        password=config.obs_password,
        source_name=config.obs_avatar_source,
    )

    brain = AIVtuberBrain(config, llm, tts, stt, obs)

    try:
        brain.initialize()
        await brain.start_skills()

        if args.web:
            from src.web.server import run_server

            logger.info("Starting Web Interface at http://localhost:8000")
            await run_server(brain, port=8000)
        else:
            await brain.run_loop()

    except KeyboardInterrupt:
        logger.info("Stopping...")
        try:
            if brain.memory_skill and brain.memory_skill.enabled:
                logger.info("Saving pending memories...")
                await brain.memory_skill.save_all_pending()
        except Exception as exc:
            logger.warning(f"Memory save skipped during shutdown: {exc}")

    finally:
        try:
            await brain.skill_manager.stop()
        except Exception as exc:
            logger.warning(f"Skill manager shutdown warning: {exc}")

        try:
            brain.shutdown()
        except Exception as exc:
            logger.warning(f"Brain shutdown warning: {exc}")


if __name__ == "__main__":
    asyncio.run(main())