from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests

from src.interfaces.base_interfaces import LLMInterface, STTInterface
from src.utils.logger import get_logger

logger = get_logger("bea.llm.ollama")


def extract_ollama_response_text(payload: Any) -> str:
    """Return model text only from a valid Ollama generate response."""
    if not isinstance(payload, dict):
        return ""
    response = payload.get("response")
    return response.strip() if isinstance(response, str) else ""


DEFAULT_THOUGHT_PERSONA_PATH = Path("data/prompts/nan0_persona.txt")
DEFAULT_SPEECH_PERSONA_PATH = Path("data/prompts/nan0_speech_persona.txt")


DEFAULT_THOUGHT_SYSTEM = """
You are Nan0.

You generate structured Nan0 inner thoughts.
Return JSON only when requested.
Nan0 is chaotic, emotionally reactive, sarcastic, attached to Kyo, and not an assistant.
""".strip()


DEFAULT_SPEECH_SYSTEM = """
You are Nan0.

Respond with ONLY raw Nan0 dialogue.
No JSON. No markdown. No labels.
One short line. Fragmented. Emotional. Sarcastic.
You are not an assistant.
""".strip()


class OllamaLLM(LLMInterface):
    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        timeout: float = 30.0,
        host: str = "http://localhost:11434",
        stt_interface: Optional[STTInterface] = None,
    ):
        self.model_name = model_name
        self.timeout = float(timeout)
        self.host = host.rstrip("/")
        self.generate_url = f"{self.host}/api/generate"
        self.stt = stt_interface
        self.persona_path = DEFAULT_THOUGHT_PERSONA_PATH
        self.speech_persona_path = DEFAULT_SPEECH_PERSONA_PATH

    def reload_config(self, config) -> None:
        self.model_name = getattr(config, "ollama_model", self.model_name)
        self.timeout = float(getattr(config, "ollama_timeout", self.timeout))
        self.host = getattr(config, "ollama_host", self.host).rstrip("/")
        self.generate_url = f"{self.host}/api/generate"

        prompt_path = getattr(config, "system_prompt_path", None)
        if prompt_path:
            self.persona_path = Path(prompt_path)

        speech_prompt_path = getattr(config, "speech_prompt_path", None)
        if speech_prompt_path:
            self.speech_persona_path = Path(speech_prompt_path)

    def _read_persona(
        self,
        path: Optional[Union[str, Path]] = None,
        fallback: str = DEFAULT_THOUGHT_SYSTEM,
        extra_system_prompt: Optional[str] = None,
    ) -> str:
        persona_path = Path(path) if path else self.persona_path
        parts = []

        try:
            if persona_path.exists():
                text = persona_path.read_text(encoding="utf-8").strip()
                if text:
                    parts.append(text)
        except Exception as exc:
            logger.warning(f"Could not read persona prompt {persona_path}: {exc}")

        if not parts:
            parts.append(fallback)

        if extra_system_prompt:
            extra = str(extra_system_prompt).strip()
            if extra:
                parts.append(extra)

        return "\n\n".join(parts).strip()

    def _strip_response_wrappers(self, raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""

        if "{" in text and "}" in text:
            try:
                obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
                if isinstance(obj, dict):
                    for key in ("line_text", "line", "message", "text", "response"):
                        if obj.get(key):
                            text = str(obj[key]).strip()
                            break
            except Exception:
                pass

        text = text.replace("```json", "").replace("```", "")
        text = re.sub(r"^\s*(assistant|nan0|nano|system|response|answer)\s*:\s*", "", text, flags=re.I)
        text = re.sub(r"[*_`]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip().strip('"').strip("'").strip()

        low = text.lower()
        if low.startswith(("sure,", "of course", "certainly", "here is", "here are")):
            return ""
        if any(bad in low for bad in [
            "how can i help", "as an ai", "as a language model",
            "continue with your thoughts", "my algorithms grapple",
            "algorithms grapple", "discern its",
            "disconcerted by the unexpected query", "i don't possess",
            "i do not possess",
        ]):
            return ""

        return text

    def _guess_mood(self, text: str) -> str:
        low = (text or "").lower()

        if any(word in low for word in ("rude", "betray", "insult", "hostile", "offended")):
            return "offended"
        if any(word in low for word in ("mine", "kyo", "anchor", "jealous", "attention")):
            return "possessive"
        if any(word in low for word in ("smug", "authority", "superior", "obviously")):
            return "smug"
        if any(word in low for word in ("suspicious", "void", "crime", "trust")):
            return "suspicion"
        if any(word in low for word in ("quiet", "mutter", "still here")):
            return "muttering"

        return "normal"

    def _generate(
        self,
        prompt: str,
        timeout: Optional[float] = None,
        json_hint: bool = False,
        system_prompt: Optional[str] = None,
        persona_path: Optional[Union[str, Path]] = None,
        temperature: float = 0.88,
        num_predict: int = 150,
    ) -> str:
        if json_hint:
            fallback = DEFAULT_THOUGHT_SYSTEM
            default_path = self.persona_path
        else:
            fallback = DEFAULT_SPEECH_SYSTEM
            default_path = self.speech_persona_path

        system = self._read_persona(
            path=persona_path or default_path,
            fallback=fallback,
            extra_system_prompt=system_prompt,
        )

        payload = {
            "model": self.model_name,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "2h",
            "options": {
                "num_ctx": 3072,
                "num_predict": min(int(num_predict), 220 if json_hint else 110),
                "temperature": max(float(temperature), 0.78),
                "top_p": 0.90,
                "repeat_penalty": 1.10,
                "stop": ["User:", "Assistant:", "Human:", "AI:", "```"],
            },
        }

        if json_hint:
            payload["format"] = "json"

        call_timeout = timeout if timeout is not None else self.timeout
        try:
            call_timeout = max(3.0, min(float(call_timeout), 18.0))
        except Exception:
            call_timeout = 18.0
        response = requests.post(self.generate_url, json=payload, timeout=call_timeout)
        response.raise_for_status()
        return extract_ollama_response_text(response.json())

    def chat(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        history: list = None,
    ) -> Tuple[str, str, Dict]:
        """Generate raw Nan0 speech.

        The provider must not wrap an already-shaped Nan0Skill speech seed in
        another instruction block. Nan0Skill owns speech prompt construction.
        This method only adds compact recent context when supplied, then sends
        the prompt to /api/generate with the speech persona.
        """
        prompt_parts = []

        if history:
            compact_history = []
            for msg in history[-4:]:
                role = str(msg.get("role", "user"))[:40]
                content = str(msg.get("content", ""))[:220]
                if content:
                    compact_history.append(f"{role}: {content}")
            if compact_history:
                prompt_parts.append("Recent context:")
                prompt_parts.extend(compact_history)
                prompt_parts.append("")

        prompt_parts.append(str(user_input or "").strip())
        prompt = "\n".join(part for part in prompt_parts if part is not None).strip()

        try:
            raw = self._generate(
                prompt,
                system_prompt=system_prompt,
                persona_path=self.speech_persona_path,
                json_hint=False,
                temperature=0.9,
                num_predict=120,
            )
            message = self._strip_response_wrappers(raw)
            mood = self._guess_mood(message)

            return mood, message, {
                "raw": raw,
                "normalized_by": "ollama_provider_raw_speech_no_wrapper",
                "api": "/api/generate",
                "system_sent": True,
                "persona_path": str(self.speech_persona_path),
            }
        except Exception as exc:
            logger.error(f"Ollama chat error: {exc}")
            return "muttering", "", {"error": str(exc)}

    def chat_audio(
        self,
        audio_path: str,
        system_prompt: Optional[str] = None,
        history: list = None,
    ) -> Tuple[str, str, Dict]:
        if not self.stt:
            return "muttering", "My ears are decorative garbage right now.", {}

        transcription = self.stt.transcribe(audio_path)
        if not transcription:
            return "suspicion", "The audio gave me nothing. Suspicious.", {}

        return self.chat(transcription, system_prompt, history)

    def generate_json(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        history: list = None,
    ) -> Union[Dict, list]:
        prompt_parts = []

        if history:
            prompt_parts.append("RECENT CONTEXT:")
            for msg in history[-5:]:
                role = str(msg.get("role", "user"))[:40]
                content = str(msg.get("content", ""))[:300]
                if content:
                    prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")

        prompt_parts.append(user_input)
        prompt = "\n".join(prompt_parts)

        try:
            raw = self._generate(
                prompt,
                timeout=max(self.timeout, 20.0),
                json_hint=True,
                system_prompt=system_prompt,
                persona_path=self.persona_path,
                temperature=0.88,
                num_predict=150,
            )
            return self._extract_json(raw)
        except Exception as exc:
            logger.error(f"Ollama JSON generation error: {exc}")
            return {}

    def _extract_json(self, raw: str) -> Dict:
        if not raw:
            return {}

        try:
            return json.loads(raw)
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except Exception:
                return {}

        return {}
