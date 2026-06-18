import asyncio
import json
import subprocess
import os
import signal
import aiohttp
import sys
from typing import Optional
from src.core.config import BrainConfig
from src.modules.skills.base_skill import BaseSkill
from pathlib import Path

class DiscordSkill(BaseSkill):
    def __init__(self, name: str, config: BrainConfig, context):
        super().__init__(name, config, context)
        self.bot_process: Optional[subprocess.Popen] = None

        # FIX: Resolve bot_dir properly. Try multiple locations.
        skill_dir = Path(__file__).parent
        possible_dirs = [
            skill_dir / "bot",                          # src/modules/skills/bot/
            skill_dir.parent / "discord" / "bot",       # src/modules/discord/bot/
            skill_dir.parent.parent.parent / "bot",     # project_root/bot/
            skill_dir,                                   # Same dir as skill (flat)
            Path(self.skill_config.get("bot_dir", "")), # Config override
        ]

        self.bot_dir = None
        for d in possible_dirs:
            if d.exists() and (d / "index.js").exists():
                self.bot_dir = d
                break
            elif d.exists() and d.is_dir() and any(d.iterdir()):
                # Directory exists but no index.js — might still be right place
                if (d / "package.json").exists():
                    self.bot_dir = d
                    break

        # Fallback: use skill_dir if nothing found (will error on node_modules check)
        if self.bot_dir is None:
            self.bot_dir = skill_dir
            print(f"[DiscordSkill] WARNING: Could not find bot dir with index.js. Falling back to: {self.bot_dir}")
        else:
            print(f"[DiscordSkill] Bot directory resolved to: {self.bot_dir}")

        api_port = config.skills.get("discord", {}).get("api_port", 3030)
        self.api_url = f"http://localhost:{api_port}"

    def initialize(self):
        pass

    async def start(self):
        if self.is_active:
            self.log("Discord Bot already running.")
            return

        # priority: env var → config → None
        discord_token = os.getenv("DISCORD_TOKEN", "")
        if not discord_token:
            discord_token = self.skill_config.get("token", "")

        if not discord_token:
            self.log("Error: Discord Token not configured (Config or Env 'DISCORD_TOKEN').")
            return

        self.log("Starting Discord Bot...")

        # env vars
        api_port = self.skill_config.get("api_port", 3030)
        self.api_url = f"http://localhost:{api_port}"
        env = os.environ.copy()
        env["DISCORD_TOKEN"] = discord_token
        env["PORT"] = str(api_port)
        discord_inbox = Path(self.skill_config.get("discord_inbox", "data/input/discord_voice_inbox.jsonl")).resolve()
        discord_audio_inbox = Path(self.skill_config.get("discord_audio_inbox", "data/input/discord_audio_inbox.jsonl")).resolve()
        discord_audio_dir = Path(self.skill_config.get("discord_audio_dir", "data/input/discord_audio")).resolve()
        discord_voice_outbox = Path(self.skill_config.get("discord_voice_outbox", "data/output/discord_voice_outbox.jsonl")).resolve()
        env["DISCORD_INBOX_PATH"] = str(discord_inbox)
        env["DISCORD_AUDIO_INBOX_PATH"] = str(discord_audio_inbox)
        env["DISCORD_AUDIO_DIR"] = str(discord_audio_dir)
        env["DISCORD_VOICE_OUTBOX_PATH"] = str(discord_voice_outbox)

        # Phase 1 perception: create the bridge files before Node starts so
        # Nan0Skill can prime file offsets without racing missing paths.
        for bridge_path in [discord_inbox, discord_audio_inbox, discord_voice_outbox]:
            try:
                bridge_path.parent.mkdir(parents=True, exist_ok=True)
                bridge_path.touch(exist_ok=True)
            except Exception:
                pass
        try:
            discord_audio_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # check if node_modules exists
        if not (self.bot_dir / "node_modules").exists():
            self.log(f"Error: node_modules not found in {self.bot_dir}. Please run 'npm install' in the bot directory.")
            return

        try:
            # shell=False for correct termination
            self.bot_process = subprocess.Popen(
                ["node", "index.js"],
                cwd=str(self.bot_dir),
                env=env,
                stdout=sys.stdout, 
                stderr=sys.stderr,
                shell=False
            )
            await super().start()  # sets self.is_active = True
            self.log(f"Discord Bot started with PID {self.bot_process.pid}")

        except Exception as e:
            self.log(f"Failed to start Discord Bot: {e}")
            self.is_active = False

    async def stop(self):
        if not self.bot_process:
            self.log("Discord Bot is not running.")
            return

        self.log("Stopping Discord Bot...")
        try:
            # force kill the process
            if self.bot_process:
                self.bot_process.kill()

                # taskkill to be sure
                subprocess.run(f"taskkill /F /T /PID {self.bot_process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                self.bot_process.wait(timeout=2)

            self.bot_process = None
            self.is_active = False
            await super().stop()
        except Exception as e:
            self.log(f"Error stopping Discord Bot: {e}")

    async def update(self):
        # monitor process
        if self.bot_process:
            ret = self.bot_process.poll()
            if ret is not None:
                self.log(f"Discord Bot process exited unexpectedly with code {ret}")
                self.bot_process = None
                self.is_active = False

    async def play_audio_file(self, audio_path: str, speech_packet: dict = None):
        """Queue a validated Nan0 TTS file for the Node Discord voice player."""
        if not self.is_active:
            self.log("Cannot play audio: Bot is offline.")
            return False

        speech_packet = speech_packet or {}
        thought_id = speech_packet.get("thought_id")
        if not thought_id:
            self.log("Refusing Discord VC audio: missing thought_id.")
            return False

        path = Path(audio_path)
        if not path.exists():
            self.log(f"Refusing Discord VC audio: file not found: {audio_path}")
            return False

        try:
            outbox = Path(self.skill_config.get("discord_voice_outbox", "data/output/discord_voice_outbox.jsonl"))
            outbox.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "type": "play_audio_file",
                "audio_path": str(path.resolve()),
                "thought_id": str(thought_id),
                "line_text": str(speech_packet.get("line_text") or ""),
                "mood": str(speech_packet.get("mood") or "normal"),
                "target_actor_id": str(speech_packet.get("target_actor_id") or "unknown"),
                "created_at": asyncio.get_event_loop().time(),
            }
            with outbox.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.log(f"Queued Discord VC audio for thought_id={thought_id}: {path}")
            return True
        except Exception as e:
            self.log(f"Failed to queue Discord VC audio: {e}")
            return False

    async def send_message(self, channel_id: str, content: str):
        if not self.is_active:
            self.log("Cannot send message: Bot is offline.")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/send", 
                    json={"channelId": channel_id, "content": content}
                ) as resp:
                    if resp.status == 200:
                        self.log(f"Message sent to {channel_id}")
                        return True
                    else:
                        text = await resp.text()
                        self.log(f"Failed to send message: {text}")
                        return False
        except Exception as e:
            self.log(f"API Request failed: {e}")
            return False
