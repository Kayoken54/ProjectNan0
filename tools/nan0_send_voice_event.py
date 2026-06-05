import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["kyo", "discord"], required=True)
ap.add_argument("--speaker", default=None)
ap.add_argument("--text", required=True)
args = ap.parse_args()

if args.source == "kyo":
    path = Path("data/input/kyo_voice_inbox.jsonl")
    speaker = args.speaker or "Kyo"
else:
    path = Path("data/input/discord_voice_inbox.jsonl")
    speaker = args.speaker or "Friend"

path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps({
        "source": args.source,
        "speaker": speaker,
        "text": args.text,
        "timestamp": time.time(),
    }) + "\n")
print(f"sent {args.source} event from {speaker}: {args.text}")
print(f"wrote {path}")
