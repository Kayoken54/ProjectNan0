
from __future__ import annotations
import json
from src.modules.skills.implementations.nan0_thought_engine_v3 import update_state_file

state = update_state_file()
pkt = state.get("thought_packet")
print("screen_state:", state.get("screen_state") or state.get("state"))
print("motion:", state.get("motion_intensity") or state.get("motion"))
print("combat:", state.get("combat"))
print("menu_open:", state.get("menu_open"))
print("new_thought:", bool(state.get("new_thought")))
print("speech_allowed:", bool(state.get("speech_allowed")))
print("thought_seed:", state.get("thought_seed") or "")
print("thought_text:", (pkt or {}).get("thought_text") if pkt else "")
print("thought_packet:")
print(json.dumps(pkt, indent=2) if pkt else "null")
