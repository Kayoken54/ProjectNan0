
from __future__ import annotations
import json
from src.modules.skills.implementations.nan0_cognition_router_v1 import route_vision_state_file

result = route_vision_state_file()
print('route:', result.get('route'))
print('seed:', result.get('seed'))
print('model:', result.get('model'))
print('used_llm:', result.get('used_llm'))
print('line:', result.get('line'))
