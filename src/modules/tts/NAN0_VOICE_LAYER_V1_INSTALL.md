# Nan0 Voice Layer V1 for ProjectBEA

This is a drop-in voice delivery layer. It does not replace ProjectBEA TTS.

## What it does

ProjectBEA keeps using its existing TTS engines:

- EdgeTTS
- Kokoro
- Orpheus
- anything else already wired in `src/modules/tts/`

This layer only formats Nan0's final line before the existing TTS call.

```text
Nan0 raw line -> Nan0 voice formatter -> existing ProjectBEA TTS
```

## Files to copy

Copy these into ProjectBEA:

```text
src/modules/tts/nan0_voice_formatter.py
src/modules/tts/nan0_voice_integration.py
```

## Config snippet

Add this to `config.json` at the top level:

```json
"nan0_voice": {
  "enabled": true,
  "max_spoken_chars": 220,
  "allow_pauses": true,
  "soft_kyo_mode": true,
  "log_voice_formatting": true,
  "remove_markdown": true,
  "collapse_repeated_punctuation": true,
  "preserve_nan0_attitude": true
}
```

If you are worried about JSON commas, add it after an existing block and make sure the previous block ends with a comma.

## Integration point

Find the place where ProjectBEA sends text to TTS. It will look roughly like one of these:

```python
await self.tts.speak(message)
```

or:

```python
await tts_engine.synthesize(text)
```

or:

```python
self.tts.speak(response_text)
```

Right before that call, add:

```python
from src.modules.tts.nan0_voice_integration import prepare_nan0_tts_text

spoken_text = prepare_nan0_tts_text(
    raw_line=response_text,
    mood=getattr(self, "mood", None),
    target="kyo",
    runtime_config=getattr(self, "config", None),
)

if spoken_text:
    await self.tts.speak(spoken_text)
```

If your TTS call is not async, use:

```python
if spoken_text:
    self.tts.speak(spoken_text)
```

## Safer generic integration

If you do not know the current variable names, use the smallest possible change:

```python
from src.modules.tts.nan0_voice_formatter import format_nan0_voice

text = format_nan0_voice(text, mood="smug", target="kyo")
```

Put that immediately before the existing TTS call.

## What to test

Run:

```powershell
python main.py --web
```

Then make Nan0 speak. Look for a log like:

```text
Nan0 voice formatter: raw='...' spoken='...'
```

## Expected result

Raw:

```text
The screen is thrashing. Kyo is either fighting something or feeding the disaster engine.
```

Spoken:

```text
Kyo. The screen is thrashing... Kyo is either fighting something or feeding the disaster engine.
```

## Rollback

Delete these files:

```text
src/modules/tts/nan0_voice_formatter.py
src/modules/tts/nan0_voice_integration.py
```

Then remove the one integration call before TTS.

No ProjectBEA TTS engine files need to be replaced.
