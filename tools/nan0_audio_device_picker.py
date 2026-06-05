from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import sounddevice as sd
except Exception as e:
    print(f"sounddevice import failed: {e}")
    raise


def list_devices():
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        host = hostapis[d['hostapi']]['name']
        ins = d.get('max_input_channels', 0)
        outs = d.get('max_output_channels', 0)
        print(f"[{i:02d}] host={host:<18} in={ins:<2} out={outs:<2} rate={d.get('default_samplerate')} name={d['name']}")


def find_device(kind: str, name_hint: str, hostapi_hint: str):
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    matches = []
    for i, d in enumerate(devices):
        host = hostapis[d['hostapi']]['name']
        if name_hint.lower() not in d['name'].lower():
            continue
        if hostapi_hint and hostapi_hint.lower() not in host.lower():
            continue
        if kind == 'input' and d.get('max_input_channels', 0) <= 0:
            continue
        if kind == 'output' and d.get('max_output_channels', 0) <= 0:
            continue
        matches.append((i, host, d))
    return matches


def patch_config(output_id: int | None, input_id: int | None):
    path = Path('config.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    if output_id is not None:
        data['audio_device_id'] = output_id
    if input_id is not None:
        data['stt_input_device_id'] = input_id
    data['audio_device_hint'] = 'HyperX'
    data['audio_hostapi_hint'] = 'MME'
    data['stt_input_device_hint'] = 'HyperX'
    data['stt_hostapi_hint'] = 'MME'
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Updated {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--apply-hyperx-mme', action='store_true')
    ap.add_argument('--output-id', type=int)
    ap.add_argument('--input-id', type=int)
    args = ap.parse_args()

    if args.list or not (args.apply_hyperx_mme or args.output_id is not None or args.input_id is not None):
        list_devices()

    output_id = args.output_id
    input_id = args.input_id

    if args.apply_hyperx_mme:
        outs = find_device('output', 'HyperX', 'MME')
        ins = find_device('input', 'HyperX', 'MME')
        print('\nHyperX MME output matches:', [(i, h, d['name']) for i, h, d in outs])
        print('HyperX MME input matches:', [(i, h, d['name']) for i, h, d in ins])
        if output_id is None and outs:
            output_id = outs[0][0]
        if input_id is None and ins:
            input_id = ins[0][0]

    if output_id is not None or input_id is not None:
        patch_config(output_id, input_id)
        print(f"Selected output={output_id}, input={input_id}")


if __name__ == '__main__':
    main()
