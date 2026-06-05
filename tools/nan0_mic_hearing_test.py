"""
Nan0 mic hearing test.
This does NOT transcribe. It proves Windows and Python can hear a microphone.
Run:
  python tools/nan0_mic_hearing_test.py --list
  python tools/nan0_mic_hearing_test.py --device 1 --seconds 5
If the level rises while you talk, Nan0 can hear that device.
"""
import argparse
import math
import time

try:
    import sounddevice as sd
except Exception as e:
    print("sounddevice import failed:", e)
    raise SystemExit(1)


def list_devices():
    print(sd.query_devices())


def test_device(device, seconds):
    samplerate = 16000
    block = 1024
    print(f"Testing mic device={device} for {seconds}s. Talk now.")
    print("If bars move, the mic is being heard. Ctrl+C to stop.")
    start = time.time()
    try:
        with sd.InputStream(device=device, channels=1, samplerate=samplerate, blocksize=block) as stream:
            while time.time() - start < seconds:
                data, overflowed = stream.read(block)
                rms = math.sqrt(float((data ** 2).mean()))
                db = 20 * math.log10(max(rms, 1e-8))
                bars = int(max(0, min(40, (db + 60) / 2)))
                print(f"level {db:6.1f} dB |" + "#" * bars)
    except Exception as e:
        print("Mic test failed:", e)
        raise SystemExit(2)
    print("Done. If the bars moved when you spoke, the mic device works.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    if args.list:
        list_devices()
    else:
        test_device(args.device, args.seconds)
