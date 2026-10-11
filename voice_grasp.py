#!/usr/bin/env python3
import os
import subprocess
import sys

import sounddevice as sd
import whisper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))
from objects import REGISTRY, resolve

RECORD_SECONDS = 4
SAMPLE_RATE = 16000
WHISPER_MODEL = "base"
PYTHON = sys.executable
DEPLOY = os.path.join(os.path.dirname(__file__), "deploy_select.py")
USE_MASK = True


def record_command(model) -> str:
    input("\nPress Enter, then speak your command...")
    print(f"Listening for {RECORD_SECONDS}s...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Transcribing...")
    result = model.transcribe(audio.flatten(), fp16=False)
    text = result["text"].strip()
    print(f'  heard: "{text}"')
    return text.lower()


def run_arm(object_name: str) -> None:
    cmd = [PYTHON, DEPLOY, object_name]
    if USE_MASK:
        cmd.append("--mask")
    print(f'\nGrasping: "{object_name}"\n')
    subprocess.run(cmd)


def main() -> None:
    print(f"Loading Whisper ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"Ready. Known objects: {', '.join(REGISTRY)}. Ctrl-C to quit.")
    while True:
        try:
            text = record_command(model)
            obj = resolve(text)
            if obj is None:
                print(f"  Didn't catch a known object ({', '.join(REGISTRY)}). Try again.")
                continue
            run_arm(obj.name)
        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
