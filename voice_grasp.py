#!/usr/bin/env python3
"""
Voice-commanded grasping for the SO-101 + SmolVLA arm.

Pipeline:  microphone -> Whisper (speech-to-text) -> resolve the words to a
           known object -> run deploy_select.py, which gates on the detector,
           masks out the other objects, and runs the grasp policy.

Because it goes through deploy_select, spoken commands get the same two-object
disambiguation as typed ones -- say "grab the pen" with a bear also on the mat
and it still goes for the pen.

Usage:
    python voice_grasp.py
    # press Enter, say e.g. "grab the pen", watch the arm do it, repeat.
    # Ctrl-C to quit.

Adding a new object is a one-line change in perception/objects.py -- no edits
here.
"""

import os
import subprocess
import sys

import sounddevice as sd
import whisper  # the `openai-whisper` package imports as `whisper`

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))
from objects import REGISTRY, resolve  # noqa: E402

# ---- Config ----------------------------------------------------------------
RECORD_SECONDS = 4             # how long we listen after you press Enter
SAMPLE_RATE = 16000            # Whisper expects 16 kHz mono audio
WHISPER_MODEL = "base"         # tiny / base / small -- base is a good tradeoff
PYTHON = sys.executable        # run deploy_select with the same interpreter
DEPLOY = os.path.join(os.path.dirname(__file__), "deploy_select.py")
USE_MASK = True                # mask out other objects (two-object disambiguation)
# ---------------------------------------------------------------------------


def record_command(model) -> str:
    """Record a few seconds from the mic and return the transcribed text."""
    input("\nPress Enter, then speak your command...")
    print(f"Listening for {RECORD_SECONDS}s...")
    # sounddevice returns float32 samples in [-1, 1], exactly what Whisper wants.
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Transcribing...")
    result = model.transcribe(audio.flatten(), fp16=False)  # fp16=False for CPU/Mac
    text = result["text"].strip()
    print(f'  heard: "{text}"')
    return text.lower()


def run_arm(object_name: str) -> None:
    """Hand the chosen object to deploy_select (gate -> mask -> grasp policy)."""
    cmd = [PYTHON, DEPLOY, object_name]
    if USE_MASK:
        cmd.append("--mask")
    print(f'\nGrasping: "{object_name}"\n')
    subprocess.run(cmd)


def main() -> None:
    print(f"Loading Whisper ({WHISPER_MODEL})... (first run downloads the model)")
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
