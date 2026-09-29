#!/usr/bin/env python3
"""
Voice-commanded grasping for the SO-101 + SmolVLA arm.

Pipeline:  microphone  ->  Whisper (speech-to-text)  ->  map the words to a
           task string the policy was trained on  ->  launch lerobot-rollout
           so the arm executes it.

Usage:
    python voice_grasp.py
    # press Enter, say e.g. "grab the pen", watch the arm do it, repeat.
    # Ctrl-C to quit.
"""

import subprocess

import sounddevice as sd
import whisper  # the `openai-whisper` package imports as `whisper`

# ---- Config: edit these to match your setup -------------------------------
FOLLOWER_PORT = "/dev/tty.usbmodem5B790815221"
FOLLOWER_ID = "my_follower_arm"
CAMERA_INDEX = 0
POLICY_PATH = "jayantrathi/smolvla_lang_grasp_v1"
ROLLOUT_DURATION = 25          # seconds the arm runs per command
RECORD_SECONDS = 4             # how long we listen after you press Enter
SAMPLE_RATE = 16000            # Whisper expects 16 kHz mono audio
WHISPER_MODEL = "base"         # tiny / base / small — base is a good tradeoff

# Map a spoken keyword -> the exact instruction the policy was trained on.
# SmolVLA learned these specific phrasings, so we normalize whatever you say
# ("grab the pen", "get the bear") down to the trained instruction.
TASKS = {
    "pen": "Pick up the pen and drop it on the plate",
    "bear": "Pick up the bear and drop it on the plate",
    "lip balm": "Pick up the lip balm and drop it on the plate",
    "balm": "Pick up the lip balm and drop it on the plate",
    "chapstick": "Pick up the lip balm and drop it on the plate",
}
# ---------------------------------------------------------------------------


def record_command(model) -> str:
    """Record a few seconds from the mic and return the transcribed text."""
    input("\nPress Enter, then speak your command...")
    print(f"Listening for {RECORD_SECONDS}s...")
    # sounddevice returns float32 samples already scaled to [-1, 1],
    # which is exactly what Whisper wants, so we can hand it straight over.
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()  # block until the recording finishes

    print("Transcribing...")
    result = model.transcribe(audio.flatten(), fp16=False)  # fp16=False for CPU/Mac
    text = result["text"].strip()
    print(f'  heard: "{text}"')
    return text.lower()


def pick_task(text: str):
    """Return the trained instruction for whichever object was named, or None."""
    for keyword, task in TASKS.items():
        if keyword in text:
            return task
    return None


def run_arm(task: str) -> None:
    """Launch lerobot-rollout with the chosen task string."""
    cameras = (f"{{ camera1: {{type: opencv, index_or_path: {CAMERA_INDEX}, "
               f"width: 640, height: 480, fps: 30}} }}")
    cmd = [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={POLICY_PATH}",
        "--policy.device=mps",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_ID}",
        f"--robot.cameras={cameras}",
        f"--task={task}",
        "--inference.type=rtc",
        f"--duration={ROLLOUT_DURATION}",
        "--display_data=true",
    ]
    print(f'\nRunning: "{task}"\n')
    subprocess.run(cmd)


def main() -> None:
    print(f"Loading Whisper ({WHISPER_MODEL})... (first run downloads the model)")
    model = whisper.load_model(WHISPER_MODEL)
    print("Ready. Ctrl-C to quit.")
    while True:
        try:
            text = record_command(model)
            task = pick_task(text)
            if task is None:
                print("  Didn't catch a known object (pen / bear / lip balm). Try again.")
                continue
            run_arm(task)
        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
