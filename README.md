# Language-Controlled Robotic Arm (SO-101 + SmolVLA)

A low-cost 6-DoF robotic arm that picks up an object you name and places it, driven
by a fine-tuned vision-language-action (VLA) foundation model running in real time
on a laptop.

**Demo:** _[add video link once hosted]_
**Artifacts:** dataset `huggingface.co/datasets/jayantrathi/lang_grasp_v1` · model `huggingface.co/jayantrathi/smolvla_lang_grasp_v1`

## What it does today

Given a natural-language command such as "pick up the pen and drop it on the plate,"
the arm locates the named object, grasps it with a grip appropriate to that object,
and places it on a target zone. It works across three objects (a plush bear, a pen,
and a moisturizer tube) and generalizes to arbitrary object positions in the
workspace, not a single fixed spot. Inference runs on-device on an Apple Silicon
laptop; training was done on a rented cloud GPU. Commands can be typed or spoken
(via `voice_grasp.py`, which uses Whisper for speech-to-text).

## System

- **Hardware:** a self-assembled SO-101 (SO-ARM101) arm with Feetech serial-bus
  servos and a wrist-mounted (eye-in-hand) USB camera. Leader-follower teleoperation
  was used to collect demonstrations.
- **Software:** Hugging Face LeRobot for data recording, training, and deployment.
- **Policy:** SmolVLA, a ~450M-parameter vision-language-action model, fine-tuned
  from the `smolvla_base` checkpoint. The model takes the camera image, the arm's
  joint state, and the language instruction, and outputs a chunk of future actions.

## Method

- **Data (breadth over depth):** ~90 teleoperated demonstrations across the three
  objects, each episode labeled with its language instruction, with the object placed
  at varied positions. Rather than hundreds of demos of a single object, coverage was
  spread across objects and positions so the model's pretraining could fill the gaps.
- **Training:** fine-tuned `smolvla_base` for 20k steps on a single A100 (about four
  hours). The trained model was pushed to the Hugging Face Hub.
- **Deployment:** run in real time on the laptop GPU using LeRobot's real-time
  chunking (RTC) inference to keep motion smooth despite the model running below
  camera frame rate.

## Voice control

`voice_grasp.py` adds spoken commands: it records a few seconds from the microphone,
transcribes with Whisper, maps the spoken keyword (pen / bear / lip balm) to the exact
instruction the policy was trained on, and launches the rollout. Run it with:

```bash
pip install openai-whisper sounddevice
python voice_grasp.py
```

## Repository contents

```
README.md                 this file
requirements.txt          Python dependencies
voice_grasp.py            voice-commanded control (mic -> Whisper -> policy)
scripts/record.sh         record leader-follower demonstrations
scripts/train_smolvla.sh  fine-tune SmolVLA on a GPU
scripts/deploy.sh         run the policy on the arm for a typed command
```

The trained model and dataset are hosted on the Hugging Face Hub (linked above),
not committed to this repo (they are ~1 GB of weights and video, and are artifacts
rather than source).

## Reproduce the full pipeline

```bash
pip install -r requirements.txt

# 1. Record demonstrations (one object per batch; repeat with --resume + a new label)
./scripts/record.sh "Pick up the pen and drop it on the plate" 30
./scripts/record.sh "Pick up the bear and drop it on the plate" 30 --resume
./scripts/record.sh "Pick up the lip balm and drop it on the plate" 30 --resume

# 2. Fine-tune SmolVLA on a GPU (e.g. a rented A100), auto-pushes the model to the Hub
./scripts/train_smolvla.sh

# 3. Run it on the arm (typed command)
./scripts/deploy.sh "Pick up the pen and drop it on the plate"

# ...or by voice
python voice_grasp.py
```

## Key design decisions

- **Imitation learning, not reinforcement learning.** RL is sample-inefficient and
  painful on real hardware; imitation learning fits a policy from a modest number of
  human demonstrations, which is far more practical on a physical arm. (An earlier
  from-scratch RL project in simulation motivated this switch.)
- **A pretrained VLA, not a per-object policy.** Training a separate policy per object
  does not scale. Fine-tuning a pretrained vision-language model adapts general
  object, grasp, and language knowledge to this specific arm with far less data.
- **Single-mode first, then generalize.** The task was built up one variable at a
  time: first a reliable grasp of one object at one position, then position
  generalization, then multiple objects with language labels. Each stage was proven
  before adding the next.

## Results

- Reliable language-conditioned pick-and-place for the trained objects, with
  object-appropriate grasps (a pinch for the pen, a squeeze for the bear).
- Position generalization: the object can start anywhere in the camera's view, not a
  memorized location.
- Real-time autonomous execution on-device, from typed or spoken commands.

## Known limitations (honest)

- **Multi-object selection is not solved yet.** With a single object in the scene the
  system is reliable, but when two objects are present at once it becomes indecisive,
  because it was trained only on single-object scenes and never had to use the
  instruction to disambiguate among distractors. This is the next milestone.
- **Low-friction objects.** The reach and approach are correct for all objects, but a
  slippery tube can slip from the gripper. This is a gripper-friction limitation, not
  a perception or policy failure; friction pads on the fingers would address it.
- **Single eye-in-hand camera.** The wrist camera loses sight of the object during the
  final descent, which limits precision at the edges of the workspace.

## Roadmap

1. **Multi-object selection via a zero-shot open-vocabulary detector.** Offload "which
   object" to a detector (for example OWL-ViT or YOLO-World) that finds the named
   object with no additional training, isolate it, and hand the grasp to the existing
   policy. This scales to new objects for free rather than retraining per object.
2. **Grasp refinement.** Gripper friction pads and additional object variety.
3. **Extended voice interaction** building on `voice_grasp.py`.
