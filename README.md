# Voice Controlled Robotic Arm

You say what to pick up, and a cheap robotic arm finds it, grabs it, and drops it on the mat. It
runs on an SO-101 arm I assembled myself and a fine-tuned vision-language-action model, all running
on my laptop.

<!-- Showcase video: in GitHub's web editor you can drag an mp4 onto this line to embed a player. -->
**Showcase video: *

https://github.com/user-attachments/assets/726b167d-a6dd-4802-8a17-194a013c16bf


*

[Model](https://huggingface.co/jayantrathi/smolvla_lang_grasp_v1) ·
[Dataset](https://huggingface.co/datasets/jayantrathi/lang_grasp_v1) ·
[How to run it](#running-it-yourself)

## What it does

You give it a command like "pick up the pen and drop it on the plate," typed or spoken. It finds the
object, picks it up with a grip that suits it (a pinch for a pen, a squeeze for a soft toy), and
drops it on the mat. It handles a few different objects, and it works wherever the object is sitting
in the camera's view, not just one fixed spot. Everything runs on my laptop. Only the training was
done on a rented GPU.

Voice commands go through `voice_grasp.py`, which uses Whisper to turn what I say into the command.

## The build, step by step

I put this together one piece at a time, getting each part working before adding the next.

**1. Started in simulation with RL.** I first tried this as a reinforcement learning project in a
simulator. It was slow to train and fiddly to get working, so I switched to imitation learning,
where the arm learns from example demonstrations instead of trial and error. That switch is what
made everything after it practical.

**2. Built the arm and got it moving.** A self-assembled SO-101 arm, driven by a second "leader"
arm that I move by hand while the main arm copies it.
<!-- ![basic teleop](media/basic-teleop.gif) -->
> _video: first arm movements / leader-follower teleop_

**3. Added VR teleop.** I also set it up so I could drive the arm from a Meta Quest headset, as
another way to move it around. (I had to find and fix a bug in the community VR code to get it
working.)


https://github.com/user-attachments/assets/821306a8-7681-43e2-94fe-b2c7e18cfcfd


> _video: Quest driving the real arm_

**4. Recorded demonstrations.** I moved the arm through the task by hand a bunch of times with a
camera on the wrist, and labeled each recording with what I was doing. I spread the recordings
across different objects and positions rather than doing hundreds of one thing, so the model had
variety to learn from.

https://github.com/user-attachments/assets/e0f9535a-5df1-4ee4-8007-5f41407c10c4


> _video: recording a demonstration_

**5. Trained it.** First a smaller policy (ACT) to check the whole pipeline worked, then a
fine-tuned SmolVLA model for the version that understands language. Training ran on a rented A100.
<!-- ![training](media/training.gif) -->

> _video: training / RunPod timelapse_

**6. Ran it.** The trained model driving the arm on its own from a command, next to what the arm's
own camera sees.
<!-- ![running](media/running.gif) -->
> _video: an autonomous run, outside view and robot camera_

## Setup

- **Arm:** a self-assembled SO-101 with Feetech serial-bus servos and a camera mounted on the wrist.
- **Software:** Hugging Face LeRobot for recording, training, and running.
- **Model:** SmolVLA, a 450M-parameter vision-language-action model, fine-tuned from the base
  checkpoint. It takes the camera image, the arm's joint positions, and the spoken command, and
  outputs the next moves.

## How it was trained

- About 90 recorded demonstrations across three objects (a soft bear, a pen, a small tube), each
  labeled with its command, with the object placed in different spots.
- Fine-tuned SmolVLA for 20k steps on one A100, roughly four hours. The trained model went up to the
  Hugging Face Hub.
- Runs in real time on the laptop GPU.

## What works

- It picks up the object you name, with a grip that fits it.
- It works wherever the object is in the camera's view, not just one memorized spot.
- It runs on its own, on the laptop, from a typed or spoken command.

## What doesn't work yet

- **Two objects at once.** With one object in the scene it's reliable. Put two things in front of it
  and it gets indecisive, because I only trained it on scenes with a single object, so it never
  learned to use the command to pick between them. That's the next thing to fix.
- **Slippery objects.** It reaches for a slick tube correctly but the gripper can lose its hold.
  That's a grip problem, not a brain problem, and grippier pads would fix it.
- **One camera.** The wrist camera loses sight of the object right at the end of the reach, which
  makes the grab less precise at the edges of the workspace.

## Files in this repo

```
README.md                 this file
requirements.txt          dependencies
voice_grasp.py            voice control (mic to Whisper to arm)
scripts/record.sh         record demonstrations
scripts/train_smolvla.sh  train the model on a GPU
scripts/deploy.sh         run the model on the arm from a typed command
media/                    videos and gifs used above
```

The trained model and the recordings live on the Hugging Face Hub (linked at the top), not in here,
since they are big files rather than code.

## Running it yourself

```bash
pip install -r requirements.txt

# 1. Record demonstrations (one object per batch, add --resume and a new label for the next)
./scripts/record.sh "Pick up the pen and drop it on the plate" 30
./scripts/record.sh "Pick up the bear and drop it on the plate" 30 --resume

# 2. Train on a GPU (uploads the model when done)
./scripts/train_smolvla.sh

# 3. Run it on the arm
./scripts/deploy.sh "Pick up the pen and drop it on the plate"

# or by voice
python voice_grasp.py
```

## A few choices I made

- **Imitation learning over reinforcement learning**, because learning from a handful of
  demonstrations is far more practical on a real arm than millions of trial-and-error attempts.
- **Fine-tuning a pretrained model instead of one policy per object**, so adding a new object needs
  a lot less data.
- **Getting one object at one spot working first**, then position, then language, rather than trying
  to do everything at once.

## What's next

1. Telling two objects apart, using an off-the-shelf object detector so it scales to new objects.
2. Turning to find an object that starts outside the camera's view.
3. Keeping track of the object if I move it while it's reaching.
4. Handing the object to my hand instead of dropping it on the mat.

## Adding the videos

For the slots above, either drop a short GIF into a `media/` folder and uncomment the matching
`![...]` line, or open this README in GitHub's web editor and drag an mp4 onto the spot to embed it.
