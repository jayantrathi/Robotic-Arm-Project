# Voice Controlled Robotic Arm

I built a robot arm that picks up the object you ask it for. Right now I give it the command by
typing it ("pick up the pen and drop it on the plate") and it finds the object, grabs it, and drops
it on the mat. Voice is the goal and the next thing I'm adding: turning a spoken command into that
same instruction with Whisper. The arm is a SO-101 I put together myself, driven by a
vision-language model I fine-tuned, and all of it runs on my laptop.

<div align="center">
  <video src="https://github.com/user-attachments/assets/726b167d-a6dd-4802-8a17-194a013c16bf" width="320" controls muted></video>
</div>

[Model](https://huggingface.co/jayantrathi/smolvla_lang_grasp_v1) ·
[Dataset](https://huggingface.co/datasets/jayantrathi/lang_grasp_v1) ·
[How to run it](#running-it-yourself)

## What it does

You give it a command like "pick up the pen and drop it on the plate." It figures out where the pen
is, picks it up with a grip that makes sense for it (a pinch for the pen, a squeeze for the bear),
and drops it on the coaster. It handles a handful of objects, and the object can be sitting anywhere the
camera can see it, not just one specific spot. Everything runs on my laptop. The only thing that
needed a real GPU was training.

For now I type the command in. Wiring up spoken commands is the next piece: `voice_grasp.py` is
written to do it (Whisper turns speech into the same command string), but I haven't recorded a real
voice demo yet, so treat voice as coming soon, not done.

## How I got here

I did this in stages:

**Started in a simulator.** My first attempt was reinforcement learning in sim. It was slow and a
pain to get anywhere with, so I gave up on it and switched to imitation learning, where the arm
just copies demonstrations instead of figuring everything out by trial and error. That was the turn
that actually got things moving.

**Built the arm and got it moving.** It's a self-assembled SO-101. Early on I drove it with a
second "leader" arm that I move by hand while the main arm mirrors it.

**Tried driving it in VR.** I also wired it up so I could move it from a Quest 2 headset. Definitely interesting to see work, but
too finicky to sit there and record hundreds of demos with, so I went back to the leader arm for
the real data recording.

<div align="center">
  <video src="https://github.com/user-attachments/assets/821306a8-7681-43e2-94fe-b2c7e18cfcfd" width="320" controls muted></video>
</div>

**Recorded a bunch of demos.** I moved the arm through the task by hand over and over with a camera
on the wrist, and tagged each recording with what I was doing. I deliberately spread them across
different objects and positions so the model had some variety to work with when faced with a new situation.

<div align="center">
  <video src="https://github.com/user-attachments/assets/e0f9535a-5df1-4ee4-8007-5f41407c10c4" width="320" controls muted></video>
</div>

**Trained it.** First a smaller policy (ACT) trained on a rented 4090 just to prove the pipeline worked end to end, then a
fine-tuned SmolVLA model for the version that actually understands what I'm saying. Training ran on
a rented A100.

<!-- drag your training timelapse mp4 here in GitHub's web editor -->

**Ran it.** The trained model drives the arm on its own from a command. That's the clip at the top.

## The setup

- **Arm:** a self-assembled SO-101 with Feetech serial-bus servos and a camera on the wrist.
- **Software:** Hugging Face LeRobot for recording, training, and running everything.
- **Model:** SmolVLA, a 450M-parameter vision-language-action model, fine-tuned from the base
  checkpoint. It takes the camera image, the arm's joint positions, and the command, and spits out
  the next moves.

## Training details

- Around 90 demos across three objects (a soft bear, a pen, a small tube), each tagged with its
  command, object moved around between recordings.
- Fine-tuned SmolVLA for 20k steps on one A100, about four hours. Trained model is up on the Hugging
  Face Hub.
- Runs in real time on the laptop GPU.

## What works and what doesn't

Works:
- It grabs the object you asked for, with a grip that suits it.
- It works wherever the object is in view, not just one memorized spot.
- It runs by itself on the laptop from a typed command.

Not yet:
- **Voice.** Commands are typed for now. The Whisper script is written but I haven't recorded a
  working voice demo, so this is the next thing.
- **Two objects at once.** One object in the scene and it's solid. Put two down and it becomes unsure and finicky,
  because I only trained on single object scenes, so it never had to use the words to choose between
  them.
- **Rubber Grips.** Main problem arises whenever the robot tries to grab anything with a smooth/plastic surface,
  and the only way around that rubber grip ends. 
- **One camera.** The wrist camera loses sight of the object right at the end of the reach, so the
  grab gets less precise near the edges of the workspace. Plus 2 cameras can help with dynamic situations where
  the drop off zone is moving and so is the object to move.

## What's in this repo

```
README.md                 
requirements.txt          dependencies
voice_grasp.py            voice control (mic -> Whisper -> arm)
scripts/record.sh         record demonstrations
scripts/train_smolvla.sh  train the model on a GPU
scripts/deploy.sh         run the model on the arm from a typed command
media/                    videos used above
```

The model and the recordings live on the Hugging Face Hub (linked up top)

## Running it yourself

```bash
pip install -r requirements.txt

# 1. Record demos (one object per batch, add --resume and a new label for the next one)
./scripts/record.sh "Pick up the pen and drop it on the plate" 30
./scripts/record.sh "Pick up the bear and drop it on the plate" 30 --resume

# 2. Train on a GPU (uploads the model when it's done)
./scripts/train_smolvla.sh

# 3. Run it on the arm from a typed command
./scripts/deploy.sh "Pick up the pen and drop it on the plate"
```

## A couple of decisions

- **Imitation learning instead of RL,** because copying a few demonstrations is way more practical
  on a real arm than millions of trial-and-error attempts that felt like they werent really going anywhere.
- **Fine-tuning a pretrained model instead of one model per object,** so adding a new object takes a
  lot less data and it allows the robot to be able to become a little more comfortable in new situations.
- **Getting one object at one spot rock solid first,** then position, then the command, instead of
  trying to do all of it at once, it just created for a smoother pipeline that I could then execute and understand.

## What's next

1. Voice: recording and wiring up the spoken-command version
2. Telling two objects apart, using an off-the-shelf object detector so it scales to new objects
   without retraining.
3. Turning to find an object that starts outside the camera's view.
4. Keeping track of the object if I move it mid-reach.
5. Handing the object to my hand instead of dropping it on the mat.
