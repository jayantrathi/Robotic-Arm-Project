# Voice Controlled Robotic Arm

I built a robot arm that picks up the object you ask it for. You can type the command or **say it**
("grab the pen") and it finds the object, grabs it, and drops it on the coaster. It can pick the right
object out of a scene with **several objects**, and if the thing you asked for isn't even in the
camera's view it will **pan around to look for it** first. The arm is a SO-101 I put together
myself, driven by a vision-language model I fine-tuned, and all of it runs on my laptop.

<div align="center">
  <video src="https://github.com/user-attachments/assets/726b167d-a6dd-4802-8a17-194a013c16bf" width="320" controls muted></video>
</div>

[Model](https://huggingface.co/jayantrathi/smolvla_lang_grasp_v2) ·
[Dataset](https://huggingface.co/datasets/jayantrathi/lang_grasp_v2) ·
[How to run it](#running-it-yourself)

## What it does

You give it a command like "pick up the pen and drop it on the plate." It figures out where the pen
is, picks it up with a grip that makes sense for it (a pinch for the pen, a squeeze for the bear),
and drops it on the coaster. It handles a handful of objects, and the object can be sitting anywhere the
camera can see it, not just one specific spot. Everything runs on my laptop. The only thing that
needed a real GPU was training.

On top of the core grasp policy I added a perception layer (an open-vocabulary detector) that does
three things the policy couldn't on its own:

- **Say it out loud.** `voice_grasp.py` turns a spoken command into the same instruction with Whisper.
- **Pick one object out of many.** With two objects in frame the policy used to dither (it was only
  trained on single-object scenes). Now a detector finds the one you named and blurs the others out
  of the camera feed, so the policy sees the clean single-object scene it's good at, with no retraining,
  and adding a new object is one line in `perception/objects.py`.
- **Look for it if it's off-screen.** If the object isn't in view, the arm pans to find it and
  centers on it before grasping.

## Results

Measured on the real arm. **20 trials per condition**, object position varied every trial. Success
is the named object ending up on the plate.

Two headline jumps: **picking the right object in a two-object scene went 25% → 85%**, and
**off-angle grasping went from 0/20 on the left to ~85% everywhere** after a fresh dataset.

**Two objects in frame, pick the one I asked for** (detector + masking):

| Setup | Correct object grasped |
| --- | :---: |
| Before (no masking) | 5 / 20 · 25% |
| After, "pick the pen" | **18 / 20 · 90%** |
| After, "pick the bear" | **16 / 20 · 80%** |

Spoken commands score the same as typed. Once Whisper turns speech into the instruction, the rest
of the pipeline is identical. Masking holds up even with several objects crowded close together.

**Search → grasp, across the workspace.** The first policy could only grasp where its demos were
(the right/front), so an object off to the left was found and centered every time but then the
policy reverted instead of grasping. Recording a fresh dataset of grasps **from the facing pose**
(200 demos across the full left-to-right range) fixed it:

| Side | First policy | After the facing-pose retrain |
| --- | :---: | :---: |
| Left | 0 / 20 · 0% | **~17 / 20 · 85%** |
| Right | 15-20 / 20 | **~17 / 20 · 85%** |

The new policy holds **80 to 90% across all four objects and the whole left-to-right range**, so the
left/right asymmetry is gone. The trick was matching training to how the arm actually runs: search
pans to face the object, so every demo was recorded starting from that same facing pose.

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

- The current model (v2) is fine-tuned on **200 demos across four objects** (bear, pen, tube, and a
  headphone case), recorded **from the facing pose**: the arm turned to face the object with it
  centered in the camera, then the grasp. Spread across the full left-to-right range so it grasps
  wherever the search leaves it.
- Fine-tuned SmolVLA for 20-30k steps on one A100, a few hours. Trained model is up on the Hugging
  Face Hub.
- Runs in real time on the laptop GPU.
- The earlier model (v1) was ~90 home-start demos across three objects. It worked, but only grasped
  in the region those demos covered, which is what the v2 facing-pose set fixed.

## What works and what doesn't

Works:
- It grabs the object you asked for, with a grip that suits it, from a typed **or spoken** command.
- **Across the whole workspace**, left to right, at 80 to 90%. Off-angle objects work now, not just
  the region the first model was trained on.
- **Two objects at once.** It picks the one you named and ignores the other (detector + masking),
  even with several crowded close together.
- **Off-screen objects.** It pans around to find the object, centers on it, then grasps.
- It runs by itself on the laptop.

Not yet / rough edges:
- **The pen on the far left** can hesitate for a moment before it commits (it is thin, so it centers
  a touch less precisely), then it grabs. Tightening the search centering smooths it.
- **Rubber grips.** The main problem is grabbing anything with a smooth or plastic surface. The
  gripper loses its hold, and the fix is rubber pads on the gripper's fingers.
- **One camera.** The wrist camera loses sight of the object right at the end of the reach, so the
  grab gets less precise near the edges of the workspace. A second camera would also help with
  dynamic situations, where the drop-off zone is moving and so is the object.
- **Stopping after one grasp** is still manual. I hit Ctrl-C when it's placed (which returns the arm
  home cleanly). There's an experimental `--auto-stop` that watches for the arm returning home, but
  it can mis-fire mid-grasp, so it's off by default.

## What's in this repo

```
README.md                   this file
requirements.txt            dependencies
deploy_select.py            grasp a named object: detect -> (search) -> (mask) -> grasp
voice_grasp.py              voice control (mic -> Whisper -> deploy_select)
perception/
  detector.py               open-vocabulary object detector (OWLv2)
  objects.py                the objects it knows (detect label <-> command <-> aliases)
  masking_robot.py          blur the other objects out of the policy's camera feed
  search.py                 pan the arm to find an off-screen object, center on it
  autostop.py               experimental: stop when the arm returns home
scripts/record.sh           record demonstrations
scripts/train_smolvla.sh    train the model on a GPU
scripts/deploy.sh           run the bare policy on the arm from a typed command
media/                      videos used above
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

# 3. Run it on the arm
./scripts/deploy.sh "Pick up the pen and drop it on the plate"   # bare policy, one command

# ...or the perception layer on top:
python deploy_select.py pen                 # detect + grasp the pen
python deploy_select.py pen --mask          # pick the pen out of a two-object scene
python deploy_select.py bear --search --mask # pan to find the bear, then grasp it
python voice_grasp.py                        # say "grab the pen"; Ctrl-C when it's placed
```

`deploy_select.py` needs `openai-whisper` only for voice; the detector pulls in `transformers`
(OWLv2). The detector runs on CPU on purpose so it doesn't fight the policy for the laptop GPU.

## A couple of decisions

- **Imitation learning instead of RL,** because copying a few demonstrations is way more practical
  on a real arm than millions of trial-and-error attempts that felt like they werent really going anywhere.
- **Fine-tuning a pretrained model instead of one model per object,** so adding a new object takes a
  lot less data and it allows the robot to be able to become a little more comfortable in new situations.
- **Getting one object at one spot rock solid first,** then position, then the command, instead of
  trying to do all of it at once, it just created for a smoother pipeline that I could then execute and understand.

## What's next

Done since the first version: voice, telling two objects apart, panning to find an off-screen
object, and grasping across the whole workspace (a fresh facing-pose dataset). Still ahead:

1. Keeping track of the object if I move it mid-reach.
2. Handing the object to my hand instead of dropping it on the mat. This one needs a second camera,
   since the wrist camera is blind to anything but the object once the gripper closes on it.
