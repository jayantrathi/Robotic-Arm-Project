# Recording plan: fresh dataset (lang_grasp_v2)

A new dataset built on one pattern that matches how the arm actually runs: the arm
faces the object with it centered in the camera, then reaches and grabs. At deploy,
search pans to center the object and hands the policy an arm already facing it, so
every demo should start from that same facing pose.

Old dataset (`lang_grasp_v1`, 155 demos) stays untouched. This records to
`lang_grasp_v2`.

## The pattern for every episode

1. **Setup (not recorded):** place the object somewhere, then pan the arm to face it
   so it is centered in the camera. Do this during the reset gap, before the episode
   recording starts.
2. **Recorded motion:** reach out, grab, bring it to the plate, drop. Then settle back
   toward center for the next setup.

The one rule that matters: **the find-pan is NOT in the recorded part.** Search does
that at deploy, so the demo must start already facing the object. If you record the
pan-to-find, the policy learns to pan first and then gets confused at deploy when it
is already facing the object.

## How many, and where

50 per object, 4 objects, 200 total. Spread the object's position across the whole
range, balanced left and right (search centers both sides the same way). Within each
band, change the exact spot, distance, and orientation every time, do not repeat the
same setup.

| Where the object is (you pan to face it) | Demos per object |
| ---------------------------------------- | :--------------: |
| Far left                                 |        10        |
| Left                                     |        10        |
| Center / straight ahead                  |        10        |
| Right                                    |        10        |
| Far right                                |        10        |

50 different setups generalize far better than a few setups repeated, so vary every one.

## Grip rules

- Consistent grip **point**: always grab the same PART of the object, let the pose
  follow the object's orientation.
  - Pen: always across the barrel.
  - Bear: always the same spot on the body (e.g. the belly).
  - Lip balm: always the tube; let it lie in different orientations.
  - Headphone case: always across its wide middle, with a squeeze like the bear.
- Only keep clean, successful demos. Fumble? Discard and redo it.

## Commands

The FIRST object creates the new dataset (no `--resume`). The rest append with
`--resume`. Recording needs the **leader arm** connected.

```bash
# first object creates lang_grasp_v2 (no --resume)
./scripts/record.sh "Pick up the bear and drop it on the plate" 50

# the rest append
./scripts/record.sh "Pick up the pen and drop it on the plate" 50 --resume
./scripts/record.sh "Pick up the lip balm and drop it on the plate" 50 --resume
./scripts/record.sh "Pick up the headphone case and drop it on the plate" 50 --resume
```

On `--resume`, the number means this many more episodes, so you can split any object
across sittings by running it again with a smaller count.

## After recording

1. Push `lang_grasp_v2` to the Hub.
2. Train with `scripts/train_smolvla.sh` (already pointed at v2, 20k steps). It fine-tunes
   from `smolvla_base` and pushes `smolvla_lang_grasp_v2`.
3. Point deploy at the new model (`deploy.sh` and `deploy_select.py` POLICY_PATH) and
   test with `--search` (no pan-shift, it grasps natively from the facing pose now).
4. Re-measure the success rates with the same 20-trial protocol for a clean before/after.

## One code change before deploy

The headphone case is already in `perception/objects.py` with a first-guess detect label
of `pouch`. Test `pouch` / `case` / `wallet` on a live frame and keep whatever scores best
before trusting masking and search for it.
