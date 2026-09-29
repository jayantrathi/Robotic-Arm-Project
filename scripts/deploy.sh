#!/usr/bin/env bash
# Run the fine-tuned policy on the arm for a single typed command.
# (For voice control, use ../voice_grasp.py instead.)
#
# Usage:
#   ./scripts/deploy.sh "Pick up the pen and drop it on the plate"
#
# The camera is named camera1 to match how the policy was trained.
set -e

TASK="${1:?Provide a task, e.g. \"Pick up the pen and drop it on the plate\"}"

lerobot-rollout \
  --strategy.type=base \
  --policy.path=jayantrathi/smolvla_lang_grasp_v1 \
  --policy.device=mps \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B790815221 \
  --robot.id=my_follower_arm \
  --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --task="$TASK" \
  --inference.type=rtc \
  --duration=25 \
  --display_data=true
