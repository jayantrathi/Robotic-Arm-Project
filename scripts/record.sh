#!/usr/bin/env bash
# Record teleoperated (leader-follower) demonstrations for ONE object/task.
#
# Usage:
#   ./scripts/record.sh "Pick up the pen and drop it on the plate" 30
#   ./scripts/record.sh "Pick up the bear and drop it on the plate" 30 --resume
#
# Notes:
#   - First object: no --resume (creates the dataset).
#   - Later objects: pass --resume to append to the same dataset. On resume,
#     num_episodes means "this many MORE", not a running total.
#   - Vary the object position every episode; keep the grasp clean and consistent.
set -e

TASK="${1:?Provide a task string, e.g. \"Pick up the pen and drop it on the plate\"}"
NUM="${2:-30}"
RESUME="${3:-}"    # pass --resume to append to an existing dataset

REPO=jayantrathi/lang_grasp_v2
ROOT="$HOME/.cache/huggingface/lerobot/$REPO"
FOLLOWER_PORT=/dev/tty.usbmodem5B790815221
LEADER_PORT=/dev/tty.usbmodem5B790800051

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=my_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=$LEADER_PORT \
  --teleop.id=my_leader_arm \
  --dataset.repo_id=$REPO \
  ${RESUME:+--dataset.root=$ROOT} \
  --dataset.num_episodes=$NUM \
  --dataset.single_task="$TASK" \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=12 \
  --dataset.no_stamp=true \
  --dataset.push_to_hub=false \
  --display_data=true \
  ${RESUME:+--resume=true}
