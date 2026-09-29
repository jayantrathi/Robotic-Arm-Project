#!/usr/bin/env bash
# Fine-tune SmolVLA on the recorded dataset.
# Run on a CUDA GPU (e.g. a rented A100). Setup on a fresh box:
#   pip install "lerobot[smolvla,dataset]==0.6.1"
#   hf auth login
#
# The dataset's camera is named "front"; smolvla_base expects "camera1", so we
# remap it. Missing camera2/camera3 slots are padded automatically.
set -e

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=jayantrathi/lang_grasp_v1 \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --output_dir=outputs/train/smolvla_lang_grasp_v1 \
  --job_name=smolvla_lang_grasp_v1 \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=5000 \
  --policy.repo_id=jayantrathi/smolvla_lang_grasp_v1 \
  --policy.push_to_hub=true \
  --wandb.enable=false
