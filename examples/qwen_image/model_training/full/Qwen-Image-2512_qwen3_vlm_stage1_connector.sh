#!/usr/bin/env bash
set -euo pipefail

# Stage 1: freeze DiT (and all other modules), disable LoRA, train only qwen3_connector.
# Output checkpoint will contain trainable params only (i.e., connector weights),
# because ModelLogger exports only requires_grad=True params.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} accelerate launch \
  --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml \
  --num_processes 1 \
  examples/qwen_image/model_training/train_qwen3_vlm.py \
  --sample_history_json sample_history.json \
  --max_pixels 1048576 \
  --dataset_repeat 2 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2512:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --trainable_models "qwen3_connector" \
  --output_path "./models/train/Qwen-Image-2512_qwen3_connector_stage1" \
  --remove_prefix_in_ckpt "pipe.qwen3_connector." \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters

# Expected connector checkpoint path (when --num_epochs 1):
# ./models/train/Qwen-Image-2512_qwen3_connector_stage1/epoch-0.safetensors
