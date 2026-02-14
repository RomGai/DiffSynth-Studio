#!/usr/bin/env bash
set -euo pipefail

# Stage 1: 仅训练 connector（冻结 DiT，不启用 LoRA）
# 运行结束后，connector 权重会保存到：
# ./models/train/Qwen-Image-2512_qwen3_connector_stage1/epoch-0.safetensors

CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml \
  --num_processes 1 \
  examples/qwen_image/model_training/train_qwen3_vlm.py \
  --sample_history_json sample_history.json \
  --max_pixels 1048576 \
  --dataset_repeat 2 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2512:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 3e-5 \
  --num_epochs 1 \
  --trainable_models "qwen3_connector" \
  --remove_prefix_in_ckpt "pipe.qwen3_connector." \
  --output_path "./models/train/Qwen-Image-2512_qwen3_connector_stage1" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
