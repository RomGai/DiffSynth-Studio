#!/usr/bin/env bash
set -euo pipefail

# Stage 2: jointly train DiT LoRA + qwen3_connector.
# It loads connector weights trained in stage 1 before training starts,
# and saves both LoRA + connector trainable params at the end.

CONNECTOR_CKPT=${CONNECTOR_CKPT:-./models/train/Qwen-Image-2512_qwen3_connector_stage1/epoch-0.safetensors}

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
  --trainable_models "dit,qwen3_connector" \
  --connector_checkpoint "${CONNECTOR_CKPT}" \
  --remove_prefix_in_ckpt "" \
  --output_path "./models/train/Qwen-Image-2512_qwen3_lora_connector_stage2" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters

# Example:
# CONNECTOR_CKPT=./models/train/Qwen-Image-2512_qwen3_connector_stage1/epoch-0.safetensors \
#   bash examples/qwen_image/model_training/full/Qwen-Image-2512_qwen3_vlm_stage2_lora_connector.sh
