import torch
from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline
from qwen3_pipeline import Qwen3VLPipelineConfig

# ========= 1) 路径配置 =========
stage2_ckpt = "./models/train/Qwen-Image-2512_qwen3_lora_connector_stage2/epoch-0.safetensors"

# 你训练时对应的 base 模型
model_id_with_origin_paths = [
    "Qwen/Qwen-Image-2512:transformer/diffusion_pytorch_model*.safetensors",
    "Qwen/Qwen-Image:text_encoder/model*.safetensors",
    "Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors",
]

# ========= 2) 构建完整 pipeline =========
model_configs = []
for item in model_id_with_origin_paths:
    split_id = item.rfind(":")
    model_id = item[:split_id]
    origin_file_pattern = item[split_id + 1:]
    model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern))

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",  # 或 "cpu"
    model_configs=model_configs,
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

# 关键：先启用 qwen3 conditioning，确保 qwen3_connector 被创建出来
pipe.enable_qwen3_conditioning(config=Qwen3VLPipelineConfig())

# ========= 3) 读取 Stage2 混合 ckpt 并拆分 =========
# Stage2 你用了 --remove_prefix_in_ckpt "pipe.dit."
# 因此 LoRA key 通常是 "xxx.lora_A/B..."（没有 pipe.dit. 前缀）
# connector key 通常仍是 "pipe.qwen3_connector.xxx"
sd = load_state_dict(stage2_ckpt, device="cpu")

lora_sd = {}
connector_sd = {}

for k, v in sd.items():
    # 拆 connector
    if k.startswith("pipe.qwen3_connector."):
        connector_sd[k[len("pipe.qwen3_connector."):]] = v
    elif k.startswith("qwen3_connector."):
        connector_sd[k[len("qwen3_connector."):]] = v
    # 其余按 LoRA 处理（通常都是 DiT LoRA 项）
    else:
        lora_sd[k] = v

print(f"[INFO] stage2 ckpt keys = {len(sd)}")
print(f"[INFO] lora keys = {len(lora_sd)}")
print(f"[INFO] connector keys = {len(connector_sd)}")

# ========= 4) 加载 LoRA + connector =========
# 4.1 DiT LoRA
if len(lora_sd) > 0:
    # load_lora 支持直接传 state_dict
    pipe.load_lora(pipe.dit, state_dict=lora_sd, verbose=1)

# 4.2 connector
if len(connector_sd) > 0:
    missing, unexpected = pipe.qwen3_connector.load_state_dict(connector_sd, strict=False)
    print(f"[INFO] connector missing keys: {len(missing)}")
    print(f"[INFO] connector unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("missing:", missing[:20])
    if len(unexpected) > 0:
        print("unexpected:", unexpected[:20])

print("[DONE] pipeline loaded with your Stage2-trained LoRA + connector.")
