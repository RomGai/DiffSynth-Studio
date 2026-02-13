import argparse
import json
import os
import copy
from pathlib import Path

import accelerate
import torch
from PIL import Image

from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion import *
from diffsynth.diffusion.parsers import (
    add_gradient_config,
    add_image_size_config,
    add_lora_config,
    add_model_config,
    add_output_config,
    add_training_config,
)
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline
from qwen3_pipeline import Qwen3VLPipelineConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SampleHistoryDataset(torch.utils.data.Dataset):
    """Training dataset where JSON is used for VLM conditioning and PNG is supervision target."""

    def __init__(
        self,
        sample_history_json: str,
        repeat: int = 1,
        max_pixels: int = 1024 * 1024,
        height: int | None = None,
        width: int | None = None,
        prompt_mode: str = "description",
        target_image_path: str | None = None,
    ):
        self.sample_history_json = str(sample_history_json)
        self.repeat = repeat
        self.load_from_cache = False
        self.prompt_mode = prompt_mode

        history_path = Path(sample_history_json)
        with history_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list) or len(records) == 0:
            raise ValueError("sample_history.json must be a non-empty list.")

        supervision_image_path = Path(target_image_path) if target_image_path is not None else history_path.with_suffix(".png")
        if not supervision_image_path.is_absolute():
            supervision_image_path = history_path.parent / supervision_image_path
        if not supervision_image_path.exists():
            raise FileNotFoundError(
                f"Supervision image not found: {supervision_image_path}. "
                "Please provide --target_image_path or place a same-name .png next to sample_history.json."
            )

        image_processor = ImageCropAndResize(height, width, max_pixels, 16, 16)
        image = Image.open(supervision_image_path).convert("RGB")
        image = image_processor(image)

        prompt = ""
        if prompt_mode == "description":
            prompt = "\n".join([str(record.get("description", "")) for record in records]).strip()

        self.items = [{
            "image": image,
            "prompt": prompt,
            "qwen3_history_json": self.sample_history_json,
        }]

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, idx):
        return self.items[idx % len(self.items)]


class Qwen3VLMTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
        qwen3_model_name_or_path=None,
        qwen3_max_length=640,
        qwen3_attn_implementation=None,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
        )
        qwen3_config = Qwen3VLPipelineConfig(
            model_name_or_path=qwen3_model_name_or_path or Qwen3VLPipelineConfig.model_name_or_path,
            max_length=qwen3_max_length,
            attn_implementation=qwen3_attn_implementation,
        )
        self.pipe.enable_qwen3_conditioning(config=qwen3_config)
        if trainable_models is None:
            trainable_models = "dit,qwen3_connector"
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def set_trainable_models(self, trainable_models: str | None, keep_lora_for_model: str | None = None):
        model_names = [] if trainable_models is None or trainable_models == "" else trainable_models.split(",")
        self.pipe.freeze_except(model_names)
        if keep_lora_for_model is not None and hasattr(self.pipe, keep_lora_for_model):
            model = getattr(self.pipe, keep_lora_for_model)
            if model is not None:
                model.train()
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data.get("prompt", "")}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": self.zero_cond_t,
            "qwen3_history_json": data.get("qwen3_history_json"),
        }
        image = data["image"]
        inputs_shared.update({
            "input_image": image,
            "height": image.size[1],
            "width": image.size[0],
        })
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def qwen3_vlm_parser():
    parser = argparse.ArgumentParser(
        description="Training script for Qwen3-VL conditioning with Qwen-Image DiT using sample_history.json as the primary data source."
    )
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_image_size_config(parser)

    parser.add_argument("--sample_history_json", type=str, required=True, help="Path to sample_history.json used for VLM conditioning.")
    parser.add_argument("--target_image_path", type=str, default=None, help="Supervision image path for diffusion training. If omitted, use <sample_history_json_basename>.png.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="How many times to repeat records from sample_history.json per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--history_prompt_mode", type=str, default="description", choices=["description", "empty"], help="How to form the diffusion prompt from descriptions in sample_history.json.")

    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--qwen3_model_name_or_path", type=str, default=None, help="Qwen3-VL model id or path.")
    parser.add_argument("--qwen3_max_length", type=int, default=640, help="Max length of Qwen3-VL embeddings.")
    parser.add_argument("--qwen3_attn_implementation", type=str, default=None, help="Optional attention implementation for Qwen3-VL.")

    parser.add_argument("--enable_two_phase", default=False, action="store_true", help="Enable automatic two-phase training schedule.")
    parser.add_argument("--phase1_num_epochs", type=int, default=0, help="Epochs for phase-1 training. If <=0, phase-1 is skipped.")
    parser.add_argument("--phase1_trainable_models", type=str, default="qwen3_connector", help="Trainable models in phase-1.")
    parser.add_argument("--phase1_learning_rate", type=float, default=None, help="Optional phase-1 learning rate override.")
    parser.add_argument("--phase2_num_epochs", type=int, default=None, help="Epochs for phase-2 training. If None, uses --num_epochs.")
    parser.add_argument("--phase2_trainable_models", type=str, default="dit,qwen3_connector", help="Trainable models in phase-2.")
    parser.add_argument("--phase2_learning_rate", type=float, default=None, help="Optional phase-2 learning rate override.")
    return parser


def run_two_phase_training(accelerator, dataset, model, args):
    if args.phase1_num_epochs > 0:
        phase1_args = copy.deepcopy(args)
        phase1_args.num_epochs = args.phase1_num_epochs
        phase1_args.trainable_models = args.phase1_trainable_models
        if args.phase1_learning_rate is not None:
            phase1_args.learning_rate = args.phase1_learning_rate
        model.set_trainable_models(phase1_args.trainable_models)
        phase1_logger = ModelLogger(
            os.path.join(args.output_path, "phase1"),
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        )
        print(f"[TwoPhase] Phase-1: epochs={phase1_args.num_epochs}, trainable={phase1_args.trainable_models}, lr={phase1_args.learning_rate}")
        launch_training_task(accelerator, dataset, model, phase1_logger, args=phase1_args)

    phase2_args = copy.deepcopy(args)
    phase2_args.num_epochs = args.num_epochs if args.phase2_num_epochs is None else args.phase2_num_epochs
    phase2_args.trainable_models = args.phase2_trainable_models
    if args.phase2_learning_rate is not None:
        phase2_args.learning_rate = args.phase2_learning_rate

    phase2_model_names = [] if phase2_args.trainable_models is None or phase2_args.trainable_models == "" else phase2_args.trainable_models.split(",")
    keep_lora_for_model = None
    if phase2_args.lora_base_model is not None and phase2_args.lora_base_model in phase2_model_names:
        phase2_model_names = [name for name in phase2_model_names if name != phase2_args.lora_base_model]
        keep_lora_for_model = phase2_args.lora_base_model
        phase2_args.trainable_models = ",".join(phase2_model_names)
        print(f"[TwoPhase] Detected LoRA base model '{keep_lora_for_model}'. Phase-2 will train LoRA params instead of full-parameter '{keep_lora_for_model}'.")

    model.set_trainable_models(phase2_args.trainable_models, keep_lora_for_model=keep_lora_for_model)
    phase2_logger = ModelLogger(
        os.path.join(args.output_path, "phase2" if args.phase1_num_epochs > 0 else ""),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    print(f"[TwoPhase] Phase-2: epochs={phase2_args.num_epochs}, trainable={phase2_args.trainable_models}, lr={phase2_args.learning_rate}")
    launch_training_task(accelerator, dataset, model, phase2_logger, args=phase2_args)


if __name__ == "__main__":
    parser = qwen3_vlm_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    dataset = SampleHistoryDataset(
        sample_history_json=args.sample_history_json,
        repeat=args.dataset_repeat,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        prompt_mode=args.history_prompt_mode,
        target_image_path=args.target_image_path,
    )

    model = Qwen3VLMTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        zero_cond_t=args.zero_cond_t,
        qwen3_model_name_or_path=args.qwen3_model_name_or_path,
        qwen3_max_length=args.qwen3_max_length,
        qwen3_attn_implementation=args.qwen3_attn_implementation,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    if args.enable_two_phase and args.task in ("sft", "sft:train"):
        run_two_phase_training(accelerator, dataset, model, args)
    else:
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
