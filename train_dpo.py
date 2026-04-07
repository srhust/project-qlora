from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

from utils.dataset import load_dpo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO adapter starting from an SFT LoRA checkpoint.")
    parser.add_argument("--config", default="configs/dpo.yaml", help="Path to DPO YAML config.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not config.get("load_in_4bit", True):
        return None

    compute_dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = build_quantization_config(config)
    torch_dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=config.get("trust_remote_code", True),
        quantization_config=quantization_config,
        torch_dtype=None if quantization_config else torch_dtype,
        device_map="auto",
    )
    base_model.config.use_cache = False
    if quantization_config is not None:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
        )
    elif config.get("gradient_checkpointing", True):
        base_model.gradient_checkpointing_enable()

    model = PeftModel.from_pretrained(base_model, config["sft_adapter_path"], is_trainable=True)

    train_dataset = load_dpo_dataset(config["dataset_path"])
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        num_train_epochs=training_cfg["num_train_epochs"],
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        evaluation_strategy=training_cfg.get("eval_strategy", "no"),
        bf16=config.get("use_bf16", True),
        fp16=not config.get("use_bf16", True),
        report_to=training_cfg.get("report_to", "none"),
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        seed=training_cfg.get("seed", 42),
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=config.get("beta", 0.1),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        max_prompt_length=config["max_prompt_length"],
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
