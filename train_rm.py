from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import RewardTrainer

from utils.dataset import load_dpo_dataset
from utils.prompt import format_inference_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reward model from pairwise preference data.")
    parser.add_argument("--config", default="configs/rm.yaml", help="Path to reward-model YAML config.")
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


def preprocess_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _map_fn(example: dict) -> dict:
        chosen_text = format_inference_prompt(example["prompt"]) + example["chosen"]
        rejected_text = format_inference_prompt(example["prompt"]) + example["rejected"]

        chosen_tokens = tokenizer(
            chosen_text,
            truncation=True,
            max_length=max_length,
        )
        rejected_tokens = tokenizer(
            rejected_text,
            truncation=True,
            max_length=max_length,
        )

        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"],
        }

    return dataset.map(_map_fn, remove_columns=dataset.column_names, desc="Tokenizing RM dataset")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = build_quantization_config(config)
    torch_dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=1,
        trust_remote_code=config.get("trust_remote_code", True),
        quantization_config=quantization_config,
        torch_dtype=None if quantization_config else torch_dtype,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
        )
    elif config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    raw_dataset = load_dpo_dataset(config["dataset_path"])
    train_dataset = preprocess_dataset(raw_dataset, tokenizer, config["max_length"])

    training_cfg = config["training"]
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
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

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
