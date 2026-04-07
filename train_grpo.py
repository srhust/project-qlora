from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from utils.dataset import load_jsonl_train_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a policy with GRPO using prompt-only data and heuristic rewards.")
    parser.add_argument("--config", default="configs/grpo.yaml", help="Path to GRPO YAML config.")
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


def non_empty_reward(completions, **kwargs):
    return [1.0 if completion.strip() else 0.0 for completion in completions]


def keyword_reward(prompts, completions, **kwargs):
    scores = []
    for prompt, completion in zip(prompts, completions):
        prompt_terms = [token.lower() for token in str(prompt).split() if len(token) > 3]
        completion_lower = completion.lower()
        overlap = sum(1 for term in prompt_terms if term in completion_lower)
        scores.append(min(overlap / max(len(prompt_terms), 1), 1.0))
    return scores


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
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        trust_remote_code=config.get("trust_remote_code", True),
        quantization_config=quantization_config,
        torch_dtype=None if quantization_config else torch_dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    raw_dataset = load_jsonl_train_split(config["dataset_path"])
    train_dataset = raw_dataset.map(
        lambda row: {"prompt": str(row["prompt"]).strip()},
        desc="Preparing GRPO prompts",
    )

    training_cfg = config["training"]
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    grpo_args = GRPOConfig(
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
        bf16=config.get("use_bf16", True),
        fp16=not config.get("use_bf16", True),
        report_to=training_cfg.get("report_to", "none"),
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],
        num_generations=config["num_generations"],
    )

    peft_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=[non_empty_reward, keyword_reward],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
