from __future__ import annotations

from pathlib import Path
from typing import Callable

from datasets import Dataset, load_dataset

from utils.prompt import ensure_text_fields, format_sft_sample


def _resolve_dataset_path(dataset_path: str) -> str:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return str(path)


def load_jsonl_train_split(dataset_path: str) -> Dataset:
    resolved = _resolve_dataset_path(dataset_path)
    dataset_dict = load_dataset("json", data_files=resolved)
    return dataset_dict["train"]


def load_sft_dataset(dataset_path: str) -> Dataset:
    dataset = load_jsonl_train_split(dataset_path)

    def _map_fn(example: dict) -> dict:
        row = ensure_text_fields(example, ["prompt", "response"])
        return {
            "prompt": row["prompt"],
            "response": row["response"],
            "text": format_sft_sample(row["prompt"], row["response"]),
        }

    return dataset.map(_map_fn, desc="Formatting SFT dataset")


def load_dpo_dataset(dataset_path: str) -> Dataset:
    dataset = load_jsonl_train_split(dataset_path)

    def _map_fn(example: dict) -> dict:
        row = ensure_text_fields(example, ["prompt", "chosen", "rejected"])
        return row

    return dataset.map(_map_fn, desc="Validating DPO dataset")


def apply_prompt_formatter(dataset: Dataset, formatter: Callable[[dict], dict], desc: str) -> Dataset:
    return dataset.map(formatter, desc=desc)
