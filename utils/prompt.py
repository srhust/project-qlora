from __future__ import annotations

from typing import Dict


SFT_PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n{response}"
INFERENCE_PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def format_sft_sample(prompt: str, response: str) -> str:
    return SFT_PROMPT_TEMPLATE.format(prompt=prompt.strip(), response=response.strip())


def format_inference_prompt(prompt: str) -> str:
    return INFERENCE_PROMPT_TEMPLATE.format(prompt=prompt.strip())


def ensure_text_fields(example: Dict[str, str], required_fields: list[str]) -> Dict[str, str]:
    missing = [field for field in required_fields if field not in example]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")

    cleaned = {}
    for field in required_fields:
        value = example[field]
        if value is None:
            raise ValueError(f"Field '{field}' cannot be null.")
        cleaned[field] = str(value).strip()
    return cleaned
