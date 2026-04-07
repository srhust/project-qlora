from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.prompt import format_inference_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate batch predictions from prompts in a JSONL file.")
    parser.add_argument("--input-file", default="data/sft_data.jsonl", help="JSONL with at least a prompt field.")
    parser.add_argument("--output-file", default="predictions.jsonl", help="Destination JSONL for model predictions.")
    parser.add_argument("--model-name", default="Qwen/Qwen2-7B", help="Base model name or local path.")
    parser.add_argument("--adapter-path", default="dpo_model", help="Adapter checkpoint directory.")
    parser.add_argument("--prompt-field", default="prompt", help="Field name used as model input.")
    parser.add_argument("--reference-field", default="response", help="Optional field copied into reference.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max new tokens per sample.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p used when sampling is enabled.")
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", help="Enable trust_remote_code when loading the model.")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false", help="Disable trust_remote_code when loading the model.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model with 4-bit quantization.")
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=None if quantization_config else torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    return tokenizer, model


def generate_one(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    device = next(model.parameters()).device
    formatted_prompt = format_inference_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(formatted_prompt):].strip() if decoded.startswith(formatted_prompt) else decoded.strip()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    tokenizer, model = load_model(args)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            if args.prompt_field not in row:
                raise KeyError(f"Missing prompt field '{args.prompt_field}' in row: {row}")

            prompt = str(row[args.prompt_field]).strip()
            output = generate_one(model, tokenizer, prompt, args)
            result = {
                "prompt": prompt,
                "output": output,
            }

            if args.reference_field in row:
                result["reference"] = str(row[args.reference_field]).strip()

            dst.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
