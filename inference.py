from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.prompt import format_inference_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a base model plus LoRA adapter.")
    parser.add_argument("--model-name", default="Qwen/Qwen2-7B", help="Base model name or local path.")
    parser.add_argument("--adapter-path", default="dpo_model", help="Adapter checkpoint directory.")
    parser.add_argument("--prompt", default="Explain LoRA", help="Prompt text for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling.")
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", help="Enable trust_remote_code when loading the model.")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false", help="Disable trust_remote_code when loading the model.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model with 4-bit quantization.")
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    device = next(model.parameters()).device
    prompt = format_inference_prompt(args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated)


if __name__ == "__main__":
    main()
