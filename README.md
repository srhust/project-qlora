# QLoRA + DPO + RM / GRPO Project

This repository contains an engineer-friendly training pipeline for instruction tuning and alignment on top of a causal language model such as `Qwen/Qwen2-7B`.

The project supports:

- QLoRA-based supervised fine-tuning (SFT)
- DPO alignment with pairwise preference data
- Batch inference and prediction export
- Lightweight evaluation on generated outputs
- Optional reward model (RM) training
- Optional GRPO policy optimization bootstrap

## Project Structure

```text
.
├── configs/
│   ├── dpo.yaml
│   ├── grpo.yaml
│   ├── rm.yaml
│   └── sft.yaml
├── data/
│   ├── dpo_data.jsonl
│   ├── grpo_data.jsonl
│   ├── rm_data.jsonl
│   └── sft_data.jsonl
├── utils/
│   ├── dataset.py
│   └── prompt.py
├── evaluate.py
├── export_predictions.py
├── inference.py
├── train_dpo.py
├── train_grpo.py
├── train_rm.py
└── train_sft.py
```

## Environment

Use Python 3.10 and install the main dependencies:

```bash
conda create -n lora_rl python=3.10 -y
conda activate lora_rl
pip install torch transformers accelerate peft datasets trl bitsandbytes pyyaml
```

For multi-GPU training:

```bash
accelerate config
```

Recommended baseline:

- `multi-GPU`
- `DeepSpeed: NO`

## Data Formats

### SFT

`data/sft_data.jsonl`

```json
{"prompt": "Explain what is LoRA", "response": "LoRA is a parameter-efficient fine-tuning method..."}
```

### DPO

`data/dpo_data.jsonl`

```json
{"prompt": "Explain LoRA", "chosen": "LoRA reduces training cost...", "rejected": "LoRA is just a trick..."}
```

### RM

Reward-model training reuses the same pairwise preference format as DPO:

```json
{"prompt": "Why use preference data?", "chosen": "Preference data helps ranking...", "rejected": "Preference data is useless."}
```

### GRPO

`data/grpo_data.jsonl`

```json
{"prompt": "Explain the difference between LoRA and QLoRA."}
```

## Training

### 1. SFT with QLoRA

```bash
accelerate launch train_sft.py --config configs/sft.yaml
```

Key details:

- `trust_remote_code=True` is enabled for Qwen-style models
- LoRA target modules default to `q_proj` and `v_proj`
- 4-bit quantization is enabled through `BitsAndBytesConfig`

### 2. DPO Alignment

```bash
accelerate launch train_dpo.py --config configs/dpo.yaml
```

This stage loads the base model plus the SFT adapter from `sft_model/` and continues optimization with preference pairs.

### 3. Reward Model

```bash
accelerate launch train_rm.py --config configs/rm.yaml
```

This script trains a scalar reward head from chosen/rejected pairs using `RewardTrainer`.

### 4. GRPO

```bash
accelerate launch train_grpo.py --config configs/grpo.yaml
```

This bootstrap version uses prompt-only data and heuristic reward functions. In a real production setup, you would usually replace those heuristics with a learned reward model or task-specific verification logic.

## Inference

Single-sample generation:

```bash
python inference.py --adapter-path dpo_model --prompt "Explain LoRA"
```

Batch export to JSONL:

```bash
python export_predictions.py \
  --input-file data/sft_data.jsonl \
  --output-file predictions.jsonl \
  --adapter-path dpo_model \
  --reference-field response
```

The output file contains:

```json
{"prompt": "Explain what is LoRA", "output": "...", "reference": "..."}
```

## Evaluation

```bash
python evaluate.py --pred-file predictions.jsonl
```

The current evaluator reports:

- sample count
- average output length
- exact match if a `reference` field exists

## Notes

- On 8 x A100, start with the provided batch size and gradient accumulation settings, then scale after confirming stability.
- DPO and RM quality are much more sensitive to preference data quality than small hyperparameter changes.
- GRPO support in `trl` evolves quickly. If your installed `trl` version is older and misses `GRPOTrainer`, upgrade `trl` first.
