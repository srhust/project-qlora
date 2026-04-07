from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JSONL predictions with simple quality statistics.")
    parser.add_argument("--pred-file", default="predictions.jsonl", help="Path to predictions JSONL file.")
    return parser.parse_args()


def evaluate(pred_file: str) -> None:
    path = Path(pred_file)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    total = 0
    total_words = 0
    exact_match = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            output = str(row.get("output", "")).strip()
            reference = str(row.get("reference", "")).strip()

            total += 1
            total_words += len(output.split())
            if reference and output == reference:
                exact_match += 1

    if total == 0:
        raise ValueError("Prediction file is empty.")

    print(f"Samples: {total}")
    print(f"Avg Length: {total_words / total:.2f}")
    if exact_match:
        print(f"Exact Match: {exact_match / total:.2%}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.pred_file)
