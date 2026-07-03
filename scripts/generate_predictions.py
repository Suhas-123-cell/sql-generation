"""Generate SQL for the held-out test set with the base and fine-tuned models.

Runs the same greedy decoding twice — once with the plain 4-bit base model,
once with the LoRA adapter loaded — and writes both predictions side by side
to predictions.json for scoring by evaluate.py.
"""
import argparse
import json
import re
from pathlib import Path

from mlx_lm import generate, load

from prompt_format import build_messages

ROOT = Path(__file__).resolve().parents[1]


def extract_sql(text):
    """Strip markdown fences and keep the first SQL statement."""
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    return text.strip().split(";")[0].strip()


def predict(model_name, adapter, examples, label):
    model, tokenizer = load(model_name, adapter_path=adapter)
    preds = []
    for i, ex in enumerate(examples):
        prompt = tokenizer.apply_chat_template(
            build_messages(ex["question"], ex["context"]), add_generation_prompt=True
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=120)
        preds.append(extract_sql(out))
        if (i + 1) % 20 == 0:
            print(f"  {label}: {i + 1}/{len(examples)}")
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter", default=str(ROOT / "adapters"))
    parser.add_argument("--test", default=str(ROOT / "data" / "test_raw.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "predictions.json"))
    parser.add_argument("--limit", type=int, help="only run the first N examples")
    args = parser.parse_args()

    with open(args.test) as f:
        examples = [json.loads(line) for line in f]
    if args.limit:
        examples = examples[: args.limit]

    print(f"Base model on {len(examples)} examples...")
    baseline = predict(args.model, None, examples, "base")
    print(f"Fine-tuned model on {len(examples)} examples...")
    finetuned = predict(args.model, args.adapter, examples, "fine-tuned")

    records = [
        {
            "question": ex["question"],
            "context": ex["context"],
            "gold": ex["answer"],
            "baseline": baseline[i],
            "finetuned": finetuned[i],
        }
        for i, ex in enumerate(examples)
    ]
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
