"""Download b-mc2/sql-create-context and write the MLX training splits.

Writes to data/:
  train.jsonl    8,000 chat-format examples for mlx_lm.lora
  valid.jsonl      200 chat-format examples for validation loss
  test_raw.jsonl   200 held-out examples (question/context/answer) for evaluation

Splits are disjoint and fixed by seed, so every run of the pipeline sees
the same data.
"""
import json
from pathlib import Path

from datasets import load_dataset

from prompt_format import build_messages

SEED = 42
TEST_SIZE = 200
VALID_SIZE = 200
TRAIN_SIZE = 8000

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def to_chat(example):
    messages = build_messages(example["question"], example["context"])
    messages.append({"role": "assistant", "content": example["answer"]})
    return {"messages": messages}


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{path}  ({len(rows)} examples)")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    raw = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=SEED)

    test = raw.select(range(TEST_SIZE))
    valid = raw.select(range(TEST_SIZE, TEST_SIZE + VALID_SIZE))
    train = raw.select(range(TEST_SIZE + VALID_SIZE, TEST_SIZE + VALID_SIZE + TRAIN_SIZE))

    write_jsonl(DATA_DIR / "train.jsonl", [to_chat(ex) for ex in train])
    write_jsonl(DATA_DIR / "valid.jsonl", [to_chat(ex) for ex in valid])
    write_jsonl(
        DATA_DIR / "test_raw.jsonl",
        [
            {"question": ex["question"], "context": ex["context"], "answer": ex["answer"]}
            for ex in test
        ],
    )


if __name__ == "__main__":
    main()
