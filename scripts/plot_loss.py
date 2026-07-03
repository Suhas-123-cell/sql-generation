"""Chart train/validation loss from the mlx_lm.lora log (reports/train.log)."""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "train.log"
    text = log_path.read_text()

    train = re.findall(r"Iter (\d+): Train loss ([\d.]+)", text)
    val = re.findall(r"Iter (\d+): Val loss ([\d.]+)", text)

    plt.figure(figsize=(8, 4))
    if train:
        plt.plot([int(i) for i, _ in train], [float(l) for _, l in train], label="train")
    if val:
        plt.plot([int(i) for i, _ in val], [float(l) for _, l in val], "o-", label="validation")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("LoRA fine-tuning loss (Qwen2.5-1.5B, text-to-SQL)")
    plt.legend()
    plt.grid(alpha=0.3)

    out = log_path.parent / "loss_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
