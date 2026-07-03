#!/usr/bin/env bash
# Fine-tune Qwen2.5-1.5B-Instruct (4-bit) on text-to-SQL with LoRA via MLX,
# logging everything to reports/train.log for the loss chart.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
mlx_lm.lora --config configs/lora_qwen2.5-1.5b.yaml "$@" 2>&1 | tee reports/train.log
