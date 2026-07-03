# Project Report — Text-to-SQL via LoRA Fine-Tuning

## 1. Problem Statement

Natural-language interfaces to databases let non-technical users query data without
knowing SQL. Small open-source LLMs (1–3B parameters) are cheap to run but unreliable at
SQL generation out of the box: they add explanations when only a query is wanted,
hallucinate table and column names, and produce syntactically invalid SQL.

**Goal:** fine-tune a small open-source LLM so that, given a database schema and a
question, it reliably produces the correct SQL query — and quantify the improvement
over the base model with objective metrics.

## 2. Dataset and Preprocessing

**Dataset:** [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
(~78,000 examples, CC-BY-4.0, built from WikiSQL and Spider). Each example contains:

- `context` — one or more `CREATE TABLE` statements (the schema)
- `question` — a natural-language question
- `answer` — the gold SQL query

**Preprocessing** (`scripts/prepare_data.py`):

1. Shuffle with a fixed seed (42) for reproducibility.
2. Split into disjoint sets: 200 test / 200 validation / 8,000 train. The test set is
   never seen during training.
3. Convert each training example into the model's chat format — a system instruction
   ("reply with only the SQL query"), a user turn holding schema + question, and an
   assistant turn holding the gold SQL — written as JSONL for `mlx_lm.lora`.

One deliberate decision: the prompt template lives in a single shared module
(`scripts/prompt_format.py`) imported by both data preparation and inference, because a
template mismatch between training and inference silently degrades results and is one of
the most common fine-tuning bugs.

## 3. Solution Approach

QLoRA-style parameter-efficient fine-tuning, run **entirely locally on a MacBook Pro
(Apple M5, 16 GB unified memory)** using Apple's MLX framework:

- The base model is quantized to 4-bit and stays **frozen**.
- Small trainable **LoRA adapter** matrices (rank 16) are injected into the attention
  projection layers of the top 16 transformer blocks.
- Only the adapters train — well under 2% of total parameters — which is what makes a
  1.5B-model fine-tune possible on a laptop.

The standard CUDA QLoRA stack (bitsandbytes/Unsloth) does not run on Apple Silicon;
MLX provides the equivalent workflow natively on the M-series GPU.

## 4. Model Selection

**Qwen2.5-1.5B-Instruct** (4-bit MLX build). Reasons:

- Strong code/SQL ability for its size — Qwen models are trained with a large code share.
- 1.5B in 4-bit uses ~2–3 GB of memory, leaving headroom for optimizer state and
  activations within 16 GB.
- The *instruct* variant already follows chat formatting, so fine-tuning only has to
  teach the task (SQL generation), not instruction-following from scratch.
- An openly licensed base (Apache-2.0) keeps the whole pipeline reproducible.

## 5. Training / Fine-Tuning Process

Configuration in `configs/lora_qwen2.5-1.5b.yaml`, run via `scripts/train.sh`:

| hyperparameter | value | rationale |
|---|---|---|
| LoRA rank / scale | 16 / 16 | enough capacity for a narrow task; keeps adapter small |
| Layers adapted | top 16 of 28 | quality/memory trade-off on 16 GB |
| Batch size | 4 | fits comfortably in unified memory at seq len 1024 |
| Iterations | 1,200 (~4,800 examples) | text-to-SQL converges fast; validation loss confirms |
| Learning rate | 1e-4 | standard LoRA range; stable in this run |
| Gradient checkpointing | on | trades compute for memory |
| Seed | 42 everywhere | reproducibility |

Validation loss is computed every 200 iterations on the held-out validation split;
the full log is kept in `reports/train.log` and charted in `reports/loss_curve.png`.

- Training time: **~15 minutes** (1,200 iterations at ~1.3 it/sec on the M5)
- Peak memory: **2.7 GB** of the 16 GB unified memory
- Final train / validation loss: **0.544 / 0.586** (validation started at 3.51; train and
  validation curves track each other closely, so no overfitting)

## 6. Evaluation Results

Both models decode the same 200 held-out test questions greedily (deterministic,
temperature 0), so the comparison is apples-to-apples (`scripts/generate_predictions.py`,
`scripts/evaluate.py`).

| metric | base model | fine-tuned |
|---|---|---|
| Exact match | 53.5% | **75.5%** |
| Execution validity | 95.0% | **97.0%** |

(Gold queries execute OK on 99.0% of schemas — the sanity upper bound for execution
validity.)

- **Exact match:** normalized string equality (lowercased, whitespace collapsed) with the
  gold SQL. Strict — a semantically correct query written differently counts as a miss —
  so it is a *lower bound* on true accuracy.
- **Execution validity:** the schema is built in an in-memory SQLite database and the
  predicted query must execute without error. This catches hallucinated tables, wrong
  column names, and syntax errors. Gold queries are scored the same way as a sanity
  upper bound.

**52** of 200 test examples that the base model got wrong are answered exactly
correctly after fine-tuning. The typical failure modes it fixes: the base model
invents unnecessary joins, quotes numeric-string columns incorrectly, and deviates
from the dataset's SQL conventions; the fine-tuned model matches them precisely. Representative before/after examples are printed by
`python scripts/evaluate.py --examples 5`.

## 7. Challenges Faced

1. **No CUDA on Apple Silicon.** The standard QLoRA tooling (bitsandbytes, Unsloth) is
   NVIDIA-only. Solved by switching to MLX, which implements the same
   frozen-4-bit-base + LoRA-adapter recipe natively on the M-series GPU.
2. **16 GB memory budget.** Handled with a 4-bit base model, adapters on only the top 16
   layers, batch size 4, and gradient checkpointing.
3. **Evaluating generated SQL fairly.** Exact match alone undercounts (equivalent queries
   differ in formatting); execution-*result* comparison needs populated databases, which
   this dataset does not ship. Settled on the pair of exact match (strict lower bound) +
   execution validity (schema-level correctness), reported side by side and honestly
   labeled.
4. **Train/inference prompt consistency.** Solved structurally: one shared
   `prompt_format.py` module used by both the data pipeline and the prediction script.

## 8. Possible Improvements

- **Execution-result accuracy** on a benchmark with populated databases (Spider dev set)
  — the industry-standard text-to-SQL metric.
- **Larger base model** (Qwen2.5-7B-4bit also fits in 16 GB, trains slower) for harder
  multi-table joins.
- **Full-epoch or multi-epoch training** with early stopping on validation loss.
- **Schema linking / RAG:** retrieve only the relevant tables into the prompt when the
  real database schema is too large for the context window.
- **Adapter fusion + quantized export** (`mlx_lm.fuse`) for a single deployable artifact,
  or serving behind a small API.
