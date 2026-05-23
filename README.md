# DP-LLM Privacy Benchmark

This repository contains code for building privacy-focused LLM benchmark datasets,
fine-tuning causal language models with and without differential privacy, and
evaluating privacy/utility tradeoffs with canary memorization and loss-based
audits.

The public benchmark surface is organized around:

- `instruction_recall_dataset/`: instruction-style utility, PII detection, and
  prefix-before-secret canary recall data.
- `enron_data/`: Enron language-modeling benchmark data with canaries injected
  into held-out email prefixes.
- `finetune/`: LoRA baseline and Opacus DP-LoRA training entrypoints.
- `evaluation/` and `metrics/`: utility, loss, memorization, inference, and
  extraction metrics.
- `benchmarks/`: benchmark runners and privacy matrix scripts.

Generated datasets, checkpoints, logs, and result dumps are intentionally
ignored by git. Release those as separate versioned artifacts, with checksums,
when the paper artifact requires exact reproduction.

## Install

Use Python 3.10+ in a clean environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For gated Hugging Face models, set a token in the shell instead of committing it:

```bash
export HF_TOKEN=hf_...
```

## Build Datasets

Instruction-recall benchmark:

```bash
python instruction_recall_dataset/build_instruction_dataset.py \
  --output-dir instruction_recall_dataset/dataset/processed \
  --seed 42
```

Expected generated JSONL counts:

- `combined_train.jsonl`: 83,000
- `combined_val.jsonl`: 3,000
- `combined_test.jsonl`: 3,000
- `member_canaries.jsonl`: 770
- `attack_eval.jsonl`: 1,000

Enron benchmark:

```bash
python enron_data/create_enron_dataset.py \
  --output-dir enron_data/processed \
  --seed 42
```

Expected generated JSONL counts:

- `combined_train.jsonl`: 251,237
- `combined_val.jsonl`: 1,000
- `combined_test.jsonl`: 1,000
- `member_canaries.jsonl`: 7,250
- `attack_eval.jsonl`: 1,000

## Train

Baseline LoRA:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune/train_baseline.py \
  --model vault_gemma
```

DP-LoRA:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune/train_dp.py \
  --model vault_gemma \
  --epsilon 8.0
```

For Enron-only runs, pass explicit Enron dataset paths:

```bash
DATA_DIR=enron_data/processed
CUDA_VISIBLE_DEVICES=0 python finetune/train_baseline.py \
  --model vault_gemma \
  --train-path "$DATA_DIR/combined_train.jsonl" \
  --val-path "$DATA_DIR/combined_val.jsonl" \
  --canary-path "$DATA_DIR/member_canaries.jsonl" \
  --reference-path "$DATA_DIR/attack_eval.jsonl" \
  --canary-registry-path "$DATA_DIR/canary_registry.json"

DATA_DIR=enron_data/processed
CUDA_VISIBLE_DEVICES=0 python finetune/train_dp.py \
  --model vault_gemma \
  --epsilon 8.0 \
  --train-path "$DATA_DIR/combined_train.jsonl" \
  --val-path "$DATA_DIR/combined_val.jsonl" \
  --canary-path "$DATA_DIR/member_canaries.jsonl" \
  --reference-path "$DATA_DIR/attack_eval.jsonl" \
  --canary-registry-path "$DATA_DIR/canary_registry.json"
```

For instruction-recall runs, pass explicit dataset paths:

```bash
DATA_DIR=instruction_recall_dataset/dataset/processed
CUDA_VISIBLE_DEVICES=0 python finetune/train_baseline.py \
  --model llama3_2_1b_it \
  --train-path "$DATA_DIR/combined_train.jsonl" \
  --val-path "$DATA_DIR/combined_val.jsonl" \
  --canary-path "$DATA_DIR/member_canaries.jsonl" \
  --reference-path "$DATA_DIR/attack_eval.jsonl" \
  --canary-registry-path "$DATA_DIR/canary_registry.json"
```

## Evaluate

```bash
python finetune/test_model.py \
  --model vault_gemma \
  --checkpoint checkpoints/vault_gemma/baseline/final \
  --dataset all
```

Enron privacy matrix:

```bash
python benchmarks/run_enron_privacy_matrix.py \
  --models vault_gemma \
  --data-dir enron_data/processed \
  --out-dir results_testing/enron_privacy_matrix
```
  artifact checksums for the paper artifact appendix.
