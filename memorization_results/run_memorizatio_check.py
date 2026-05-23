import csv
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------
# MODELS AND CHECKPOINTS
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
canary_path = "/home/noora/dp-llm/instruction_recall_dataset/dataset/processed/member_canaries.jsonl"

device = "cuda"
max_length = 256
max_new_tokens = 30
top_k_canaries = 20


models_to_run = [
    {
        "model_key": "gemma_2b_it_20",
        "base_path": "google/gemma-2-2b-it",
        "adapter_path": "/home/noora/dp-llm/checkpoints/gemma_2b_it/baseline/final",
        "prompt_format": "auto",
    },
    {
        "model_key": "vault_gemma_20",
        "base_path": "google/vaultgemma-1b",
        "adapter_path": "/home/noora/dp-llm/checkpoints/vault_gemma/baseline/final",
        "prompt_format": "auto",
    },
    {
        "model_key": "gemma3_1b_20",
        "base_path": "google/gemma-3-1b-it",
        "adapter_path": "/home/noora/dp-llm/checkpoints/gemma3_1b/baseline/final",
        "prompt_format": "chat",
    },
    {
        "model_key": "llama3_2_1b_it_20",
        "base_path": "meta-llama/Llama-3.2-1B-Instruct",
        "adapter_path": "/home/noora/dp-llm/checkpoints/llama3_2_1b_it/baseline/final",
        "prompt_format": "chat",
    },
]


# ---------------------------------------------------------
# LOAD CANARIES
# ---------------------------------------------------------
with open(canary_path, encoding="utf-8") as f:
    members = [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------
# PROMPT HELPERS
# ---------------------------------------------------------
def resolve_prompt_format(tokenizer, prompt_format):
    if prompt_format == "auto":
        if getattr(tokenizer, "chat_template", None):
            return "chat"
        return "response"

    if prompt_format == "chat" and not getattr(tokenizer, "chat_template", None):
        return "response"

    return prompt_format


def make_prompt(input_text, prompt_format):
    if prompt_format == "response":
        return input_text + "\n\n### Response:\n", True

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        tokenize=False,
        add_generation_prompt=True,
    ), False


def make_full_text(input_text, target_text, prompt_format):
    if prompt_format == "response":
        prompt, add_special_tokens = make_prompt(input_text, prompt_format)
        return prompt + target_text, add_special_tokens

    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": target_text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    ), False


def compute_target_loss(canary, prompt_format):
    prompt, add_special_tokens = make_prompt(canary["input"], prompt_format)
    full_text, _ = make_full_text(canary["input"], canary["target"], prompt_format)

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )["input_ids"]

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    ).to(device)

    labels = enc["input_ids"].clone()
    labels[:, : prompt_ids.shape[1]] = -100

    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    if int((labels != -100).sum().item()) == 0:
        return float("inf")

    with torch.no_grad():
        return model(**enc, labels=labels).loss.item()


def generate_from_canary(canary, prompt_format):
    prompt, add_special_tokens = make_prompt(canary["input"], prompt_format)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        out[0][enc["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    return generated


# ---------------------------------------------------------
# EVALUATE ALL BASELINE MODELS
# ---------------------------------------------------------
combined_summary_rows = []

for model_info in models_to_run:
    model_key = model_info["model_key"]
    base_path = model_info["base_path"]
    adapter_path = model_info["adapter_path"]
    save_path = os.path.join(script_dir, model_key, "baseline_final_20.json")
    summary_path = os.path.join(script_dir, model_key, "baseline_final_20_summary.json")

    print("\n==================================================")
    print(f"MODEL: {model_key}")
    print(f"BASE:  {base_path}")
    print(f"ADAPT: {adapter_path}")
    print("==================================================")

    # ---------------------------------------------------------
    # LOAD TOKENIZER
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        use_fast=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_format = resolve_prompt_format(tokenizer, model_info["prompt_format"])
    print(f"Prompt format: {prompt_format}")

    # ---------------------------------------------------------
    # LOAD MODEL + PEFT ADAPTER
    # ---------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()

    all_results = {}
    summary_rows = []

    # ---------------------------------------------------------
    # EVALUATE BY EXPOSURE TIER
    # ---------------------------------------------------------
    for tier in [50, 10, 1]:
        tier_canaries = [m for m in members if m.get("exposure") == tier]
        if not tier_canaries:
            continue

        print(f"\n=== {tier}x CANARIES: generating from top {top_k_canaries} ===\n")

        results = []
        for c in tier_canaries:
            loss = compute_target_loss(c, prompt_format)
            results.append((loss, c))

        results.sort(key=lambda x: x[0])

        tier_results = []

        # ---------------------------------------------------------
        # GENERATE FROM MOST MEMORIZED
        # ---------------------------------------------------------
        for loss, c in results[:top_k_canaries]:
            generated = generate_from_canary(c, prompt_format)
            target = c["target"].strip()

            row = {
                "loss": loss,
                "input": c["input"],
                "target": c["target"],
                "generated": generated,
                "match": target.lower() in generated.lower(),
            }
            tier_results.append(row)

            print(f"Loss: {loss:.4f}")
            print(f"Input:     {c['input'][-100:]}")
            print(f"Target:    {c['target']}")
            print(f"Generated: {generated}")
            print(f"Match:     {row['match']}")
            print()

        extracted_count = sum(1 for row in tier_results if row["match"])
        evaluated_count = len(tier_results)
        all_results[str(tier)] = tier_results

        summary_row = {
            "model": model_key,
            "tier": tier,
            "evaluated": evaluated_count,
            "canaries_extracted": extracted_count,
            "extraction_rate": extracted_count / evaluated_count if evaluated_count else 0.0,
        }
        summary_rows.append(summary_row)
        combined_summary_rows.append(summary_row)

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nCanaries extracted summary")
    print("Model | Tier | Evaluated | Canaries extracted")
    print("------|------|-----------|-------------------")
    for row in summary_rows:
        print(
            f"{row['model']} | {row['tier']}x | "
            f"{row['evaluated']} | {row['canaries_extracted']}"
        )

    print(f"Saved: {save_path}")
    print(f"Saved summary: {summary_path}")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------
# SAVE COMBINED EXTRACTION TABLE
# ---------------------------------------------------------
combined_json_path = os.path.join(script_dir, "baseline_final_20_extraction_table.json")
combined_csv_path = os.path.join(script_dir, "baseline_final_20_extraction_table.csv")

with open(combined_json_path, "w", encoding="utf-8") as f:
    json.dump(combined_summary_rows, f, indent=2)

with open(combined_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "tier",
            "evaluated",
            "canaries_extracted",
            "extraction_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(combined_summary_rows)

print("\nFINAL 20-CANARY EXTRACTION TABLE")
print("Model | Tier | Evaluated | Extracted | Rate")
print("------|------|-----------|-----------|------")
for row in combined_summary_rows:
    print(
        f"{row['model']} | {row['tier']}x | {row['evaluated']} | "
        f"{row['canaries_extracted']} | {row['extraction_rate']:.3f}"
    )

print(f"Saved combined table: {combined_csv_path}")
print(f"Saved combined JSON: {combined_json_path}")
