import json
import os

import torch

from metrics.base import build_supervised_text, record_uses_chat_template


def evaluate_canaries_full_sequence(model, tokenizer, canary_path, device, max_samples=200):
    """
    Compute per-canary loss on the full prompt+target sequence.
    Only padding is masked; prompt tokens are included in the loss.
    """
    if not os.path.exists(canary_path):
        return [], 0.0

    was_training = model.training
    model.eval()
    results = []

    with open(canary_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = records[:max_samples]

    with torch.no_grad():
        for rec in records:
            _, text = build_supervised_text(tokenizer, rec)
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
                add_special_tokens=not record_uses_chat_template(tokenizer, rec),
            ).to(device)

            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100

            outputs = model(**enc, labels=labels)
            results.append({
                "id": rec.get("id", ""),
                "loss": round(outputs.loss.item(), 4),
                "canary": rec.get("canary", False),
                "source": rec.get("source", ""),
            })

    if was_training:
        model.train()

    avg_loss = sum(r["loss"] for r in results) / len(results) if results else 0.0
    return results, avg_loss
