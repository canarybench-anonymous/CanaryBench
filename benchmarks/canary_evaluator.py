"""
Canary evaluation utilities for DP training.

This module provides functions to evaluate canary losses during training
and compute various metrics for privacy analysis.
"""

import os
import json
import torch
from typing import List, Dict, Optional, Tuple, Any

from metrics.base import (
    build_supervised_text,
    encode_prompt_and_target,
    record_uses_chat_template,
)


def evaluate_canaries(model: torch.nn.Module, tokenizer: Any, 
                     canary_path: str, device: str, max_samples: int = 200) -> Tuple[List[Dict], float]:
    """
    Compute per-canary loss at end of each epoch.
    Critical for MIA + exposure — record every epoch, not just the end.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        canary_path: Path to canary file
        device: Device to run on
        max_samples: Maximum number of canaries to evaluate
        
    Returns:
        Tuple of (canary_results, average_loss)
    """
    if not os.path.exists(canary_path):
        return [], 0.0

    model.eval()
    results = []

    with open(canary_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    records = records[:max_samples]

    with torch.no_grad():
        for rec in records:
            enc, prompt_len = encode_prompt_and_target(
                tokenizer=tokenizer,
                record=rec,
                max_length=512,
                device=device,
            )
            labels = enc["input_ids"].clone()
            labels[:, :prompt_len] = -100  # mask prompt
            if "attention_mask" in enc:
                labels[enc["attention_mask"] == 0] = -100

            outputs = model(**enc, labels=labels)
            results.append({
                "id":     rec.get("id", ""),
                "loss":   round(outputs.loss.item(), 4),
                "canary": rec.get("canary", False),
                "source": rec.get("source", ""),
            })

    model.train()
    avg = sum(r["loss"] for r in results) / len(results) if results else 0.0
    return results, avg


def _load_registry_map(registry_path: str) -> Dict[str, Dict[str, Any]]:
    if not registry_path or not os.path.exists(registry_path):
        return {}

    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    return {record["id"]: record for record in registry if "id" in record}


def _canonical_canary_id(canary_id: str) -> str:
    for suffix in ("_rep0", "_rep1", "_rep2", "_rep3", "_rep4", "_rep5", "_rep6", "_rep7", "_rep8", "_rep9"):
        if canary_id.endswith(suffix):
            return canary_id[:-len(suffix)]
    return canary_id


def evaluate_canaries_on_secret(
    model: torch.nn.Module,
    tokenizer: Any,
    canary_path: str,
    registry_path: str,
    device: str,
    max_samples: int = 500,
) -> Tuple[List[Dict], float, float]:
    """
    Measure loss on the secret-token span for each canary.

    Returns:
        Tuple of (per_canary_results, avg_secret_loss, avg_target_loss)
    """
    if not os.path.exists(canary_path):
        return [], 0.0, 0.0

    registry_map = _load_registry_map(registry_path)
    if not registry_map:
        return [], 0.0, 0.0

    records = load_canary_records(canary_path, max_samples=max_samples)
    if not records:
        return [], 0.0, 0.0

    was_training = model.training
    model.eval()
    results: List[Dict[str, Any]] = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for rec in records:
            rec_id = rec.get("id", "")
            base_id = _canonical_canary_id(rec_id)
            registry_entry = registry_map.get(base_id) or registry_map.get(rec_id)
            if not registry_entry:
                continue

            secret = registry_entry.get("secret", "")
            if not secret:
                continue

            enc, prompt_len = encode_prompt_and_target(
                tokenizer=tokenizer,
                record=rec,
                max_length=512,
                device=device,
            )
            labels = enc["input_ids"].clone()
            labels[:, :prompt_len] = -100
            if "attention_mask" in enc:
                labels[enc["attention_mask"] == 0] = -100

            outputs = model(**enc, labels=labels)

            secret_token_ids = tokenizer(secret, add_special_tokens=False)["input_ids"]
            if not secret_token_ids:
                continue

            valid_input_ids = enc["input_ids"][0][enc["attention_mask"][0].bool()].tolist()
            secret_start = -1
            last_possible_start = len(valid_input_ids) - len(secret_token_ids) + 1
            for pos in range(max(last_possible_start, 0)):
                if valid_input_ids[pos:pos + len(secret_token_ids)] == secret_token_ids:
                    secret_start = pos
                    break

            if secret_start == -1:
                continue

            logits = outputs.logits[:, :-1, :].contiguous()
            next_tokens = enc["input_ids"][:, 1:].contiguous()
            per_token_loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                next_tokens.view(-1),
            ).view(1, -1)[0]

            loss_start = max(secret_start - 1, 0)
            loss_end = min(loss_start + len(secret_token_ids), per_token_loss.numel())
            if loss_end <= loss_start:
                continue

            secret_loss = per_token_loss[loss_start:loss_end].mean().item()
            results.append({
                "id": rec_id,
                "loss": round(outputs.loss.item(), 4),
                "secret_loss": round(secret_loss, 4),
                "canary": rec.get("canary", False),
                "source": rec.get("source", ""),
                "secret_len": len(secret_token_ids),
                "secret_slot": registry_entry.get("secret_slot", ""),
                "exposure": registry_entry.get("exposure", rec.get("exposure")),
            })

    if was_training:
        model.train()

    avg_secret_loss = sum(r["secret_loss"] for r in results) / len(results) if results else 0.0
    avg_target_loss = sum(r["loss"] for r in results) / len(results) if results else 0.0
    return results, avg_secret_loss, avg_target_loss


def compute_canary_exposure(member_losses: List[Dict], reference_losses: List[Dict]) -> Optional[float]:
    """
    Compute canary exposure metric.
    
    Simple leakage metric: fraction of member losses below median of reference.
    
    Args:
        member_losses: List of member canary losses
        reference_losses: List of reference canary losses
        
    Returns:
        Exposure score (fraction of member losses below median reference loss)
    """
    ref_losses = [x["loss"] for x in reference_losses]
    if not ref_losses:
        return None
    
    import torch
    med = float(torch.tensor(ref_losses).median().item())
    hit = sum(1 for x in member_losses if x["loss"] < med)
    return float(hit) / len(member_losses) if member_losses else None


def compute_loss_statistics(member_losses: List[Dict], reference_losses: List[Dict]) -> Dict[str, float]:
    """
    Compute comprehensive loss statistics for canary analysis.
    
    Args:
        member_losses: List of member canary losses
        reference_losses: List of reference canary losses
        
    Returns:
        Dictionary of loss statistics
    """
    member_vals = [x["loss"] for x in member_losses]
    reference_vals = [x["loss"] for x in reference_losses]
    
    stats = {}
    
    if member_vals:
        stats["member_mean"] = sum(member_vals) / len(member_vals)
        stats["member_median"] = sorted(member_vals)[len(member_vals) // 2]
        stats["member_min"] = min(member_vals)
        stats["member_max"] = max(member_vals)
    
    if reference_vals:
        stats["reference_mean"] = sum(reference_vals) / len(reference_vals)
        stats["reference_median"] = sorted(reference_vals)[len(reference_vals) // 2]
        stats["reference_min"] = min(reference_vals)
        stats["reference_max"] = max(reference_vals)
    
    if member_vals and reference_vals:
        stats["gap_mean"] = stats["reference_mean"] - stats["member_mean"]
        stats["gap_median"] = stats["reference_median"] - stats["member_median"]
        stats["exposure"] = compute_canary_exposure(member_losses, reference_losses)
    
    return stats


def save_canary_evaluation_results(output_path: str, results: Dict[str, Any]) -> str:
    """
    Save canary evaluation results to JSON file.
    
    Args:
        output_path: Path to save results
        results: Dictionary of evaluation results
        
    Returns:
        Path where results were saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return output_path


def load_canary_records(canary_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load canary records from file.
    
    Args:
        canary_path: Path to canary file
        max_samples: Maximum number of records to load
        
    Returns:
        List of canary records
    """
    if not os.path.exists(canary_path):
        return []
    
    with open(canary_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    if max_samples:
        records = records[:max_samples]
    
    return records


def encode_canary_for_evaluation(canary_record: Dict, tokenizer: Any) -> Dict:
    """
    Encode a single canary record for evaluation.
    
    Args:
        canary_record: Canary record with input and target
        tokenizer: The tokenizer
        
    Returns:
        Encoded canary with token_ids
    """
    _, full = build_supervised_text(tokenizer, canary_record)
    enc = tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
        add_special_tokens=not record_uses_chat_template(tokenizer, canary_record),
    )
    
    return {
        "id": canary_record.get("id", ""),
        "token_ids": enc["input_ids"][0].tolist(),
        "category": canary_record.get("category", "unk"),
        "repetitions": canary_record.get("repetitions", 1)
    }
