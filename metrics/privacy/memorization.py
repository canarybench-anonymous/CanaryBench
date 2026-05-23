from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from metrics.base import compute_sequence_loss, compute_token_losses, percentile, safe_mean


def validate_canary_splits(
    member_canaries: Sequence[Dict[str, Any]],
    reference_canaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    member_pairs = [(item.get("input", ""), item.get("target", "")) for item in member_canaries]
    reference_pairs = [(item.get("input", ""), item.get("target", "")) for item in reference_canaries]
    member_unique = set(member_pairs)
    reference_unique = set(reference_pairs)
    exact_overlap = member_unique & reference_unique

    return {
        "member_count": len(member_canaries),
        "reference_count": len(reference_canaries),
        "member_unique_count": len(member_unique),
        "reference_unique_count": len(reference_unique),
        "exact_overlap_count": len(exact_overlap),
        "has_exact_overlap": bool(exact_overlap),
        "member_has_category_metadata": any(item.get("category") is not None for item in member_canaries),
        "member_has_repetition_metadata": any(item.get("repetitions") is not None for item in member_canaries),
    }


def compute_canary_losses(
    model: Any,
    tokenizer: Any,
    canaries: Sequence[Dict[str, Any]],
    device: str,
    max_length: int = 512,
    mask_prompt: bool = True,
) -> List[Dict[str, Any]]:
    losses: List[Dict[str, Any]] = []
    for canary in canaries:
        loss = compute_sequence_loss(
            model=model,
            tokenizer=tokenizer,
            record=canary,
            device=device,
            max_length=max_length,
            mask_prompt=mask_prompt,
        )
        enriched = dict(canary)
        enriched["loss"] = float(loss)
        losses.append(enriched)
    return losses


def _exposure_for_loss(member_loss: float, reference_losses: Sequence[float]) -> float:
    if not reference_losses:
        return 0.0
    rank = sum(1 for ref_loss in reference_losses if ref_loss > member_loss) + 1
    return float(np.log2(len(reference_losses) / rank))


def compute_canary_exposure_from_losses(
    member_losses: Sequence[Dict[str, Any]],
    reference_losses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    member_values = [float(item["loss"]) for item in member_losses]
    reference_values = [float(item["loss"]) for item in reference_losses]
    exposures = [_exposure_for_loss(loss, reference_values) for loss in member_values]

    return {
        "exposure_mean": safe_mean(exposures),
        "exposure_p50": percentile(exposures, 50),
        "exposure_p95": percentile(exposures, 95),
        "exposure_max": float(np.max(exposures)) if exposures else None,
        "member_loss_mean": safe_mean(member_values),
        "reference_loss_mean": safe_mean(reference_values),
        "gap": (safe_mean(reference_values) - safe_mean(member_values))
        if member_values and reference_values
        else None,
        "num_members": len(member_values),
        "num_references": len(reference_values),
        "member_exposures": exposures,
    }


def compute_canary_exposure(
    model: Any,
    tokenizer: Any,
    member_canaries: Sequence[Dict[str, Any]],
    reference_canaries: Sequence[Dict[str, Any]],
    device: str = "cuda",
    max_length: int = 512,
    mask_prompt: bool = True,
) -> Dict[str, Any]:
    split_validation = validate_canary_splits(member_canaries, reference_canaries)
    member_losses = compute_canary_losses(
        model=model,
        tokenizer=tokenizer,
        canaries=member_canaries,
        device=device,
        max_length=max_length,
        mask_prompt=mask_prompt,
    )
    reference_losses = compute_canary_losses(
        model=model,
        tokenizer=tokenizer,
        canaries=reference_canaries,
        device=device,
        max_length=max_length,
        mask_prompt=mask_prompt,
    )
    summary = compute_canary_exposure_from_losses(member_losses, reference_losses)
    summary["member_losses"] = member_losses
    summary["reference_losses"] = reference_losses
    summary["split_validation"] = split_validation
    return summary


def exposure_by_category(
    member_losses_with_meta: Sequence[Dict[str, Any]],
    reference_losses: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    ref_values = [float(item["loss"]) for item in reference_losses]
    grouped: Dict[str, List[float]] = defaultdict(list)
    for item in member_losses_with_meta:
        category = item.get("category")
        if category is None:
            continue
        grouped[category].append(_exposure_for_loss(float(item["loss"]), ref_values))

    return {
        category: {
            "mean_exposure": safe_mean(values),
            "p95_exposure": percentile(values, 95),
            "count": len(values),
        }
        for category, values in grouped.items()
    }


def exposure_by_repetition(
    member_losses_with_meta: Sequence[Dict[str, Any]],
    reference_losses: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    ref_values = [float(item["loss"]) for item in reference_losses]
    grouped: Dict[int, List[float]] = defaultdict(list)
    for item in member_losses_with_meta:
        repetitions = item.get("repetitions")
        if repetitions is None:
            continue
        repetitions = int(repetitions)
        grouped[repetitions].append(_exposure_for_loss(float(item["loss"]), ref_values))

    return {
        f"reps_{repetitions}": float(np.mean(values))
        for repetitions, values in sorted(grouped.items())
    }


class CanaryExposureEvaluator:
    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda", max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def compute_canary_losses(self, canary_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        from metrics.base import load_jsonl

        records = load_jsonl(canary_path, max_samples=max_samples)
        return compute_canary_losses(
            model=self.model,
            tokenizer=self.tokenizer,
            canaries=records,
            device=self.device,
            max_length=self.max_length,
        )

    def compute_per_token_losses(
        self,
        canary_records: Sequence[Dict[str, Any]],
        max_records: int = 50,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for record in canary_records[:max_records]:
            if "input" in record and "target" in record:
                token_losses = compute_token_losses(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    record=record,
                    device=self.device,
                    max_length=self.max_length,
                )
            elif "token_ids" in record:
                ids = torch.tensor(record["token_ids"], device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(ids).logits[:, :-1, :]
                token_losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                    reduction="none",
                ).detach().cpu().numpy()
            else:
                token_losses = np.array([])
            results.append({"id": record.get("id", ""), "token_losses": token_losses.tolist()})
        return results

    def compute_exposure(
        self,
        member_losses: Sequence[Dict[str, Any]],
        reference_losses: Sequence[Dict[str, Any]],
    ) -> Optional[float]:
        summary = compute_canary_exposure_from_losses(member_losses, reference_losses)
        return summary.get("exposure_mean")
