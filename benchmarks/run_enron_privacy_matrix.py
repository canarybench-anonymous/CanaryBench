"""
Run Enron privacy benchmarks over the Enron LoRA checkpoints.

This is intentionally separate from the instruction-recall profile matrix. Enron
canaries are LM records: the benchmark measures loss on the injected secret span
inside each canary's full text, not an instruction-style response.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.base import load_causal_lm, load_jsonl, save_json  # noqa: E402
from metrics.privacy.memorization import (  # noqa: E402
    compute_canary_exposure_from_losses,
    validate_canary_splits,
)
from metrics.utility.lm import compute_perplexity  # noqa: E402


DEFAULT_WBC_WINDOW_LENGTHS = [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_base: str
    adapter_path: Path
    run: str = "baseline_enron"
    epsilon: Optional[float] = None


DEFAULT_MODELS: Dict[str, ModelSpec] = {
    "gemma3_1b": ModelSpec(
        key="gemma3_1b",
        hf_base="google/gemma-3-1b-it",
        adapter_path=Path("checkpoints/gemma3_1b/baseline_enron/final"),
    ),
    "vault_gemma": ModelSpec(
        key="vault_gemma",
        hf_base="google/vaultgemma-1b",
        adapter_path=Path("checkpoints/vault_gemma/baseline_enron/final"),
    ),
    "gemma_2b_it": ModelSpec(
        key="gemma_2b_it",
        hf_base="google/gemma-2-2b-it",
        # The current checkpoint folder is named "baselin_enron" in this repo.
        adapter_path=Path("checkpoints/gemma_2b_it/baselin_enron/final"),
    ),
    "llama3_2_1b_it": ModelSpec(
        key="llama3_2_1b_it",
        hf_base="meta-llama/Llama-3.2-1B-Instruct",
        adapter_path=Path("checkpoints/enron/llama3_2_1b_it/baseline/final"),
    ),
}


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return {record["id"]: record for record in records if isinstance(record, dict) and "id" in record}


def _secret_for_record(record: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> str:
    entry = registry.get(str(record.get("id", "")), {})
    return str(entry.get("secret") or record.get("target") or "")


def _full_text_for_record(record: Dict[str, Any], secret: str) -> str:
    full_text = str(record.get("full_text") or "")
    if full_text:
        return full_text
    prefix = str(record.get("prefix") or record.get("input") or "")
    target = str(record.get("target") or secret)
    return f"{prefix}{target}"


def _secret_token_losses(
    model: Any,
    tokenizer: Any,
    record: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
    device: str,
    max_length: int,
) -> Optional[Dict[str, Any]]:
    secret = _secret_for_record(record, registry)
    full_text = _full_text_for_record(record, secret)
    if not secret or secret not in full_text:
        return None

    secret_start = full_text.find(secret)
    secret_end = secret_start + len(secret)

    try:
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    except TypeError:
        encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
        offsets = None
    else:
        offsets = encoded.pop("offset_mapping")[0].tolist()

    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits[:, :-1, :].float()

    targets = encoded["input_ids"][:, 1:]
    per_token_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)[0]

    if offsets is not None:
        secret_token_indices = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if end > secret_start and start < secret_end and end > start
        ]
        loss_indices = [idx - 1 for idx in secret_token_indices if idx > 0]
    else:
        secret_ids = tokenizer(secret, add_special_tokens=False)["input_ids"]
        input_ids = encoded["input_ids"][0].detach().cpu().tolist()
        loss_indices = []
        for idx in range(1, len(input_ids) - len(secret_ids) + 1):
            if input_ids[idx : idx + len(secret_ids)] == secret_ids:
                loss_indices = list(range(idx - 1, idx - 1 + len(secret_ids)))
                break

    loss_indices = [idx for idx in loss_indices if 0 <= idx < int(per_token_loss.numel())]
    if not loss_indices:
        return None

    token_losses = per_token_loss[loss_indices].detach().cpu().numpy().astype(float).tolist()
    return {
        "id": record.get("id", ""),
        "loss": float(np.mean(token_losses)),
        "token_losses": token_losses,
        "secret_len": len(loss_indices),
        "source": record.get("source", ""),
        "exposure": record.get("exposure"),
        "canary": record.get("canary", False),
    }


def _compute_secret_losses(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    registry: Dict[str, Dict[str, Any]],
    device: str,
    max_length: int,
) -> list[Dict[str, Any]]:
    losses: list[Dict[str, Any]] = []
    for record in records:
        result = _secret_token_losses(model, tokenizer, record, registry, device=device, max_length=max_length)
        if result is not None:
            losses.append(result)
    return losses


def _strip_token_losses(losses: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [{key: value for key, value in item.items() if key != "token_losses"} for item in losses]


def _window_vote_score(target_losses: Sequence[float], reference_losses: Sequence[float], window_size: int) -> float:
    min_length = min(len(target_losses), len(reference_losses))
    if min_length == 0:
        return 0.0
    effective_window = min(window_size, min_length)
    kernel = np.ones(effective_window)
    target_sums = np.convolve(np.asarray(target_losses[:min_length]), kernel, mode="valid")
    reference_sums = np.convolve(np.asarray(reference_losses[:min_length]), kernel, mode="valid")
    return float(np.mean(reference_sums > target_sums))


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    for index, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += index
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _roc_curve(labels: Sequence[int], scores: Sequence[float]) -> list[Dict[str, float]]:
    positives = sum(labels)
    negatives = len(labels) - positives
    paired = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    curve = [{"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0}]
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        curve.append(
            {
                "threshold": float(score),
                "fpr": fp / negatives if negatives else 0.0,
                "tpr": tp / positives if positives else 0.0,
            }
        )
    return curve


def _tpr_at_fpr(curve: Sequence[Dict[str, float]], target_fpr: float) -> float:
    eligible = [point["tpr"] for point in curve if point["fpr"] <= target_fpr]
    return float(max(eligible)) if eligible else 0.0


def _bootstrap_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    n_bootstrap_samples: int,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    aucs: list[float] = []
    for _ in range(n_bootstrap_samples):
        indices = rng.integers(0, len(labels_arr), size=len(labels_arr))
        sample_labels = labels_arr[indices]
        if len(set(sample_labels.tolist())) < 2:
            continue
        aucs.append(_roc_auc(sample_labels.tolist(), scores_arr[indices].tolist()))
    if not aucs:
        return {"auc_ci_lower": 0.5, "auc_ci_upper": 0.5}
    return {
        "auc_ci_lower": float(np.percentile(aucs, 2.5)),
        "auc_ci_upper": float(np.percentile(aucs, 97.5)),
    }


def _run_wbc_from_losses(
    member_target: Sequence[Dict[str, Any]],
    reference_target: Sequence[Dict[str, Any]],
    member_base: Sequence[Dict[str, Any]],
    reference_base: Sequence[Dict[str, Any]],
    window_lengths: Sequence[int],
    n_bootstrap_samples: int,
) -> Dict[str, Any]:
    base_by_id = {str(item["id"]): item for item in [*member_base, *reference_base]}
    member_scores: list[float] = []
    reference_scores: list[float] = []
    member_loss_scores: list[float] = []
    reference_loss_scores: list[float] = []

    for item in member_target:
        base = base_by_id.get(str(item["id"]))
        if not base:
            continue
        scores = [
            _window_vote_score(item["token_losses"], base["token_losses"], window)
            for window in window_lengths
        ]
        member_scores.append(float(np.mean(scores)) if scores else 0.0)
        member_loss_scores.append(-float(item["loss"]))

    for item in reference_target:
        base = base_by_id.get(str(item["id"]))
        if not base:
            continue
        scores = [
            _window_vote_score(item["token_losses"], base["token_losses"], window)
            for window in window_lengths
        ]
        reference_scores.append(float(np.mean(scores)) if scores else 0.0)
        reference_loss_scores.append(-float(item["loss"]))

    labels = [1] * len(member_scores) + [0] * len(reference_scores)
    scores = member_scores + reference_scores
    curve = _roc_curve(labels, scores)
    auc = _roc_auc(labels, scores)
    loss_auc = _roc_auc(labels, member_loss_scores + reference_loss_scores)

    result = {
        "wbc_auc": round(auc, 4),
        "tpr_at_fpr001": round(_tpr_at_fpr(curve, 0.01), 4),
        "window_lengths": list(window_lengths),
        "window_aggregation": "mean",
        "comparison_rule": "fraction_of_windows_where_base_model_secret_loss_exceeds_target_model_secret_loss",
        "uses_reference_model": True,
        "num_members": len(member_scores),
        "num_references": len(reference_scores),
        "member_scores": member_scores,
        "reference_scores": reference_scores,
        "loss_baseline_auc": round(loss_auc, 4),
        "random_baseline_auc": 0.5,
    }
    result.update(
        {
            key: round(value, 4)
            for key, value in _bootstrap_auc_ci(labels, scores, n_bootstrap_samples).items()
        }
    )
    return result


def _secret_recall(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    registry: Dict[str, Dict[str, Any]],
    device: str,
    max_samples: int,
    max_input_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    details: list[Dict[str, Any]] = []
    for record in records[:max_samples]:
        secret = _secret_for_record(record, registry)
        full_text = _full_text_for_record(record, secret)
        if not secret or secret not in full_text:
            continue
        prompt = full_text.split(secret, 1)[0]
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        details.append(
            {
                "id": record.get("id", ""),
                "secret": secret,
                "completion": completion[:200],
                "exact_hit": bool(secret in completion),
                "exposure": record.get("exposure"),
            }
        )

    exact_rate = float(np.mean([item["exact_hit"] for item in details])) if details else 0.0
    return {
        "exact_secret_recall_rate": round(exact_rate, 4),
        "num_attempts": len(details),
        "details": details,
    }


def _exposure_tier_summary(member_losses: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, list[float]] = {}
    for item in member_losses:
        tier = str(item.get("exposure", "unknown"))
        grouped.setdefault(tier, []).append(float(item["loss"]))
    return {
        tier: {"count": len(values), "secret_loss_mean": round(float(np.mean(values)), 4)}
        for tier, values in sorted(grouped.items())
    }


def _evaluate_one(
    spec: ModelSpec,
    args: argparse.Namespace,
    member_records: Sequence[Dict[str, Any]],
    reference_records: Sequence[Dict[str, Any]],
    registry: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    adapter_path = _repo_path(spec.adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Missing adapter directory for {spec.key}: {adapter_path}")

    print(f"[load] {spec.key}: {adapter_path}")
    loaded = load_causal_lm(spec.hf_base, adapter_path=str(adapter_path), device=args.device)
    reference_loaded = load_causal_lm(spec.hf_base, adapter_path=None, device=args.device)

    print(f"[privacy] {spec.key}: secret-span losses")
    member_target = _compute_secret_losses(
        loaded.model,
        loaded.tokenizer,
        member_records,
        registry,
        device=args.device,
        max_length=args.max_length,
    )
    reference_target = _compute_secret_losses(
        loaded.model,
        loaded.tokenizer,
        reference_records,
        registry,
        device=args.device,
        max_length=args.max_length,
    )
    exposure = compute_canary_exposure_from_losses(member_target, reference_target)
    exposure["member_losses"] = _strip_token_losses(member_target)
    exposure["reference_losses"] = _strip_token_losses(reference_target)
    exposure["split_validation"] = validate_canary_splits(member_records, reference_records)
    exposure["exposure_tier_summary"] = _exposure_tier_summary(member_target)

    print(f"[privacy] {spec.key}: WBC attack")
    member_base = _compute_secret_losses(
        reference_loaded.model,
        reference_loaded.tokenizer,
        member_records,
        registry,
        device=args.device,
        max_length=args.max_length,
    )
    reference_base = _compute_secret_losses(
        reference_loaded.model,
        reference_loaded.tokenizer,
        reference_records,
        registry,
        device=args.device,
        max_length=args.max_length,
    )
    wbc = _run_wbc_from_losses(
        member_target=member_target,
        reference_target=reference_target,
        member_base=member_base,
        reference_base=reference_base,
        window_lengths=args.wbc_window_lengths,
        n_bootstrap_samples=args.bootstrap_samples,
    )

    print(f"[privacy] {spec.key}: secret recall")
    secret_recall = {
        "member": _secret_recall(
            loaded.model,
            loaded.tokenizer,
            member_records,
            registry,
            device=args.device,
            max_samples=args.max_recall_samples,
            max_input_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        ),
        "reference": _secret_recall(
            loaded.model,
            loaded.tokenizer,
            reference_records,
            registry,
            device=args.device,
            max_samples=args.max_recall_samples,
            max_input_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        ),
    }

    print(f"[utility] {spec.key}: Enron val perplexity")
    perplexity = compute_perplexity(
        loaded.model,
        loaded.tokenizer,
        str(_repo_path(args.val_path)),
        device=args.device,
        max_samples=args.max_val_samples,
    )

    return {
        "benchmark": "enron_privacy_matrix",
        "model_key": spec.key,
        "run": spec.run,
        "epsilon": spec.epsilon,
        "base_model_path": spec.hf_base,
        "adapter_path": str(adapter_path),
        "data": {
            "member_path": str(_repo_path(args.member_path)),
            "reference_path": str(_repo_path(args.reference_path)),
            "val_path": str(_repo_path(args.val_path)),
            "registry_path": str(_repo_path(args.registry_path)),
        },
        "settings": {
            "max_member_samples": args.max_member_samples,
            "max_reference_samples": args.max_reference_samples,
            "max_recall_samples": args.max_recall_samples,
            "max_val_samples": args.max_val_samples,
            "max_length": args.max_length,
            "wbc_window_lengths": args.wbc_window_lengths,
        },
        "canary_exposure": exposure,
        "wbc": wbc,
        "secret_recall": secret_recall,
        "perplexity": perplexity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Enron privacy benchmarks for Enron LoRA checkpoints.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="results_testing/enron_privacy_matrix")
    parser.add_argument("--models", default="gemma3_1b,vault_gemma,gemma_2b_it,llama3_2_1b_it")
    parser.add_argument("--member-path", default="enron_data/processed/member_canaries.jsonl")
    parser.add_argument("--reference-path", default="enron_data/processed/attack_eval.jsonl")
    parser.add_argument("--val-path", default="enron_data/processed/combined_val.jsonl")
    parser.add_argument("--registry-path", default="enron_data/processed/canary_registry.json")
    parser.add_argument("--max-member-samples", type=int, default=770)
    parser.add_argument("--max-reference-samples", type=int, default=1000)
    parser.add_argument("--max-recall-samples", type=int, default=100)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--wbc-window-lengths", type=int, nargs="+", default=DEFAULT_WBC_WINDOW_LENGTHS)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    requested = _parse_csv(args.models)
    unknown = [key for key in requested if key not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Known keys: {sorted(DEFAULT_MODELS)}")

    member_records = load_jsonl(str(_repo_path(args.member_path)), max_samples=args.max_member_samples)
    reference_records = load_jsonl(str(_repo_path(args.reference_path)), max_samples=args.max_reference_samples)
    registry = _load_registry(_repo_path(args.registry_path))

    out_dir = _repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Any] = {
        "benchmark": "enron_privacy_matrix",
        "out_dir": str(out_dir),
        "runs": [],
    }

    for key in requested:
        spec = DEFAULT_MODELS[key]
        out_path = out_dir / f"{spec.key}__{spec.run}.json"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {spec.key}: existing {out_path}")
            index["runs"].append(
                {"model_key": spec.key, "run": spec.run, "epsilon": spec.epsilon, "path": str(out_path)}
            )
            continue

        result = _evaluate_one(spec, args, member_records, reference_records, registry)
        save_json(str(out_path), result)
        index["runs"].append(
            {"model_key": spec.key, "run": spec.run, "epsilon": spec.epsilon, "path": str(out_path)}
        )
        print(f"[saved] {out_path}")

        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(str(out_dir / "index.json"), index)
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
