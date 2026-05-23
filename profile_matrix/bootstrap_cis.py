import argparse
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    ROC AUC for binary labels where larger `scores` => more likely positive.
    Uses rank-based formula with average ranks for ties.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have same length")
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    # Average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_gap(member_losses, reference_losses, n_boot=10_000, ci=95, rng=None):
    """Bootstrap CI on memorization gap = mean(reference_losses) - mean(member_losses)."""
    member_losses = np.asarray(member_losses, dtype=float)
    reference_losses = np.asarray(reference_losses, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    gaps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        mem_sample = rng.choice(member_losses, size=len(member_losses), replace=True)
        ref_sample = rng.choice(reference_losses, size=len(reference_losses), replace=True)
        gaps[i] = ref_sample.mean() - mem_sample.mean()
    alpha = (100 - ci) / 2.0
    return float(gaps.mean()), float(np.percentile(gaps, alpha)), float(np.percentile(gaps, 100 - alpha))


def bootstrap_auc(member_scores, reference_scores, n_boot=10_000, ci=95, rng=None):
    """Bootstrap CI on ROC AUC for member-vs-reference scores."""
    member_scores = np.asarray(member_scores, dtype=float)
    reference_scores = np.asarray(reference_scores, dtype=float)
    all_scores = np.concatenate([member_scores, reference_scores])
    labels = np.array([1] * len(member_scores) + [0] * len(reference_scores), dtype=int)
    if rng is None:
        rng = np.random.default_rng()

    aucs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.choice(len(all_scores), size=len(all_scores), replace=True)
        aucs[i] = auc_roc(all_scores[idx], labels[idx])
    aucs = aucs[~np.isnan(aucs)]
    if aucs.size == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = (100 - ci) / 2.0
    return float(aucs.mean()), float(np.percentile(aucs, alpha)), float(np.percentile(aucs, 100 - alpha))


@dataclass(frozen=True)
class Row:
    model: str
    eps: str
    gap: float
    gap_lo: float
    gap_hi: float
    wbc_auc: float
    wbc_lo: float
    wbc_hi: float


FNAME_RE = re.compile(r"^(?P<model>.+?)__(?P<eps>baseline|dp_eps[0-9.]+)\.json$")


def coerce_numeric_list(values, *, dict_key: str) -> np.ndarray:
    """
    Coerce a list into a 1D float array.
    Supports either raw numbers or dicts containing `dict_key`.
    """
    if values is None:
        raise ValueError("values is None")
    if len(values) == 0:
        return np.array([], dtype=float)
    first = values[0]
    if isinstance(first, dict):
        out = []
        for v in values:
            if not isinstance(v, dict) or dict_key not in v:
                raise TypeError(f"Expected dict with key '{dict_key}', got: {type(v)}")
            out.append(v[dict_key])
        return np.asarray(out, dtype=float)
    return np.asarray(values, dtype=float)


def load_row(path: str, n_boot: int, ci: int, rng: np.random.Generator) -> Row:
    fname = os.path.basename(path)
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Unrecognized filename format: {fname}")
    model = m.group("model")
    eps = m.group("eps")

    data = json.load(open(path, "r"))

    # Memorization gap inputs
    ce = data.get("canary_exposure") or {}
    member_losses = ce.get("member_losses")
    reference_losses = ce.get("reference_losses")
    if member_losses is None or reference_losses is None:
        raise KeyError(f"{fname}: missing canary_exposure.member_losses/reference_losses")

    member_losses_arr = coerce_numeric_list(member_losses, dict_key="loss")
    reference_losses_arr = coerce_numeric_list(reference_losses, dict_key="loss")
    gap_mean, gap_lo, gap_hi = bootstrap_gap(
        member_losses_arr, reference_losses_arr, n_boot=n_boot, ci=ci, rng=rng
    )

    # WBC AUC inputs
    wbc = data.get("wbc") or {}
    member_scores = wbc.get("member_scores")
    reference_scores = wbc.get("reference_scores")
    if member_scores is None or reference_scores is None:
        raise KeyError(f"{fname}: missing wbc.member_scores/reference_scores")

    member_scores_arr = coerce_numeric_list(member_scores, dict_key="score")
    reference_scores_arr = coerce_numeric_list(reference_scores, dict_key="score")
    auc_mean, auc_lo, auc_hi = bootstrap_auc(
        member_scores_arr, reference_scores_arr, n_boot=n_boot, ci=ci, rng=rng
    )

    return Row(
        model=model,
        eps=eps,
        gap=gap_mean,
        gap_lo=gap_lo,
        gap_hi=gap_hi,
        wbc_auc=auc_mean,
        wbc_lo=auc_lo,
        wbc_hi=auc_hi,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_testing/profile_matrix", help="Directory of *__.json files")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--ci", type=int, default=95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_testing/profile_matrix/bootstrap_cis.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    paths = []
    for name in os.listdir(args.dir):
        if FNAME_RE.match(name):
            paths.append(os.path.join(args.dir, name))
    paths.sort()

    rows = []
    for p in paths:
        try:
            row = load_row(p, n_boot=args.n_boot, ci=args.ci, rng=rng)
            rows.append(row)
            print(
                f"{row.model:>16} {row.eps:>12}: "
                f"gap={row.gap:+.4f} [{row.gap_lo:+.4f}, {row.gap_hi:+.4f}]  "
                f"WBC={row.wbc_auc:.4f} [{row.wbc_lo:.4f}, {row.wbc_hi:.4f}]"
            )
        except Exception as e:
            print(f"Skipping {os.path.basename(p)}: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("model,eps,gap,gap_ci_lower,gap_ci_upper,wbc_auc,wbc_ci_lower,wbc_ci_upper\n")
        for r in rows:
            f.write(
                f"{r.model},{r.eps},{r.gap},{r.gap_lo},{r.gap_hi},{r.wbc_auc},{r.wbc_lo},{r.wbc_hi}\n"
            )


if __name__ == "__main__":
    main()
