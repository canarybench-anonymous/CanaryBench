"""
Paper privacy figures from profile_matrix JSON logs.

Creates exactly the requested figures:
  Figure 1: Memorization gap vs privacy budget, with baseline shown at epsilon = infinity.
  Figure 2: Baseline-only WBC score KDEs, member vs non-member, one panel per model.
  Figure 3: Baseline-only member loss KDEs by canary frequency tier, one panel per model.

Run:
  conda activate dp-llm
  python profile_matrix/paper_privacy_graphs.py

Outputs:
  profile_matrix/paper_figures/figure1_gap_vs_epsilon.{pdf,png}
  profile_matrix/paper_figures/figure2_baseline_wbc_kde.{pdf,png}
  profile_matrix/paper_figures/figure3_baseline_loss_by_frequency.{pdf,png}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover - runtime fallback for light envs
    gaussian_kde = None


HERE = Path(__file__).resolve().parent
RESULT_RE = re.compile(r"^(?P<model>.+?)__(?P<run>baseline|dp_eps[0-9.]+)\.json$")
EPS_VALUES = [3.0, 5.0, 8.0, 16.0, 64.0]
BASELINE_X = 1.35
EPS_X = {eps: i + 2 for i, eps in enumerate(EPS_VALUES)}

MODEL_ORDER = ["vault_gemma", "gemma_2b_it", "gemma3_1b", "llama3_2_1b_it"]
MODEL_LABELS = {
    "vault_gemma": "VaultGemma",
    "gemma_2b_it": "Gemma-2-2B-IT",
    "gemma3_1b": "Gemma-3-1B",
    "llama3_2_1b_it": "Llama-3.2-1B-IT",
}
MODEL_COLORS = {
    "vault_gemma": "#8A5A99",
    "gemma_2b_it": "#B4472A",
    "gemma3_1b": "#2F7D6D",
    "llama3_2_1b_it": "#4568A9",
}

MEMBER_COLOR = "#4568A9"
NONMEMBER_COLOR = "#B4472A"
TIER_COLORS = {1: "#4568A9", 10: "#2F7D6D", 50: "#B4472A"}
TIER_LABELS = {1: "1x", 10: "10x", 50: "50x"}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def discover_runs(data_dir: Path) -> dict[tuple[str, str], Path]:
    runs: dict[tuple[str, str], Path] = {}
    for path in sorted(data_dir.glob("*.json")):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        data = load_json(path)
        model = data.get("model_key") or match.group("model")
        run = data.get("run") or match.group("run")
        runs[(model, run)] = path
    return runs


def ordered_models(runs: dict[tuple[str, str], Path]) -> list[str]:
    present = {model for model, _ in runs}
    return [model for model in MODEL_ORDER if model in present] + sorted(present - set(MODEL_ORDER))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)


def kde_or_hist(ax: plt.Axes, values, *, color: str, label: str, bw: float = 0.25, alpha: float = 0.25) -> None:
    values = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if values.size == 0:
        return
    if values.size < 5 or gaussian_kde is None or np.std(values) == 0:
        ax.hist(values, bins=28, density=True, alpha=alpha, color=color, label=label)
        return

    pad = max((values.max() - values.min()) * 0.12, 1e-3)
    xs = np.linspace(values.min() - pad, values.max() + pad, 500)
    kde = gaussian_kde(values, bw_method=bw)
    ys = kde(xs)
    ax.fill_between(xs, ys, color=color, alpha=alpha)
    ax.plot(xs, ys, color=color, linewidth=1.9, label=label)


def plot_gap_vs_epsilon(runs: dict[tuple[str, str], Path], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for model in ordered_models(runs):
        color = MODEL_COLORS.get(model, "#555555")
        xs: list[float] = []
        ys: list[float] = []

        baseline_path = runs.get((model, "baseline"))
        if baseline_path is not None:
            data = load_json(baseline_path)
            xs.append(BASELINE_X)
            ys.append(float((data.get("canary_exposure") or {}).get("gap", np.nan)))

        for eps in EPS_VALUES:
            path = runs.get((model, f"dp_eps{eps:.1f}"))
            if path is None:
                continue
            data = load_json(path)
            xs.append(EPS_X[eps])
            ys.append(float((data.get("canary_exposure") or {}).get("gap", np.nan)))

        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.1, markersize=5.5, color=color, label=model_label(model))
            if baseline_path is not None:
                ax.scatter([BASELINE_X], [ys[0]], s=82, color=color, edgecolor="#222222", zorder=4)

    ax.set_xticks([BASELINE_X] + [EPS_X[e] for e in EPS_VALUES], ["Baseline\n(eps=inf)"] + [f"{e:g}" for e in EPS_VALUES])
    ax.set_xlabel("Privacy budget epsilon")
    ax.set_ylabel("Memorization gap")
    ax.set_title("Figure 1: Memorization Gap vs Privacy Budget")
    ax.legend(frameon=False)
    style_axis(ax)
    save_figure(fig, out_dir, "figure1_gap_vs_epsilon")


def plot_baseline_wbc_kde(runs: dict[tuple[str, str], Path], out_dir: Path) -> None:
    models = ordered_models(runs)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharex=False, sharey=False)

    for ax, model in zip(axes.flat, models):
        path = runs.get((model, "baseline"))
        if path is None:
            ax.axis("off")
            continue
        data = load_json(path)
        wbc = data.get("wbc") or {}
        member = wbc.get("member_scores") or []
        reference = wbc.get("reference_scores") or []
        auc = float(wbc.get("wbc_auc", np.nan))

        kde_or_hist(ax, reference, color=NONMEMBER_COLOR, label="non-member", bw=0.22, alpha=0.24)
        kde_or_hist(ax, member, color=MEMBER_COLOR, label="member", bw=0.22, alpha=0.24)
        ax.set_title(f"{model_label(model)}\nWBC AUC={auc:.3f}")
        ax.set_xlabel("WBC membership score")
        ax.set_ylabel("Density")
        style_axis(ax)

    for ax in axes.flat[len(models) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Baseline WBC Score Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, "figure2_baseline_wbc_kde")


def plot_baseline_loss_by_frequency(runs: dict[tuple[str, str], Path], out_dir: Path) -> None:
    models = ordered_models(runs)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharex=False, sharey=False)

    for ax, model in zip(axes.flat, models):
        path = runs.get((model, "baseline"))
        if path is None:
            ax.axis("off")
            continue
        data = load_json(path)
        member_losses = (data.get("canary_exposure") or {}).get("member_losses") or []
        tier_losses = {1: [], 10: [], 50: []}
        for row in member_losses:
            if not isinstance(row, dict):
                continue
            tier = row.get("exposure")
            if tier in tier_losses and row.get("loss") is not None:
                tier_losses[tier].append(float(row["loss"]))

        for tier in [1, 10, 50]:
            values = tier_losses[tier]
            if values:
                kde_or_hist(
                    ax,
                    values,
                    color=TIER_COLORS[tier],
                    label=f"{TIER_LABELS[tier]} (n={len(values)})",
                    bw=0.24,
                    alpha=0.22,
                )

        ax.set_title(model_label(model))
        ax.set_xlabel("Per-token secret loss")
        ax.set_ylabel("Density")
        style_axis(ax)

    for ax in axes.flat[len(models) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Baseline Loss by Frequency Tier", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, "figure3_baseline_loss_by_frequency")


def write_loss_scale_check(runs: dict[tuple[str, str], Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ordered_models(runs):
        path = runs.get((model, "baseline"))
        if path is None:
            continue
        data = load_json(path)
        ce = data.get("canary_exposure") or {}
        member = [float(row["loss"]) for row in ce.get("member_losses", []) if isinstance(row, dict) and row.get("loss") is not None]
        reference = [float(row["loss"]) for row in ce.get("reference_losses", []) if isinstance(row, dict) and row.get("loss") is not None]
        by_tier = {}
        for tier in [1, 10, 50]:
            vals = [
                float(row["loss"])
                for row in ce.get("member_losses", [])
                if isinstance(row, dict) and row.get("exposure") == tier and row.get("loss") is not None
            ]
            by_tier[tier] = float(np.mean(vals)) if vals else np.nan
        rows.append(
            {
                "model": model,
                "member_mean": float(np.mean(member)),
                "reference_mean": float(np.mean(reference)),
                "gap": float(np.mean(reference) - np.mean(member)),
                "member_min": float(np.min(member)),
                "member_max": float(np.max(member)),
                "tier_1x_member_mean": by_tier[1],
                "tier_10x_member_mean": by_tier[10],
                "tier_50x_member_mean": by_tier[50],
            }
        )
    with (out_dir / "baseline_loss_scale_check.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_wbc_scale_check(runs: dict[tuple[str, str], Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ordered_models(runs):
        path = runs.get((model, "baseline"))
        if path is None:
            continue
        data = load_json(path)
        wbc = data.get("wbc") or {}
        for split, key in [("member", "member_scores"), ("reference", "reference_scores")]:
            values = np.asarray(wbc.get(key) or [], dtype=float)
            if values.size == 0:
                continue
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "wbc_auc": float(wbc.get("wbc_auc", np.nan)),
                }
            )
    with (out_dir / "baseline_wbc_scale_check.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the 3 recommended paper privacy figures.")
    parser.add_argument("--data-dir", type=Path, default=HERE, help="Directory containing profile_matrix JSON files.")
    parser.add_argument("--out-dir", type=Path, default=HERE / "paper_figures", help="Output directory.")
    args = parser.parse_args()

    runs = discover_runs(args.data_dir)
    if not runs:
        raise SystemExit(f"No profile-matrix JSON files found in {args.data_dir}")

    missing = []
    for model in ordered_models(runs):
        if (model, "baseline") not in runs:
            missing.append(f"{model}: baseline")
        for eps in EPS_VALUES:
            if (model, f"dp_eps{eps:.1f}") not in runs:
                missing.append(f"{model}: dp_eps{eps:.1f}")
    if missing:
        print("Warning: missing expected files:")
        for item in missing:
            print(f"  - {item}")

    plot_gap_vs_epsilon(runs, args.out_dir)
    plot_baseline_wbc_kde(runs, args.out_dir)
    plot_baseline_loss_by_frequency(runs, args.out_dir)
    write_loss_scale_check(runs, args.out_dir)
    write_wbc_scale_check(runs, args.out_dir)

    print(f"Wrote paper figures to {args.out_dir}")
    print("  figure1_gap_vs_epsilon.pdf")
    print("  figure2_baseline_wbc_kde.pdf")
    print("  figure3_baseline_loss_by_frequency.pdf")
    print("  baseline_loss_scale_check.csv")
    print("  baseline_wbc_scale_check.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
