"""
CanaryBench — Publication-Quality Figure Generator
Generates 3 figures from per-model JSON log files:
  Figure 1: Loss distribution (member vs non-member) — one panel per model
  Figure 2: Exposure distribution by frequency tier — one panel per model
  Figure 3: WBC score distribution (member vs non-member) — one panel per model

Usage:
  Place all JSON files in the same directory and set FILE_MAP below.
  Run: python3 plot_canarybench.py
  Outputs: figure1_loss_dist.pdf, figure2_exposure_dist.pdf, figure3_wbc_dist.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
# Map display name → path to JSON file
# Add all your model files here
FILE_MAP = {
    "Gemma-2-2B-IT": "/mnt/user-data/uploads/gemma_2b_it__baseline.json",
    # "Gemma-3-1B-IT":     "path/to/gemma3_1b__baseline.json",
    # "LLaMA-3.2-1B-IT":   "path/to/llama3_2_1b_it__baseline.json",
    # "VaultGemma-1B":     "path/to/vault_gemma__baseline.json",
}

OUTPUT_DIR = Path("/mnt/user-data/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Palette ─────────────────────────────────────────────────────────────────
MEMBER_COLOR    = "#4C72B0"   # blue
NONMEMBER_COLOR = "#DD8452"   # orange
FREQ_COLORS     = {1: "#4C72B0", 10: "#55A868", 50: "#C44E52"}

TIER_LABELS = {1: "1× (single)", 10: "10× (repeated)", 50: "50× (high-freq)"}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})

# ── Helpers ─────────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)

def kde_plot(ax, data, color, label, bw=0.3):
    data = np.array(data)
    if len(data) < 5:
        return
    try:
        kde = gaussian_kde(data, bw_method=bw)
        xs  = np.linspace(data.min() - 1, data.max() + 1, 500)
        ys  = kde(xs)
        ax.fill_between(xs, ys, alpha=0.35, color=color)
        ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
    except Exception:
        ax.hist(data, bins=30, density=True, alpha=0.4, color=color, label=label)

# ── Figure 1: Loss distributions ─────────────────────────────────────────────
def plot_loss_distributions(file_map):
    n = len(file_map)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (model_name, fpath) in zip(axes, file_map.items()):
        d   = load(fpath)
        ce  = d["canary_exposure"]
        m_losses   = [x["loss"] for x in ce["member_losses"]]
        ref_losses = [x["loss"] for x in ce["reference_losses"]] \
                     if "reference_losses" in ce else []

        # fall back to wbc scores if no reference_losses
        if not ref_losses:
            ref_losses = d["wbc"]["reference_scores"]
            m_losses   = d["wbc"]["member_scores"]
            xlabel = "WBC Score"
        else:
            xlabel = "Loss"

        kde_plot(ax, m_losses,   MEMBER_COLOR,    "member")
        kde_plot(ax, ref_losses, NONMEMBER_COLOR, "non-member")

        gap  = ce["gap"]
        auc  = d["wbc"]["wbc_auc"]
        ax.set_title(f"{model_name}\ngap={gap:+.3f}, AUC={auc:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Member vs. Non-Member Score Distributions (Baseline, ε = ∞)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "figure1_loss_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)

# ── Figure 2: Exposure by frequency tier ─────────────────────────────────────
def plot_exposure_by_tier(file_map):
    n = len(file_map)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (model_name, fpath) in zip(axes, file_map.items()):
        d  = load(fpath)
        ce = d["canary_exposure"]
        ml = ce["member_losses"]

        # group by exposure tier
        tier_losses = {1: [], 10: [], 50: []}
        for entry in ml:
            t = entry["exposure"]
            if t in tier_losses:
                tier_losses[t].append(entry["loss"])

        for tier, losses in tier_losses.items():
            if losses:
                kde_plot(ax, losses,
                         FREQ_COLORS[tier],
                         TIER_LABELS[tier], bw=0.25)

        ax.set_title(model_name)
        ax.set_xlabel("Loss")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Member Loss Distribution by Canary Frequency Tier (Baseline, ε = ∞)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "figure2_exposure_by_tier.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)

# ── Figure 3: WBC scores distribution ────────────────────────────────────────
def plot_wbc_distributions(file_map):
    n = len(file_map)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (model_name, fpath) in zip(axes, file_map.items()):
        d   = load(fpath)
        wbc = d["wbc"]
        m_scores   = wbc["member_scores"]
        ref_scores = wbc["reference_scores"]

        kde_plot(ax, m_scores,   MEMBER_COLOR,    "member")
        kde_plot(ax, ref_scores, NONMEMBER_COLOR, "non-member")

        auc = wbc["wbc_auc"]
        ax.set_title(f"{model_name}\nWBC AUC = {auc:.3f}")
        ax.set_xlabel("WBC Score")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                   label="chance (0.5)")

    fig.suptitle("WBC Membership Inference Score Distributions (Baseline, ε = ∞)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "figure3_wbc_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)

# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Figure 1: Loss distributions...")
    plot_loss_distributions(FILE_MAP)

    print("Generating Figure 2: Exposure by frequency tier...")
    plot_exposure_by_tier(FILE_MAP)

    print("Generating Figure 3: WBC score distributions...")
    plot_wbc_distributions(FILE_MAP)

    print("\nDone! All figures saved to:", OUTPUT_DIR)
    print("\nTo add more models, edit FILE_MAP at the top of this script.")