import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Models
# =========================================================
models = ["Gemma3", "Gemma2B", "Llama3", "VaultGemma"]

# =========================================================
# Data from your table
# =========================================================

# Baseline (ε = ∞)
baseline_1x  = [0.0498, 0.3898, 0.0571, 0.1802]
baseline_10x = [0.1814, 1.5217, 0.2970, 1.0287]
baseline_50x = [0.6936, 1.8655, 0.3416, 1.9775]

# ε = 3
eps3_1x  = [0.0213, 0.0266, 0.0149, 0.0294]
eps3_10x = [0.0086, 0.0035, 0.0189, 0.0089]
eps3_50x = [0.0745, 0.0596, -0.0043, 0.0980]

# ε = 64
eps64_1x  = [0.0255, 0.0317, 0.0181, 0.0281]
eps64_10x = [0.0067, -0.0013, 0.0223, 0.0111]
eps64_50x = [0.0938, 0.0792, 0.0053, 0.1090]

# =========================================================
# Common plotting settings
# =========================================================
x = np.arange(len(models))
width = 0.23

# Research-paper colors
color_1x  = "#4C72B0"
color_10x = "#55A868"
color_50x = "#C44E52"

# =========================================================
# Function to create figure
# =========================================================
def create_figure(title, gap1, gap10, gap50, save_name):

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Bars
    ax.bar(x - width, gap1,  width, label="1×",  color=color_1x)
    ax.bar(x,         gap10, width, label="10×", color=color_10x)
    ax.bar(x + width, gap50, width, label="50×", color=color_50x)

    # Labels
    ax.set_ylabel("Memorization Gap", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)

    ax.set_title(title, fontsize=13, pad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)

    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Remove top/right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend = ax.legend(
        title="Canary Frequency",
        frameon=False,
        fontsize=10
    )

    legend.get_title().set_fontsize(10)

    # Layout
    plt.tight_layout()

    # Save PNG
    plt.savefig(
        f"{save_name}.png",
        dpi=400,
        bbox_inches="tight"
    )

    # Save PDF
    plt.savefig(
        f"{save_name}.pdf",
        bbox_inches="tight"
    )

    plt.show()


# =========================================================
# Generate Figure 1 — Baseline
# =========================================================
create_figure(
    title="Frequency-Stratified Memorization Gaps at Baseline (ε = ∞)",
    gap1=baseline_1x,
    gap10=baseline_10x,
    gap50=baseline_50x,
    save_name="baseline_memorization_gaps"
)

# =========================================================
# Generate Figure 2 — ε = 3
# =========================================================
create_figure(
    title="Frequency-Stratified Memorization Gaps (ε = 3)",
    gap1=eps3_1x,
    gap10=eps3_10x,
    gap50=eps3_50x,
    save_name="epsilon3_memorization_gaps"
)

# =========================================================
# Generate Figure 3 — ε = 64
# =========================================================
create_figure(
    title="Frequency-Stratified Memorization Gaps (ε = 64)",
    gap1=eps64_1x,
    gap10=eps64_10x,
    gap50=eps64_50x,
    save_name="epsilon64_memorization_gaps"
)