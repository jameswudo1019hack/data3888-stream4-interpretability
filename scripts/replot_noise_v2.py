"""Regenerate the v6.1 noise-eval figures from the cached results.csv —
no inference, no GPU. Use this whenever the plotting code changes but the
numbers haven't (which is most of the time).

Mirrors the plotting blocks in scripts/noise_eval_v2.py. If you change the
look there, change it here too.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "v5_grouped" / "noise_v2"
CSV = OUT / "results.csv"
if not CSV.exists():
    raise SystemExit(f"results.csv not found at {CSV} — run noise_eval_v2.py first.")

df = pd.read_csv(CSV)
print(f"loaded {len(df)} rows from {CSV}")

MODELS = ["resnet50", "vit_small_patch16_224"]
CLASSES = ["Immune", "Stromal", "Tumour"]

plt.rcParams.update({"font.size": 11})
COLORS = {"resnet50": "#1f77b4", "vit_small_patch16_224": "#d62728"}
SHORT  = {"resnet50": "ResNet50", "vit_small_patch16_224": "ViT-small"}

# Mask type styling: solid line = centre fill, long-dash = edge fill.
# No markers (tutor feedback: "stick to dotted line and solid line").
MASK_STYLES = [
    ("centre", "-",         "Centre fill (solid)"),
    ("edge",   (0, (6, 3)), "Edge fill (dashed)"),
]

# 1) Overall accuracy: 2 subplots (per model).
fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.5), sharey=True)
for ax, model in zip(axes, MODELS):
    sub = df[df["model"] == model]
    for mask_name, ls, label in MASK_STYLES:
        s = sub[sub["mask_type"] == mask_name]
        agg = s.groupby("area_frac")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["area_frac"], agg["mean"], yerr=agg["std"],
                    label=label, linestyle=ls,
                    capsize=3, color=COLORS[model], linewidth=2.6)
    ax.set_title(SHORT[model], fontweight="bold")
    ax.set_xlabel("Fraction of image filled with H&E-pink")
    ax.set_xlim(-0.02, 0.72); ax.set_ylim(0, 1.0)
    ax.axhline(1/3, ls=":", c="grey", alpha=0.6, label="Chance (1/3)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", handlelength=3.5, framealpha=0.95)
axes[0].set_ylabel("Test accuracy (mean ± std, 5 seeds)")
fig.suptitle("Accuracy under H&E-pink fill — ResNet50 vs ViT-small",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT / "overall.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved overall.png")

# 2) Combined single-plot view with TWO legends (architecture | mask type).
fig, ax = plt.subplots(figsize=(9, 5))
for model in MODELS:
    sub = df[df["model"] == model]
    for mask_name, ls, _ in MASK_STYLES:
        s = sub[sub["mask_type"] == mask_name]
        agg = s.groupby("area_frac")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["area_frac"], agg["mean"], yerr=agg["std"],
                    linestyle=ls,
                    capsize=3, color=COLORS[model], linewidth=2.6)
chance_line = ax.axhline(1/3, ls=":", c="grey", alpha=0.6, label="Chance (1/3)")
ax.set_xlabel("Fraction of image filled with H&E-pink")
ax.set_ylabel("Test accuracy (mean ± std, 5 seeds)")
ax.set_xlim(-0.02, 0.72); ax.set_ylim(0, 1.0)
ax.grid(alpha=0.3)
ax.set_title("Accuracy under H&E-pink fill — both architectures",
             fontsize=13, fontweight="bold")

# Architecture legend — colour encodes which model.
arch_handles = [
    plt.Line2D([0], [0], color=COLORS["resnet50"], linewidth=2.8, label="ResNet50"),
    plt.Line2D([0], [0], color=COLORS["vit_small_patch16_224"],
               linewidth=2.8, label="ViT-small"),
]
arch_legend = ax.legend(handles=arch_handles, loc="upper right",
                        title="Architecture (colour)", framealpha=0.95)
ax.add_artist(arch_legend)

# Mask-type legend — linestyle encodes which region was filled.
mask_handles = [
    plt.Line2D([0], [0], color="black", linestyle="-",
               linewidth=2.6, label="Centre fill (solid line)"),
    plt.Line2D([0], [0], color="black", linestyle=(0, (6, 3)),
               linewidth=2.6, label="Edge fill (dashed line)"),
    chance_line,
]
ax.legend(handles=mask_handles, loc="lower left",
          title="Mask type (line style)", handlelength=3.8, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT / "combined.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved combined.png")

# 3) Per-class breakdown — 2 rows × 3 cols. ONE figure-level legend so the
# small panels aren't crowded.
fig, axes = plt.subplots(len(MODELS), len(CLASSES),
                         figsize=(4 * len(CLASSES), 3.5 * len(MODELS)),
                         sharex=True, sharey=True)
for r, model in enumerate(MODELS):
    for c, cls in enumerate(CLASSES):
        ax = axes[r, c]
        sub = df[df["model"] == model]
        for mask_name, ls, label in MASK_STYLES:
            s = sub[sub["mask_type"] == mask_name]
            agg = s.groupby("area_frac")[f"f1_{cls}"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["area_frac"], agg["mean"], yerr=agg["std"],
                        label=label, linestyle=ls,
                        capsize=3, color=COLORS[model], linewidth=2.4)
        if r == 0:
            ax.set_title(cls, fontweight="bold")
        if c == 0:
            ax.set_ylabel(f"{SHORT[model]}\nF1 (mean ± std)")
        if r == len(MODELS) - 1:
            ax.set_xlabel("Fraction of image filled")
        ax.set_xlim(-0.02, 0.72); ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)

# Figure-level legend explaining the two line styles.
mask_handles = [
    plt.Line2D([0], [0], color="black", linestyle="-",
               linewidth=2.4, label="Centre fill (solid line)"),
    plt.Line2D([0], [0], color="black", linestyle=(0, (6, 3)),
               linewidth=2.4, label="Edge fill (dashed line)"),
]
fig.legend(handles=mask_handles, loc="upper center",
           bbox_to_anchor=(0.5, 0.96),
           ncol=2, frameon=False, fontsize=11, handlelength=3.5)

fig.suptitle("Per-class F1 under H&E-pink fill",
             fontsize=13, fontweight="bold", y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT / "per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved per_class.png")

print("\ndone.")
