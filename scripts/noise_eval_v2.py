"""v6.2 — refined centre vs edge mask methodology.

Improvements over scripts/noise_eval.py:
  1. H&E-pink fill instead of pure black (removes OOD-pixel confound).
  2. Area-fraction parameterisation — centre and edge sweeps share the same
     x-axis (fraction of pixels removed), so they're directly comparable.
  3. More fine-grained levels for smoother curves.
  4. Uniform plot axes for easy interpretability across models / classes.

Tests the tutor's hypothesis (Week 11):
  ResNet relies on the centre cell  →  centre-fill should hurt it more.
  ViT relies on edges/context        →  edge-fill should hurt it more.

Outputs land in figures/v5_grouped/noise_v2/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import timm

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v5_grouped"
OUT = ROOT / "figures" / VERSION / "noise_v2"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "Tumour":  ["DCIS_1", "DCIS_2", "Invasive_Tumor", "Prolif_Invasive_Tumor"],
    "Immune":  ["B_Cells", "CD4+_T_Cells", "CD8+_T_Cells",
                "Macrophages_1", "Macrophages_2",
                "LAMP3+_DCs", "IRF7+_DCs", "Mast_Cells"],
    "Stromal": ["Stromal", "Perivascular-Like", "Endothelial"],
}
CLASSES = sorted(GROUPS.keys())
SOURCE_MAP = {g: GROUPS[g] for g in CLASSES}

MAX_PER_CLASS = 1000
LOAD_SIZE = 100
DATA_DIR = ROOT / "data" / "100"

MODELS = ["resnet50", "vit_small_patch16_224"]
SEEDS  = [42, 43, 44, 45, 46]

# Area fractions to sweep (fraction of pixels replaced by fill colour)
AREA_FRACS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"device: {device}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- data ---------------------------------------------------------------

def load_images_for_seed(seed):
    X, y = [], []
    rng = np.random.default_rng(seed)
    for label, cls in enumerate(CLASSES):
        all_paths = []
        for src in SOURCE_MAP[cls]:
            all_paths.extend(sorted((DATA_DIR / src).glob("*.png")))
        if len(all_paths) > MAX_PER_CLASS:
            pick = rng.choice(len(all_paths), MAX_PER_CLASS, replace=False)
            all_paths = [all_paths[i] for i in pick]
        for p in all_paths:
            img = Image.open(p).convert("RGB").resize((LOAD_SIZE, LOAD_SIZE))
            X.append(np.asarray(img, dtype=np.uint8))
            y.append(label)
    return np.stack(X), np.array(y)

def get_test_set(seed):
    X, y = load_images_for_seed(seed)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20,
                                        stratify=y, random_state=seed)
    return X_te, y_te

# --- masking -------------------------------------------------------------
# Both centre and edge are parameterised by `area_frac` — fraction of pixels replaced.

def _side_for_area(area_frac, H):
    """Centred-square side length (px) for a given area fraction (≤ 1)."""
    return int(round(np.sqrt(max(0.0, min(1.0, area_frac))) * H))

def fill_centre(X, area_frac, fill):
    """Replace a centred square of given area fraction with `fill`."""
    if area_frac <= 0:
        return X.copy()
    N, H, W, _ = X.shape
    h = _side_for_area(area_frac, H); w = _side_for_area(area_frac, W)
    if h == 0 or w == 0:
        return X.copy()
    y0 = (H - h) // 2; x0 = (W - w) // 2
    out = X.copy()
    out[:, y0:y0+h, x0:x0+w] = fill
    return out

def fill_edge(X, area_frac, fill):
    """Replace edges (everything outside a centred square) with `fill`,
    such that the FILLED area (i.e. the edges) covers `area_frac`."""
    if area_frac <= 0:
        return X.copy()
    keep_area = 1.0 - area_frac
    N, H, W, _ = X.shape
    h = _side_for_area(keep_area, H); w = _side_for_area(keep_area, W)
    y0 = (H - h) // 2; x0 = (W - w) // 2
    out = np.full_like(X, fill)
    if h > 0 and w > 0:
        out[:, y0:y0+h, x0:x0+w] = X[:, y0:y0+h, x0:x0+w]
    return out

# --- inference -----------------------------------------------------------

def to_tensor(X):
    arr = X.astype(np.float32) / 255.
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))

def resize224(x):
    return F.interpolate(x, size=224, mode="bilinear", align_corners=False)

def predict_proba(model, X_t, batch=32):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            xb = resize224(X_t[i:i+batch].to(device))
            out.append(F.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()

# --- main loop -----------------------------------------------------------

rows = []
for seed in SEEDS:
    print(f"\n=== seed {seed} ===")
    X_te, y_te = get_test_set(seed)

    # H&E-pink fill: per-channel mean of THIS test set (defensible, deterministic)
    he_pink = X_te.mean(axis=(0, 1, 2)).astype(np.uint8)
    print(f"  test set: {X_te.shape}, H&E pink RGB ~= {tuple(int(c) for c in he_pink)}")

    for model_name in MODELS:
        ckpt_path = ROOT / "figures" / VERSION / model_name / f"seed_{seed}" / f"{model_name}_stream4.pt"
        if not ckpt_path.exists():
            print(f"  [skip] no checkpoint at {ckpt_path}")
            continue
        print(f"  loading {model_name} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = timm.create_model(model_name, pretrained=False,
                                  num_classes=len(CLASSES)).to(device)
        model.load_state_dict(ckpt["model_state"])

        for mask_name, fn in [("centre", fill_centre), ("edge", fill_edge)]:
            for p in AREA_FRACS:
                X_noisy = fn(X_te, p, fill=he_pink)
                X_noisy_t = to_tensor(X_noisy)
                proba = predict_proba(model, X_noisy_t)
                pred  = proba.argmax(axis=1)
                conf  = proba.max(axis=1)
                acc   = (pred == y_te).mean()
                f1_per = f1_score(y_te, pred, average=None,
                                  labels=list(range(len(CLASSES))))
                wrong = pred != y_te
                rows.append({
                    "seed": seed, "model": model_name,
                    "mask_type": mask_name, "area_frac": p,
                    "accuracy": acc,
                    **{f"f1_{c}": f1_per[i] for i, c in enumerate(CLASSES)},
                    "conf_correct": conf[~wrong].mean(),
                    "conf_wrong":   conf[wrong].mean() if wrong.any() else float("nan"),
                })
                print(f"    {mask_name:<6} area={p:.2f}  acc={acc:.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "results.csv", index=False)
print(f"\nsaved {OUT / 'results.csv'}  ({len(df)} rows)")

# --- plots ---------------------------------------------------------------

plt.rcParams.update({"font.size": 11})
COLORS = {"resnet50": "#1f77b4", "vit_small_patch16_224": "#d62728"}
SHORT  = {"resnet50": "ResNet50", "vit_small_patch16_224": "ViT-small"}

# Mask type styling: solid line for centre fill, long-dash for edge fill.
# No markers (tutor feedback Week 12: line style alone with a clear legend).
MASK_STYLES = [
    ("centre", "-",         "Centre fill (solid)"),
    ("edge",   (0, (6, 3)), "Edge fill (dashed)"),
]

# 1) Overall accuracy: 2 subplots (per model) with uniform axes.
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

# 2) Combined single-plot view (4 lines, uniform axes).
# Two legends: colour = architecture, line style = mask type.
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

# Mask-type legend — line style encodes which region was filled.
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

# 3) Per-class breakdown — 2 rows (model) × 3 cols (class), uniform axes.
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

# 4) Visual reference: what the fills look like at a few area fractions
sample = X_te[0]            # any image; pink mean already computed
he_pink_local = X_te.mean(axis=(0, 1, 2)).astype(np.uint8)
fracs_to_show = [0.0, 0.1, 0.3, 0.5, 0.7]
fig, axes = plt.subplots(2, len(fracs_to_show),
                         figsize=(2.4 * len(fracs_to_show), 5))
for c, p in enumerate(fracs_to_show):
    img_c = fill_centre(sample[None], p, fill=he_pink_local)[0]
    img_e = fill_edge(sample[None], p, fill=he_pink_local)[0]
    axes[0, c].imshow(img_c); axes[0, c].set_title(f"area={p}")
    axes[1, c].imshow(img_e)
    for r in (0, 1):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
axes[0, 0].set_ylabel("Centre fill", fontweight="bold")
axes[1, 0].set_ylabel("Edge fill",   fontweight="bold")
fig.suptitle("Fill examples (H&E pink)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fill_examples.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fill_examples.png")

print("\nDone.")
