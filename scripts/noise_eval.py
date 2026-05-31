"""v6 — noise robustness evaluation.

For each (model, seed) checkpoint trained in v5, reload the deterministic
test set and evaluate accuracy / per-class F1 / confidence under:
  - Gaussian blur at varying sigma
  - random pixel masking at varying fraction
  - centred-square masking (black centre)
  - edge masking (keep only centre)

Test-time only — no retraining. Produces:
  figures/v5_grouped/noise/results.csv          long-format results
  figures/v5_grouped/noise/<cond>.png           accuracy curves with error bars
  figures/v5_grouped/noise/<cond>_per_class.png per-class F1 curves

Usage:  python scripts/noise_eval.py
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
from torchvision.transforms.functional import gaussian_blur as tv_blur
import timm

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v5_grouped"
OUT = ROOT / "figures" / VERSION / "noise"
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

SIGMAS      = [0, 1, 2, 4, 8]                # Gaussian blur stddev (px)
MASK_RANDOM = [0.0, 0.25, 0.5, 0.75]         # fraction of pixels blacked
MASK_CENTRE = [0.0, 0.3, 0.5, 0.7]           # side fraction of centred black square
MASK_EDGE   = [1.0, 0.7, 0.5, 0.3]           # kept centre fraction

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"device: {device}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- data ---------------------------------------------------------------

def load_images_for_seed(seed):
    """Reproduces the deterministic load in stream4_pipeline.py."""
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

# --- noise functions ----------------------------------------------------

def apply_blur(X, sigma):
    if sigma == 0:
        return X
    # Run on GPU/MPS — at high sigma the kernel is large (σ=8 → 49×49) and
    # CPU-side conv is the bottleneck. Process in chunks to bound memory.
    k = int(2 * np.ceil(3 * sigma) + 1) | 1
    chunks = []
    for i in range(0, len(X), 256):
        t = (torch.from_numpy(X[i:i+256].astype(np.float32) / 255.)
             .permute(0, 3, 1, 2).to(device))
        t = tv_blur(t, kernel_size=[k, k], sigma=[float(sigma), float(sigma)])
        chunks.append((t.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8))
    return np.concatenate(chunks, axis=0)

def apply_mask_random(X, p, seed=0):
    if p == 0:
        return X
    rng = np.random.default_rng(seed * 1000 + int(p * 1000))
    mask = rng.random(X.shape[:3]) < p
    out = X.copy()
    out[mask] = 0
    return out

def apply_mask_centre(X, frac):
    if frac == 0:
        return X
    N, H, W, _ = X.shape
    h = int(H * frac); w = int(W * frac)
    y0 = (H - h) // 2; x0 = (W - w) // 2
    out = X.copy()
    out[:, y0:y0 + h, x0:x0 + w] = 0
    return out

def apply_mask_edge(X, keep_frac):
    if keep_frac == 1.0:
        return X
    N, H, W, _ = X.shape
    h = int(H * keep_frac); w = int(W * keep_frac)
    y0 = (H - h) // 2; x0 = (W - w) // 2
    out = np.zeros_like(X)
    out[:, y0:y0 + h, x0:x0 + w] = X[:, y0:y0 + h, x0:x0 + w]
    return out

# --- inference ----------------------------------------------------------

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
            xb = resize224(X_t[i:i + batch].to(device))
            out.append(F.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()

# --- main loop ----------------------------------------------------------

rows = []
for seed in SEEDS:
    print(f"\n=== seed {seed} ===")
    X_te, y_te = get_test_set(seed)
    print(f"  test set: {X_te.shape}, balance {np.bincount(y_te)}")

    for model_name in MODELS:
        ckpt_path = ROOT / "figures" / VERSION / model_name / f"seed_{seed}" / f"{model_name}_stream4.pt"
        if not ckpt_path.exists():
            print(f"  [skip] no checkpoint: {ckpt_path}")
            continue
        print(f"  loading {model_name} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = timm.create_model(model_name, pretrained=False,
                                  num_classes=len(CLASSES)).to(device)
        model.load_state_dict(ckpt["model_state"])

        for cond_name, levels, fn in [
            ("blur",        SIGMAS,      lambda X, lv: apply_blur(X, lv)),
            ("mask_random", MASK_RANDOM, lambda X, lv: apply_mask_random(X, lv, seed=seed)),
            ("mask_centre", MASK_CENTRE, lambda X, lv: apply_mask_centre(X, lv)),
            ("mask_edge",   MASK_EDGE,   lambda X, lv: apply_mask_edge(X, lv)),
        ]:
            for level in levels:
                X_noisy = fn(X_te, level)
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
                    "noise_type": cond_name, "level": level,
                    "accuracy": acc,
                    **{f"f1_{c}": f1_per[i] for i, c in enumerate(CLASSES)},
                    "conf_correct": conf[~wrong].mean(),
                    "conf_wrong":   conf[wrong].mean() if wrong.any() else float("nan"),
                })
                print(f"    {cond_name:<11} level={level:>5}  acc={acc:.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "results.csv", index=False)
print(f"\nsaved {OUT / 'results.csv'}  ({len(df)} rows)")

# --- plots --------------------------------------------------------------

def plot_acc(df, cond, x_label, save_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df["noise_type"] == cond]
    for model in MODELS:
        s = sub[sub["model"] == model]
        if s.empty: continue
        agg = s.groupby("level")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["level"], agg["mean"], yerr=agg["std"],
                    label=model, marker="o", capsize=4)
    ax.set_xlabel(x_label); ax.set_ylabel("test accuracy")
    ax.set_title(f"Accuracy under {cond}  (mean ± std across 5 seeds)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / save_name, dpi=150); plt.close()

def plot_per_class(df, cond, x_label, save_name):
    fig, axes = plt.subplots(1, len(MODELS),
                             figsize=(6 * len(MODELS), 4), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    sub = df[df["noise_type"] == cond]
    for ax, model in zip(axes, MODELS):
        s = sub[sub["model"] == model]
        if s.empty: continue
        for cls in CLASSES:
            agg = s.groupby("level")[f"f1_{cls}"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["level"], agg["mean"], yerr=agg["std"],
                        label=cls, marker="o", capsize=3)
        ax.set_xlabel(x_label); ax.set_ylabel("F1")
        ax.set_title(f"{model} — F1 vs {cond}")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / save_name, dpi=150); plt.close()

CONDS = [
    ("blur",        "Gaussian σ"),
    ("mask_random", "fraction pixels masked"),
    ("mask_centre", "side fraction of centred mask"),
    ("mask_edge",   "kept centre fraction"),
]
for cond, xlab in CONDS:
    plot_acc(df, cond, xlab, f"{cond}.png")
    plot_per_class(df, cond, xlab, f"{cond}_per_class.png")

print(f"\nsaved plots to {OUT}/")
