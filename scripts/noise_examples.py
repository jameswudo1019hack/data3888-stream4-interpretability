"""Visual reference for the noise types used in v6 evaluation.
Saves figures/v5_grouped/noise/noise_examples.png — one example image per class,
each shown under blur / random mask / centre mask / edge mask at increasing
intensity.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import gaussian_blur as tv_blur

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "100"
OUT  = ROOT / "figures" / "v5_grouped" / "noise"
OUT.mkdir(parents=True, exist_ok=True)

# One representative crop per class (deterministic — first PNG in folder).
EXAMPLES = {
    "Tumour":  next((DATA / "Invasive_Tumor").glob("*.png")),
    "Immune":  next((DATA / "CD4+_T_Cells").glob("*.png")),
    "Stromal": next((DATA / "Stromal").glob("*.png")),
}

# Same noise levels as scripts/noise_eval.py
SIGMAS      = [0, 1, 2, 4, 8]
MASK_RANDOM = [0.0, 0.25, 0.5, 0.75]
MASK_CENTRE = [0.0, 0.3, 0.5, 0.7]
MASK_EDGE   = [1.0, 0.7, 0.5, 0.3]

# --- noise functions (CPU, single image — keep it simple for visualisation) ---

def blur(img, sigma):
    if sigma == 0: return img
    t = torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)
    k = int(2 * np.ceil(3 * sigma) + 1) | 1
    t = tv_blur(t, kernel_size=[k, k], sigma=[float(sigma), float(sigma)])
    return (t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def mask_random(img, p, seed=0):
    if p == 0: return img
    rng = np.random.default_rng(seed)
    m = rng.random(img.shape[:2]) < p
    out = img.copy(); out[m] = 0
    return out

def mask_centre(img, frac):
    if frac == 0: return img
    H, W, _ = img.shape
    h = int(H * frac); w = int(W * frac)
    y0 = (H - h) // 2; x0 = (W - w) // 2
    out = img.copy(); out[y0:y0+h, x0:x0+w] = 0
    return out

def mask_edge(img, keep):
    if keep == 1.0: return img
    H, W, _ = img.shape
    h = int(H * keep); w = int(W * keep)
    y0 = (H - h) // 2; x0 = (W - w) // 2
    out = np.zeros_like(img); out[y0:y0+h, x0:x0+w] = img[y0:y0+h, x0:x0+w]
    return out

NOISES = [
    ("Blur",         SIGMAS,      blur,        lambda x: f"σ={x}"),
    ("Random mask",  MASK_RANDOM, mask_random, lambda x: f"{int(x*100)}%"),
    ("Centre mask",  MASK_CENTRE, mask_centre, lambda x: f"{int(x*100)}%"),
    ("Edge mask",    MASK_EDGE,   mask_edge,   lambda x: f"keep={int(x*100)}%"),
]

# Compact figure: one example image (Tumour), all 4 noise types as rows.
img = np.asarray(Image.open(EXAMPLES["Tumour"]).convert("RGB"), dtype=np.uint8)

ncols = max(len(lv) for _, lv, _, _ in NOISES)
nrows = len(NOISES)
fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))

for r, (noise_name, levels, fn, fmt) in enumerate(NOISES):
    for c in range(ncols):
        ax = axes[r, c]
        if c < len(levels):
            lv = levels[c]
            ax.imshow(fn(img, lv))
            ax.set_title(fmt(lv), fontsize=10)
            if c == 0:
                ax.set_ylabel(noise_name, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

plt.suptitle("Noise types — increasing intensity →   (Tumour example)",
             fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(OUT / "noise_examples.png", dpi=140, bbox_inches="tight")
print(f"saved {OUT / 'noise_examples.png'}")

# Also save per-class, smaller figures
for cls, path in EXAMPLES.items():
    img2 = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    for r, (noise_name, levels, fn, fmt) in enumerate(NOISES):
        for c in range(ncols):
            ax = axes[r, c]
            if c < len(levels):
                ax.imshow(fn(img2, levels[c]))
                ax.set_title(fmt(levels[c]), fontsize=10)
                if c == 0:
                    ax.set_ylabel(noise_name, fontsize=12, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
    plt.suptitle(f"Noise types — {cls} example", fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(OUT / f"noise_examples_{cls.lower()}.png", dpi=140, bbox_inches="tight")
    print(f"saved {OUT / f'noise_examples_{cls.lower()}.png'}")
    plt.close()
