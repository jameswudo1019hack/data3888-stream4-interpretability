"""Reload a saved checkpoint, recompute predictions on the deterministic
test split, and re-render confusion_matrix.png with the correct title.

Usage:  python scripts/render_confusion.py
"""
import sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import timm

# Same config as scripts/stream4_pipeline.py — must stay in sync to keep the
# split deterministic. If you bump SEED / MAX_PER_CLASS / GROUPS in the main
# pipeline, mirror it here.
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "100"

VERSION    = "v4_grouped_resnet50"
MODEL_NAME = "resnet50"
MAX_PER_CLASS = 3000

GROUPS = {
    "Tumour":  ["DCIS_1", "DCIS_2", "Invasive_Tumor", "Prolif_Invasive_Tumor"],
    "Immune":  ["B_Cells", "CD4+_T_Cells", "CD8+_T_Cells",
                "Macrophages_1", "Macrophages_2",
                "LAMP3+_DCs", "IRF7+_DCs", "Mast_Cells"],
    "Stromal": ["Stromal", "Perivascular-Like", "Endothelial"],
}

FIG_DIR = ROOT / "figures" / VERSION
classes = sorted(GROUPS.keys())

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"device: {device}")

# Reload data deterministically
def load_images(classes, source_map, max_per_class, size=100):
    X, y = [], []
    rng = np.random.default_rng(SEED)
    for label, cls in enumerate(classes):
        all_paths = []
        for src in source_map[cls]:
            all_paths.extend(sorted((DATA_DIR / src).glob("*.png")))
        if len(all_paths) > max_per_class:
            pick = rng.choice(len(all_paths), max_per_class, replace=False)
            all_paths = [all_paths[i] for i in pick]
        for p in all_paths:
            img = Image.open(p).convert("RGB").resize((size, size))
            X.append(np.asarray(img, dtype=np.uint8))
            y.append(label)
    return np.stack(X), np.array(y)

print("loading data ...")
source_map = {g: GROUPS[g] for g in classes}
X, y = load_images(classes, source_map, MAX_PER_CLASS)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                          stratify=y, random_state=SEED)
print(f"test set: {X_te.shape}  class balance: {np.bincount(y_te)}")

# To tensor
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
def to_tensor(X):
    arr = X.astype(np.float32)/255.
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))
X_te_t = to_tensor(X_te)

# Reload model
ckpt_path = FIG_DIR / f"{MODEL_NAME}_stream4.pt"
print(f"loading {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = timm.create_model(MODEL_NAME, pretrained=False,
                          num_classes=len(classes)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Predict
print("predicting ...")
preds = []
with torch.no_grad():
    for i in range(0, len(X_te_t), 32):
        xb = F.interpolate(X_te_t[i:i+32].to(device), size=224,
                           mode="bilinear", align_corners=False)
        preds.append(model(xb).argmax(1).cpu())
te_pred = torch.cat(preds).numpy()
print(f"accuracy: {(te_pred == y_te).mean():.4f}")

# Re-render confusion matrix with correct title
cm = confusion_matrix(y_te, te_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(1.1*len(classes)+2, 1.0*len(classes)+1))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=10)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=10)
plt.colorbar(im); plt.title(f"{MODEL_NAME} — test confusion (counts)")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
print(f"saved {FIG_DIR / 'confusion_matrix.png'}")
