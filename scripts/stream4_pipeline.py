"""DATA3888 Stream 4 — cell-identity classification pipeline.

Fork of DA2 binary pipeline, adapted for multi-class cluster classification.
ViT fine-tune (complement to Hossion's CNN), per-cluster average Grad-CAM.

Run as a script:   python stream4_pipeline.py
Or step through in VS Code — cells separated by `# %%`.
"""

# %% --------------------------------------------------------------------
# 1. Setup
import os, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import timm
from pytorch_grad_cam import GradCAM
from tqdm.auto import tqdm

# Script lives at scripts/stream4_pipeline.py; project root is one level up.
ROOT = Path(__file__).resolve().parent.parent

# --- experiment config ---------------------------------------------------
# Override any of these via env var: SEED=43 MODEL=vit_small_patch16_224 ...
VERSION      = os.environ.get("VERSION",      "v5_grouped")     # subdir name
USE_GROUPS   = True                                              # collapse fine clusters → broad groups
MODEL_NAME   = os.environ.get("MODEL",        "resnet50")        # "resnet50" or "vit_small_patch16_224"
IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", "100")             # "100" (cell+context) or "50" (cell only)
SEED         = int(os.environ.get("SEED",     "42"))             # changes train/val/test split + sample subset
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", "1000"))     # tutor (Week 9): smaller + repeats > one big run
# -------------------------------------------------------------------------

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR  = ROOT / "data" / IMAGE_FOLDER
LOAD_SIZE = int(IMAGE_FOLDER)         # native image size — don't upscale at load

# Broad-group mapping — tutor directive Week 8 (2026-04-20):
#   3 classes only: Tumour / Immune / Stromal.
#   "Other" dropped per tutor (too noisy). Myoepi + Hybrid folders excluded —
#   tutor explicitly said the three biological groups above are the scope.
GROUPS = {
    "Tumour":  ["DCIS_1", "DCIS_2", "Invasive_Tumor", "Prolif_Invasive_Tumor"],
    "Immune":  ["B_Cells", "CD4+_T_Cells", "CD8+_T_Cells",
                "Macrophages_1", "Macrophages_2",
                "LAMP3+_DCs", "IRF7+_DCs", "Mast_Cells"],
    "Stromal": ["Stromal", "Perivascular-Like", "Endothelial"],
}

# Per-seed subdir so multi-seed runs don't collide.
FIG_DIR = ROOT / "figures" / VERSION / MODEL_NAME / f"seed_{SEED}"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"output dir: {FIG_DIR}")
print(f"config: VERSION={VERSION}  MODEL={MODEL_NAME}  IMG={IMAGE_FOLDER}  SEED={SEED}  MAX_PER_CLASS={MAX_PER_CLASS}")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
print(f"device: {device}")

# %% --------------------------------------------------------------------
# 2. Class / group selection
counts = {d.name: len(list(d.glob("*.png")))
          for d in DATA_DIR.iterdir() if d.is_dir()}
for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {c:<32} {n:>6}")

if USE_GROUPS:
    classes = sorted(GROUPS.keys())
    source_map = {g: GROUPS[g] for g in classes}
    print(f"\nUsing {len(classes)} GROUPED classes (cap {MAX_PER_CLASS}/group, balanced):")
    for g in classes:
        total = sum(counts.get(s, 0) for s in source_map[g])
        print(f"  {g:<10} <- {len(source_map[g])} folders, {total} images available")
else:
    MIN_PER_CLASS = 500
    classes = sorted(c for c, n in counts.items() if n >= MIN_PER_CLASS)
    source_map = {c: [c] for c in classes}
    print(f"\nUsing {len(classes)} fine classes (>= {MIN_PER_CLASS}):")
    for c in classes: print(f"  {c}")

# %% --------------------------------------------------------------------
# 3. Load images into memory (uniform per-group sampling)
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

t0 = time.time()
X, y = load_images(classes, source_map, MAX_PER_CLASS, size=LOAD_SIZE)
print(f"\nloaded {X.shape} in {time.time()-t0:.1f}s  class balance: {np.bincount(y)}")

# %% --------------------------------------------------------------------
# 4. Stratified train / val / test split (70 / 10 / 20)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                          stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.125,
                                          stratify=y_tr, random_state=SEED)
print(f"train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}")

# %% --------------------------------------------------------------------
# 5. To tensors (ImageNet normalised, kept at 100x100; resize to 224 in the loop)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_tensor(X):
    arr = X.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))

X_tr_t = to_tensor(X_tr); y_tr_t = torch.from_numpy(y_tr).long()
X_va_t = to_tensor(X_va); y_va_t = torch.from_numpy(y_va).long()
X_te_t = to_tensor(X_te); y_te_t = torch.from_numpy(y_te).long()

def resize224(x):  # on-device resize to keep memory low
    return F.interpolate(x, size=224, mode="bilinear", align_corners=False)

# %% --------------------------------------------------------------------
# 6. Model — pretrained backbone, fine-tune end-to-end
model = timm.create_model(MODEL_NAME, pretrained=True,
                          num_classes=len(classes)).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{MODEL_NAME}  trainable params: {n_params/1e6:.1f}M")

# %% --------------------------------------------------------------------
# 7. Training loop — v3: augmentation + class-weighted loss + early stopping
EPOCHS    = 10
BATCH     = 32
LR        = 3e-5
PATIENCE  = 3  # stop if val_acc doesn't improve for this many epochs

def augment(xb):
    # cells have no canonical orientation: flip both axes + random 90° rotation
    if torch.rand(1).item() < 0.5: xb = torch.flip(xb, dims=[3])  # h-flip
    if torch.rand(1).item() < 0.5: xb = torch.flip(xb, dims=[2])  # v-flip
    k = torch.randint(0, 4, (1,)).item()
    if k: xb = torch.rot90(xb, k=k, dims=[2, 3])
    return xb

# Tutor (Week 8): use balanced sampling (already done at load time) instead
# of class weights. So plain CrossEntropyLoss here.
print("class balance (train):", np.bincount(y_tr))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=BATCH, shuffle=True)

def predict(model, X_t, batch=32):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            xb = resize224(X_t[i:i+batch].to(device))
            out.append(model(xb).argmax(1).cpu())
    return torch.cat(out).numpy()

def predict_proba(model, X_t, batch=32):
    """Return softmax probabilities (N, n_classes). Used for confidence analysis."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            xb = resize224(X_t[i:i+batch].to(device))
            out.append(F.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()

history = []
best_val = -1.0
best_state = None
bad = 0
for ep in range(EPOCHS):
    model.train()
    tloss = 0.0; n = 0; t0 = time.time()
    pbar = tqdm(train_loader, desc=f"epoch {ep+1}/{EPOCHS}",
                mininterval=2.0, ascii=True, dynamic_ncols=True)
    for xb, yb in pbar:
        xb = augment(xb)
        xb = resize224(xb.to(device)); yb = yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        tloss += loss.item() * xb.size(0); n += xb.size(0)
        pbar.set_postfix(loss=f"{tloss/n:.3f}")
    va_pred = predict(model, X_va_t)
    va_acc  = (va_pred == y_va).mean()
    dt = time.time() - t0
    history.append({"epoch": ep+1, "train_loss": tloss/n,
                    "val_acc": va_acc, "seconds": dt})
    improved = va_acc > best_val
    if improved:
        best_val = va_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
        flag = "  *"
    else:
        bad += 1
        flag = f"  (no-improve {bad}/{PATIENCE})"
    print(f"epoch {ep+1}: train_loss={tloss/n:.3f}  val_acc={va_acc:.3f}  ({dt:.0f}s){flag}")
    if bad >= PATIENCE:
        print(f"early stop at epoch {ep+1}; best val_acc={best_val:.3f}")
        break

# restore best weights before eval
model.load_state_dict(best_state)
print(f"restored best model (val_acc={best_val:.3f})")

# %% --------------------------------------------------------------------
# 8. Test-set metrics + confusion matrix + softmax probabilities
te_proba = predict_proba(model, X_te_t)              # (N, n_classes) softmax
te_pred  = te_proba.argmax(axis=1)
te_conf  = te_proba.max(axis=1)                      # confidence of the chosen class
print("\n" + classification_report(y_te, te_pred, target_names=classes, digits=3))

# Save model + metrics + raw predictions IMMEDIATELY — don't lose training time
# to a downstream viz bug.
pd.DataFrame(history).to_csv(FIG_DIR / "train_history.csv", index=False)
report = classification_report(y_te, te_pred, target_names=classes,
                               digits=3, output_dict=True)
pd.DataFrame(report).T.to_csv(FIG_DIR / "test_metrics.csv")
np.save(FIG_DIR / "test_y.npy",     y_te)
np.save(FIG_DIR / "test_pred.npy",  te_pred)
np.save(FIG_DIR / "test_proba.npy", te_proba)        # for confidence-vs-correctness analysis
torch.save({"model_state": model.state_dict(),
            "classes": classes,
            "config": {"seed": SEED, "model": MODEL_NAME,
                       "image_folder": IMAGE_FOLDER,
                       "max_per_class": MAX_PER_CLASS,
                       "epochs": EPOCHS, "batch": BATCH, "lr": LR}},
           FIG_DIR / f"{MODEL_NAME}_stream4.pt")
print(f"saved model + metrics + probs to {FIG_DIR}")
print(f"mean confidence: correct={te_conf[te_pred==y_te].mean():.3f}  "
      f"wrong={te_conf[te_pred!=y_te].mean():.3f}")

DISPLAY = {"resnet50": "ResNet50",
           "vit_small_patch16_224": "ViT-small"}.get(MODEL_NAME, MODEL_NAME)

cm = confusion_matrix(y_te, te_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(1.1*len(classes)+2, 1.0*len(classes)+1))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=9)
ax.set_xlabel("Predicted class", fontsize=10)
ax.set_ylabel("True class", fontsize=10)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=9)
cbar = plt.colorbar(im)
cbar.set_label("Proportion of true class (row-normalised)", fontsize=9)
plt.title(f"{DISPLAY} confusion matrix on test set", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %% --------------------------------------------------------------------
# 9. Grad-CAM — target layer + reshape depend on architecture
if MODEL_NAME.startswith("vit"):
    def reshape_transform(tensor, h=14, w=14):
        return tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(-1)) \
                               .permute(0, 3, 1, 2)
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)
elif MODEL_NAME.startswith("resnet"):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
else:
    raise ValueError(f"Grad-CAM target layer not configured for {MODEL_NAME}")

def denorm(x_224):
    arr = x_224.cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * IMAGENET_STD + IMAGENET_MEAN, 0, 1).astype(np.float32)

def heatmap_for(i):
    # .contiguous() is needed for ResNet on MPS — pytorch-grad-cam's internal
    # view() on activations fails when strides aren't compact.
    x = resize224(X_te_t[i:i+1].to(device)).contiguous()
    return cam(input_tensor=x, targets=None)[0]  # (224,224) in [0,1]

# Sample panel: 4 correct + 4 wrong
try:
    rng = np.random.default_rng(SEED)
    correct = np.where(te_pred == y_te)[0]
    wrong   = np.where(te_pred != y_te)[0]
    pick_c  = rng.choice(correct, size=min(4, len(correct)), replace=False)
    pick_w  = rng.choice(wrong,   size=min(4, len(wrong)),   replace=False)
    samples = np.concatenate([pick_c, pick_w])

    fig, axes = plt.subplots(2, len(samples), figsize=(2*len(samples), 4.5))
    for col, i in enumerate(samples):
        img = denorm(resize224(X_te_t[i:i+1])[0])
        h = heatmap_for(i)
        ok = te_pred[i] == y_te[i]
        axes[0, col].imshow(img); axes[0, col].axis("off")
        axes[0, col].set_title(
            f"true: {classes[y_te[i]][:15]}\npred: {classes[te_pred[i]][:15]}\n{'OK' if ok else 'WRONG'}",
            fontsize=7)
        axes[1, col].imshow(img); axes[1, col].imshow(h, cmap="jet", alpha=0.45)
        axes[1, col].axis("off")
    plt.suptitle(f"{MODEL_NAME} Grad-CAM — 4 correct + 4 wrong"); plt.tight_layout()
    plt.savefig(FIG_DIR / "gradcam_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"WARNING: gradcam_samples failed: {type(e).__name__}: {e}")
    print("model + metrics already saved; rerun gradcam separately.")

# %% --------------------------------------------------------------------
# 10. Per-cluster AVERAGE Grad-CAM — the tutor's specific ask
#  For each class, average the heatmaps over correctly-classified test images.
AVG_CAP = 50  # cap per class for speed

try:
    avg = {}
    cls_bar = tqdm(list(enumerate(classes)), desc="avg Grad-CAM per class",
                   mininterval=2.0, ascii=True, dynamic_ncols=True)
    for label, cls in cls_bar:
        idx = np.where((te_pred == y_te) & (y_te == label))[0]
        if len(idx) == 0:
            continue
        if len(idx) > AVG_CAP:
            idx = idx[:AVG_CAP]
        hs = [heatmap_for(i) for i in idx]
        avg[cls] = np.mean(hs, axis=0)
        cls_bar.set_postfix(cls=cls[:20], n=len(idx))

    cols = 4
    rows = (len(avg) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if rows > 1 else axes if hasattr(axes, "__len__") else [axes]
    for ax, (cls, h) in zip(axes, avg.items()):
        ax.imshow(h, cmap="jet", vmin=0, vmax=1)
        ax.set_title(cls, fontsize=9); ax.axis("off")
    for ax in axes[len(avg):]: ax.axis("off")
    plt.suptitle("Average Grad-CAM per cluster (correctly-classified test images)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "avg_heatmaps_per_cluster.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"WARNING: avg_heatmaps_per_cluster failed: {type(e).__name__}: {e}")
    print("model + metrics already saved; rerun gradcam separately.")

# %% --------------------------------------------------------------------
# 11. Done
# (model + metrics + history were saved in section 8 before Grad-CAM, so this
# section just confirms.)
print(f"\nSaved figures + metrics + model to {FIG_DIR}")
