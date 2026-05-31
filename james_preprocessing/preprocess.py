"""Unified preprocessing for the DATA3888 Stream 4 group.

Loads the H&E crops from per-class folders, applies the agreed 3-class
grouping (Tumour / Immune / Stromal), balanced sampling, and a deterministic
70 / 10 / 20 stratified train / val / test split.

Use this so everyone trains on identical splits — same seed → same images
in each set across team members.

Library use (returns numpy arrays, framework-agnostic):
    from preprocess import load_dataset
    X_tr, X_va, X_te, y_tr, y_va, y_te, classes = load_dataset()

CLI use (saves to .npz):
    python preprocess.py --data_dir data/100 --output data_split_100.npz
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# --- agreed configuration (team-shared, do not change without consensus) ---
SEED          = 42
MAX_PER_CLASS = 1000   # per group, post-grouping, balanced
LOAD_SIZE     = 100    # set to 50 when pointing at data/50/

# 3-class biological grouping (Week 8 tutor directive, 2026-04-20):
#   Drop the "Other" class — too noisy.
#   Myoepi and Hybrid folders excluded — out of scope.
GROUPS = {
    "Tumour":  ["DCIS_1", "DCIS_2", "Invasive_Tumor", "Prolif_Invasive_Tumor"],
    "Immune":  ["B_Cells", "CD4+_T_Cells", "CD8+_T_Cells",
                "Macrophages_1", "Macrophages_2",
                "LAMP3+_DCs", "IRF7+_DCs", "Mast_Cells"],
    "Stromal": ["Stromal", "Perivascular-Like", "Endothelial"],
}
CLASSES = sorted(GROUPS.keys())   # [Immune, Stromal, Tumour] — labels 0, 1, 2

TEST_FRAC = 0.20
VAL_FRAC  = 0.125   # of remainder; gives ~10% of total
# ---------------------------------------------------------------------------


def _load_images(data_dir, max_per_class, size, seed):
    """Per-group balanced sampling from per-class subfolders."""
    X, y = [], []
    rng = np.random.default_rng(seed)
    for label, cls in enumerate(CLASSES):
        all_paths = []
        for src in GROUPS[cls]:
            all_paths.extend(sorted((data_dir / src).glob("*.png")))
        if len(all_paths) > max_per_class:
            pick = rng.choice(len(all_paths), max_per_class, replace=False)
            all_paths = [all_paths[i] for i in pick]
        for p in all_paths:
            img = Image.open(p).convert("RGB").resize((size, size))
            X.append(np.asarray(img, dtype=np.uint8))
            y.append(label)
    return np.stack(X), np.array(y)


def load_dataset(data_dir="data/100", seed=SEED,
                 max_per_class=MAX_PER_CLASS, load_size=LOAD_SIZE):
    """Return (X_tr, X_va, X_te, y_tr, y_va, y_te, classes).

    Args:
        data_dir: path to extracted per-class folder (e.g. "data/100").
        seed: random seed for sampling and split. Default 42.
        max_per_class: per-group cap (post-grouping). Default 1000.
        load_size: image side in px. 100 for `data/100/`, 50 for `data/50/`.

    Returns:
        X_tr, X_va, X_te: uint8 ndarrays of shape (N, load_size, load_size, 3).
        y_tr, y_va, y_te: int ndarrays of shape (N,), values in 0..2.
        classes: list[str] — index → class name (alphabetical).
                 0 = Immune, 1 = Stromal, 2 = Tumour.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"data_dir {data_dir.resolve()} does not exist.\n"
            "Run unzip_all.py first to extract the per-class zips, e.g.:\n"
            "  python unzip_all.py --src Images --dst data/100"
        )

    X, y = _load_images(data_dir, max_per_class, load_size, seed)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=VAL_FRAC, stratify=y_tr, random_state=seed)

    return X_tr, X_va, X_te, y_tr, y_va, y_te, CLASSES


# Optional helpers — model-specific, NOT part of the agreed unified
# preprocessing. Use them if it's convenient; skip if your model expects
# different normalization / shape.
def to_imagenet_normalised_torch(X_uint8):
    """Convert uint8 (N,H,W,3) -> float32 torch (N,3,H,W), ImageNet-normalised.

    Only useful if you're using ImageNet-pretrained PyTorch models.
    """
    import torch
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = X_uint8.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/100",
                    help="extracted per-class folder (data/100 or data/50)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max_per_class", type=int, default=MAX_PER_CLASS)
    ap.add_argument("--load_size", type=int, default=LOAD_SIZE,
                    help="image side; match folder (100 or 50)")
    ap.add_argument("--output", default="data_split.npz",
                    help="output npz path")
    args = ap.parse_args()

    print(f"loading from {args.data_dir} ...")
    X_tr, X_va, X_te, y_tr, y_va, y_te, classes = load_dataset(
        data_dir=args.data_dir, seed=args.seed,
        max_per_class=args.max_per_class, load_size=args.load_size)

    print(f"\nclasses (label → name): {dict(enumerate(classes))}")
    print(f"train: {X_tr.shape}  balance {np.bincount(y_tr)}")
    print(f"val:   {X_va.shape}  balance {np.bincount(y_va)}")
    print(f"test:  {X_te.shape}  balance {np.bincount(y_te)}")

    np.savez_compressed(args.output,
                        X_tr=X_tr, y_tr=y_tr,
                        X_va=X_va, y_va=y_va,
                        X_te=X_te, y_te=y_te,
                        classes=np.array(classes))
    print(f"\nsaved {args.output}")
    print("load it later with:")
    print("    d = np.load('data_split.npz', allow_pickle=True)")
    print("    X_tr, y_tr = d['X_tr'], d['y_tr']")


if __name__ == "__main__":
    main()
