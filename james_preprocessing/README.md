# Stream 4 — Unified Preprocessing

Author: James (DATA3888 group)
Last updated: 2026-05-04

This bundle gives the whole group an identical view of the dataset:
**same classes, same sample sizes, same train / val / test splits.**
Pass `seed=42` to `load_dataset()` and any teammate gets exactly the same
images in each set.

## What's in here

| File | Purpose |
|---|---|
| `preprocess.py` | The library. `load_dataset()` returns numpy arrays + the agreed split. Run as a script to save a `.npz`. |
| `unzip_all.py` | Extracts per-class `.zip` files into `data/100/<class>/...` and skips spurious `*_10.png` (Ed #254). Idempotent. |
| `example_usage.py` | Tiny demo showing how to call `load_dataset()` and hand off to PyTorch / TF / sklearn. |

## Agreed configuration

These constants live in `preprocess.py`. Don't change them locally — change them as a group.

```python
SEED          = 42
MAX_PER_CLASS = 1000   # per group, balanced after grouping
LOAD_SIZE     = 100    # set 50 if loading from data/50/
TEST_FRAC     = 0.20
VAL_FRAC      = 0.125  # of remainder ≈ 10% of total
```

3-class biological grouping (per Week 8 tutor directive):

```python
GROUPS = {
    "Tumour":  ["DCIS_1", "DCIS_2", "Invasive_Tumor", "Prolif_Invasive_Tumor"],
    "Immune":  ["B_Cells", "CD4+_T_Cells", "CD8+_T_Cells",
                "Macrophages_1", "Macrophages_2",
                "LAMP3+_DCs", "IRF7+_DCs", "Mast_Cells"],
    "Stromal": ["Stromal", "Perivascular-Like", "Endothelial"],
}
```

After alphabetical sort, label IDs are: **0 = Immune, 1 = Stromal, 2 = Tumour.**

## Setup (once per machine)

1. Create the `data/100/` layout from the OneDrive zips.

   - Download the per-cluster `.zip` files from OneDrive `100/`. Don't bulk-download the whole folder — it corrupts (Ed #254).
   - Drop them all into a folder called `Images/` at the project root.
   - Run:

     ```bash
     python unzip_all.py
     ```

   - Result: `data/100/<ClassName>/cell_*.png`.
   - For the 50/ context experiment, repeat with the 50/ zips into `Images_50/`, then `python unzip_all.py --size 50`.

2. Install dependencies (Python ≥ 3.9):

   ```bash
   pip install numpy pillow scikit-learn
   ```

   (Optional: `torch` if you want the `to_imagenet_normalised_torch` helper.)

## Usage — as a library

```python
from preprocess import load_dataset

X_tr, X_va, X_te, y_tr, y_va, y_te, classes = load_dataset(data_dir="data/100")
# X_*: uint8 numpy arrays, shape (N, 100, 100, 3)
# y_*: int numpy arrays, shape (N,), values in {0, 1, 2}
# classes: ['Immune', 'Stromal', 'Tumour']
```

With the defaults you'll get:

```
train: (2100, 100, 100, 3)  balance [700 700 700]
val:   ( 300, 100, 100, 3)  balance [100 100 100]
test:  ( 600, 100, 100, 3)  balance [200 200 200]
```

## Usage — as a script

Saves the split to a single compressed `.npz` so you don't reload PNGs every run:

```bash
python preprocess.py --output data_split_100.npz
# then in your own code:
import numpy as np
d = np.load("data_split_100.npz", allow_pickle=True)
X_tr, y_tr = d['X_tr'], d['y_tr']
```

## Hand-off to your model

The arrays are plain `uint8` images — you decide what normalization, tensor library, and resize to use. A few common patterns are sketched at the bottom of `example_usage.py`:

- **PyTorch + ImageNet-pretrained backbone:** call `to_imagenet_normalised_torch(X_tr)` from `preprocess.py`. It converts to `(N, 3, H, W)` float32 with ImageNet mean/std.
- **TensorFlow / Keras:** `X_tr.astype("float32") / 255.0` is enough for many models. ImageNet stats apply if you're using a Keras Applications model.
- **scikit-learn baselines (logreg, etc.):** flatten with `X_tr.reshape(len(X_tr), -1).astype("float32") / 255.0`.

## Why the split is reproducible

Both the per-group subsampling and the stratified train/val/test split take `seed=42`. As long as everyone uses the same `seed`, `MAX_PER_CLASS`, and `data_dir`, we all train and test on the **exact same image sets** — so model comparisons across team members are clean.
