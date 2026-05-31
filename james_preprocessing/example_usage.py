"""Minimal example — load the unified split, print shapes, sanity-check labels.

Run after extracting the per-class zips:
    python unzip_all.py
    python example_usage.py
"""
import numpy as np
from preprocess import load_dataset, CLASSES

X_tr, X_va, X_te, y_tr, y_va, y_te, classes = load_dataset(data_dir="data/100")

print(f"classes (label → name): {dict(enumerate(classes))}")
print(f"train: {X_tr.shape}  uint8  balance {np.bincount(y_tr)}")
print(f"val:   {X_va.shape}  uint8  balance {np.bincount(y_va)}")
print(f"test:  {X_te.shape}  uint8  balance {np.bincount(y_te)}")

# Quick sanity: every set should be balanced 700 / 100 / 200 per class
# (with the default MAX_PER_CLASS=1000 and 70/10/20 stratified split).

# Now hand off to your model. Examples:
#
# --- PyTorch ImageNet-pretrained ---
# from preprocess import to_imagenet_normalised_torch
# X_tr_tensor = to_imagenet_normalised_torch(X_tr)   # (N, 3, H, W) float32
#
# --- TensorFlow / Keras ---
# X_tr_tf = X_tr.astype("float32") / 255.0
#
# --- scikit-learn / flat features ---
# X_tr_flat = X_tr.reshape(len(X_tr), -1).astype("float32") / 255.0
