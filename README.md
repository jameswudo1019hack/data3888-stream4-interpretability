# Same patch. Same label. Same reasoning?

**DATA3888 — Stream 4 (Biotechnology: Image Analysis)**

Do a CNN (ResNet-50) and a Vision Transformer (ViT-small) that reach the same
accuracy on breast-cancer H&E cell classification rely on the *same* parts of
the image? We fine-tune both architectures on three cell classes (Tumour /
Immune / Stromal), then probe *where* each model looks with a controlled
centre-vs-edge occlusion study and Grad-CAM. Headline finding: ResNet has a
sharp **centre cliff** (filling 5% of the patch centre with in-distribution
"H&E-pink" drops accuracy 0.69 → 0.51, chance by 30%), while ViT degrades only
gradually (still 0.52 at 70% centre fill).

**Authors:** Liam Nguyen, James Wu, Hossion Ali, Joy Luo, Ewan Yuan.

---

## 1. Setup

```bash
# Python 3.11 recommended
python -m venv .venv && source .venv/bin/activate
pip install torch timm pytorch-grad-cam scikit-learn numpy pandas matplotlib pillow tqdm shiny
```

GPU optional. Training runs on CUDA, Apple MPS, or CPU (auto-detected); each
seed takes ~5–10 min on a Colab GPU.

> **Just want to reproduce the report's figures and numbers?** See
> [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Tier 0 regenerates every figure
> and table from cached results in ~10 s — no data download, no GPU, no
> retraining. Exact package versions are pinned in
> [`requirements.txt`](requirements.txt).

## 2. Data

The image bundle (per-class H&E crops) comes from the 10x Genomics Xenium
breast-cancer dataset (Janesick et al. 2023), supplied as per-class `.zip`
files.

```bash
# put the per-class zips in Images/, then extract to data/100/<ClassName>/
python scripts/unzip_all.py
```

Notes (see `obsidian/Data/Dataset Overview.md`):
- Download the per-class zips **individually** — bulk-downloading the folder
  corrupts the archives.
- `unzip_all.py` skips the spurious `*_10.png` files automatically.
- To regenerate the crops from the raw whole-slide image instead, see
  `metadata/register_and_export_images.R` (registers the post-Xenium H&E to the
  Xenium cell boundaries via a 3-point affine transform, then exports each cell
  as its boundary bounding box + a context margin).

## 3. Reproduce the study

```bash
# 1. train both architectures across 5 seeds
#    -> figures/v5_grouped/<model>/seed_<n>/
for M in resnet50 vit_small_patch16_224; do
  for S in 42 43 44 45 46; do
    MODEL=$M SEED=$S python scripts/stream4_pipeline.py
  done
done

# 2. centre-vs-edge H&E-pink occlusion study (no retraining)
#    -> figures/v5_grouped/noise_v2/
python scripts/noise_eval_v2.py

# 3. aggregate per-seed metrics (mean +/- std)
python scripts/aggregate_seeds.py
```

All randomness (sampling, train/val/test split, weight init) is seeded, so a
given seed is fully reproducible. The **only** path a new user must change is
the data directory (`DATA_DIR` / `IMAGE_FOLDER` in the scripts).

## 4. Interactive demo (the deployed product)

```bash
pip install -r app/requirements.txt
shiny run app/app.py     # then open http://127.0.0.1:8000
```

Upload an H&E patch (or pick a curated demo), and compare ResNet vs ViT
predictions, softmax probabilities and Grad-CAM heatmaps side by side, with
live blur and centre/edge-fill controls. See `app/README.md`.

## 5. Repository map

| Path | What |
|---|---|
| `scripts/stream4_pipeline.py` | Train + evaluate one model/seed; saves checkpoint, metrics, confusion matrix, Grad-CAM |
| `scripts/noise_eval_v2.py` | Centre-vs-edge H&E-pink occlusion sweep (Figures 2–3) |
| `scripts/aggregate_seeds.py` | Mean ± std across seeds |
| `james_preprocessing/` | Shared, deterministic data loading + 70/10/20 split |
| `app/` | Shiny-for-Python Grad-CAM Explorer (`app.py`, `inference.py`, demo images) |
| `metadata/` | Cell boundaries, cluster metadata, and the R registration/export script |
| `figures/v5_grouped/` | Trained checkpoints, per-seed metrics, and all figures |

## 6. Configuration (team-shared defaults)

```
SEED          = 42        # 42–46 used for the 5-seed runs
MAX_PER_CLASS = 1000      # balanced sample per class, per seed
LOAD_SIZE     = 100       # crop side in px (resized to 224 for the model)
split         = 70 / 10 / 20  (train / val / test), stratified
optimiser     = AdamW (lr 3e-5, weight_decay 0.01), batch 32
epochs        = 10 max, early stopping on val accuracy (patience 3)
```

## Acknowledgement

An AI assistant (Anthropic Claude) was used for code debugging, drafting and
copy-editing. All experimental design, modelling choices, results and their
interpretation were produced and verified by the authors.
