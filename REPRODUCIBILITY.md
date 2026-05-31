# Reproducibility guide

Everything in the report can be reproduced from this repository. There are
three tiers depending on how much you want to recompute — **Tier 0 needs no
data download, no GPU, and runs in seconds**, and regenerates every figure and
number in the report from cached intermediate results.

```bash
# one-time setup (Python 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All randomness (class sampling, the 70/10/20 split, weight init) is seeded, so
a given seed is bit-for-bit reproducible.

---

## Tier 0 — regenerate all figures & numbers from cached results (~10 s, no data, no GPU)

The repo ships the cached analysis artifacts (`results.csv`, per-seed
prediction arrays, per-seed metrics), so the entire results section of the
report can be rebuilt without the raw images or the trained models.

```bash
# Headline centre-vs-edge figures (Fig. of the mask study) from cached results.csv
python scripts/replot_noise_v2.py
#   -> figures/v5_grouped/noise_v2/{combined,per_class,overall}.png

# Per-seed → mean ± std summary tables (the baseline-performance numbers)
python scripts/aggregate_seeds.py --version v5_grouped --model resnet50
python scripts/aggregate_seeds.py --version v5_grouped --model vit_small_patch16_224
```

Cached inputs these read from (all committed):
- `figures/v5_grouped/noise_v2/results.csv` — the full 200-row mask sweep
  (2 models × 2 mask types × 10 area fractions × 5 seeds).
- `figures/v5_grouped/<model>/seed_<n>/test_pred.npy`, `test_y.npy`,
  `test_proba.npy`, `test_metrics.csv` — per-seed test-set outputs used for the
  confusion matrices and per-class F1.

**This is the fastest way to verify the report's results came from this code.**

---

## Tier 1 — run the interactive Shiny demo (~30 s, no data, no GPU)

The two **seed-42** checkpoints the app loads (ResNet-50 and ViT-small) **are
committed** to the repo, so the demo runs out-of-the-box — no data download and
no training needed:

```bash
shiny run app/app.py    # open http://127.0.0.1:8000
```

Upload an H&E patch (or pick a curated demo) and compare both models'
predictions, softmax probabilities and Grad-CAM heatmaps, with live blur and
centre/edge-fill controls.

The other 8 checkpoints (seeds 43–46 × 2 architectures) are *not* committed
(~90 MB each); regenerate them via Tier 2 if you want the full multi-seed set.

---

## Tier 2 — full reproduction from raw data (downloads + training, hours)

### 1. Get the data

The imaging data is the 10x Genomics Xenium breast-cancer dataset
(Janesick et al. 2023), GEO accession
[**GSM7780153**](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7780153).
It is **not** redistributed here (~28 GB, and it has its own usage terms).

- Download the per-class patch zips into `Images/`, then extract:
  ```bash
  python scripts/unzip_all.py      # -> data/100/<ClassName>/
  ```
- Download the per-class zips **individually** — bulk-downloading the parent
  folder corrupts the archives. `unzip_all.py` skips the spurious `*_10.png`
  files automatically.
- To regenerate crops from the raw whole-slide image instead, see
  `metadata/register_and_export_images.R`.

### 2. Train all 10 runs (2 architectures × 5 seeds)

```bash
for M in resnet50 vit_small_patch16_224; do
  for S in 42 43 44 45 46; do
    MODEL=$M SEED=$S python scripts/stream4_pipeline.py
  done
done
#   -> figures/v5_grouped/<model>/seed_<n>/{checkpoint, metrics, confusion, Grad-CAM}
```

### 3. Re-run the mask study and aggregate

```bash
python scripts/noise_eval_v2.py     # regenerates results.csv + figures
python scripts/aggregate_seeds.py --version v5_grouped --model resnet50
python scripts/aggregate_seeds.py --version v5_grouped --model vit_small_patch16_224
```

This reproduces `figures/v5_grouped/noise_v2/results.csv` from scratch, which
Tier 0 then plots — closing the loop.

---

## Environment

- Python **3.11**; exact package versions pinned in `requirements.txt`.
- Device auto-detected: CUDA → Apple MPS → CPU. No GPU required for Tier 0.
- Config defaults (overridable via env vars, see `scripts/stream4_pipeline.py`):
  `SEED=42`, `MAX_PER_CLASS=1000`, `LOAD_SIZE=100`, split 70/10/20 stratified,
  AdamW (lr 3e-5, wd 0.01), batch 32, ≤10 epochs, early stopping (patience 3).
