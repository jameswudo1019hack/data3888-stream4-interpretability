# Stream 4 — Interpretability Demo (Shiny for Python)

Two-tab Shiny app:

- **Demo** — upload an H&E cell patch (or pick one of the curated demo
  images) and see how **ResNet-50** and **ViT-small** classify it
  side-by-side, with Grad-CAM heatmaps showing *where each model looked*.
- **Analysis** — the v6.1 centre-vs-edge mask study results, per-class
  breakdown, baseline confusion matrices and methodology, embedded straight
  from `figures/v5_grouped/`.

## What this demonstrates

The two architectures behave very differently on the same image:

- **ResNet-50** places a sharp red Grad-CAM blob on the centre cell.
- **ViT-small** spreads attention across the whole patch.

This matches the finding from our v6.1 centre-vs-edge mask study: filling 5 %
of the centre with H&E-pink collapses ResNet's Immune-class accuracy from
0.64 → ~0, while ViT is largely unaffected.

## Run it

From the **project root** (`InterProject/`):

```bash
pip install -r app/requirements.txt
shiny run --reload app/app.py
```

Then open <http://127.0.0.1:8000>.

## What's in here

| File / folder | Purpose |
|---|---|
| `app.py` | Shiny UI + server (flatly theme, navbar, Demo + Analysis tabs) |
| `inference.py` | Loads both seed-42 checkpoints, runs predict + Grad-CAM |
| `assets/usyd_logo.svg` | USyd logo shown in the navbar (served at `/assets/`) |
| `demo_images/` | 15 curated demo patches in three categories |
| `demo_images/INDEX.md` | What each demo image is, and the predicted classes |
| `requirements.txt` | Python deps |

The app also mounts the project's `figures/` directory at `/figures/` so the
Analysis tab can embed the existing v6.1 plots (`combined.png`,
`per_class.png`, `fill_examples.png`) and per-seed confusion matrices
without copying files.

## Demo image categories

- **`demo_images/easy/`** — 9 images (3 per class), both models high-confidence correct. Good for the "look how it works" intro.
- **`demo_images/disagreement/`** — 3 images where ResNet and ViT predict different classes. Motivates having both architectures.
- **`demo_images/low_confidence/`** — 3 images where both models are uncertain. Honest about model limits.

## Configuration

The app uses the **seed-42** checkpoints (one per architecture) from:

```
figures/v5_grouped/resnet50/seed_42/resnet50_stream4.pt
figures/v5_grouped/vit_small_patch16_224/seed_42/vit_small_patch16_224_stream4.pt
```

To swap to a different seed, change `SEED = 42` at the top of `inference.py`.

## Preprocessing

Exactly matches training (`scripts/stream4_pipeline.py`):

1. Convert to RGB
2. Resize to 100×100 (the native patch size used at training load)
3. Divide by 255, ImageNet mean/std normalise
4. Bilinear interpolate to 224×224 (what the model actually sees)

Grad-CAM is computed at:
- ResNet-50: last residual block (`layer4[-1]`)
- ViT-small: norm before the last attention block (`blocks[-1].norm1`)
