# Demo image catalogue

Each image is from the held-out v5_grouped seed-42 dataset. The model predictions and confidences below are what the app will show.

## Easy — both models high-confidence correct

| File | Source folder | True | ResNet pred / conf | ViT pred / conf |
|---|---|---|---|---|
| `easy/01_tum_DCIS_1.png` | DCIS_1 | Tumour | Tumour / 0.81 | Tumour / 0.90 |
| `easy/02_tum_DCIS_2.png` | DCIS_2 | Tumour | Tumour / 0.81 | Tumour / 0.92 |
| `easy/03_tum_Invasive_Tumor.png` | Invasive_Tumor | Tumour | Tumour / 0.89 | Tumour / 0.88 |
| `easy/04_imm_B_Cells.png` | B_Cells | Immune | Immune / 0.83 | Immune / 0.95 |
| `easy/05_imm_CD4plus_T_Cells.png` | CD4+_T_Cells | Immune | Immune / 0.84 | Immune / 0.96 |
| `easy/06_imm_Macrophages_2.png` | Macrophages_2 | Immune | Immune / 0.83 | Immune / 0.92 |
| `easy/07_str_Stromal.png` | Stromal | Stromal | Stromal / 0.69 | Stromal / 0.86 |
| `easy/08_str_Perivascular-Like.png` | Perivascular-Like | Stromal | Stromal / 0.60 | Stromal / 0.88 |
| `easy/09_str_Endothelial.png` | Endothelial | Stromal | Stromal / 0.62 | Stromal / 0.78 |

## Disagreement — models predict different classes

| File | Source folder | True | ResNet pred / conf | ViT pred / conf |
|---|---|---|---|---|
| `disagreement/01_imm_R-Str_V-Imm.png` | B_Cells | Immune | Stromal / 0.48 | Immune / 0.87 |
| `disagreement/02_tum_R-Tum_V-Str.png` | DCIS_2 | Tumour | Tumour / 0.45 | Stromal / 0.62 |
| `disagreement/03_tum_R-Imm_V-Str.png` | DCIS_2 | Tumour | Immune / 0.44 | Stromal / 0.57 |

## Low confidence — both models uncertain

| File | Source folder | True | ResNet pred / conf | ViT pred / conf |
|---|---|---|---|---|
| `low_confidence/01_imm_uncertain.png` | Macrophages_1 | Immune | Stromal / 0.37 | Immune / 0.34 |
| `low_confidence/02_imm_uncertain.png` | Macrophages_1 | Immune | Tumour / 0.39 | Stromal / 0.38 |
| `low_confidence/03_str_uncertain.png` | Perivascular-Like | Stromal | Stromal / 0.42 | Tumour / 0.40 |