"""Aggregate per-seed test_metrics.csv files into mean ± std summary.

Reads:  figures/<VERSION>/<MODEL>/seed_*/test_metrics.csv
Writes: figures/<VERSION>/<MODEL>/summary.csv          (mean per metric per class)
        figures/<VERSION>/<MODEL>/summary_std.csv      (std per metric per class)
        figures/<VERSION>/<MODEL>/summary.png          (per-class F1 with error bars)

Usage:
  python scripts/aggregate_seeds.py --version v5_grouped --model resnet50
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--version", required=True)
ap.add_argument("--model",   required=True)
args = ap.parse_args()

base = ROOT / "figures" / args.version / args.model
seeds = sorted(d for d in base.glob("seed_*") if d.is_dir())
if not seeds:
    raise SystemExit(f"no seed_* dirs under {base}")

print(f"found {len(seeds)} seeds: {[s.name for s in seeds]}")

frames = []
for s in seeds:
    csv = s / "test_metrics.csv"
    if not csv.exists():
        print(f"  skipping {s.name} (no test_metrics.csv)")
        continue
    df = pd.read_csv(csv, index_col=0)
    df["seed"] = s.name
    frames.append(df)

all_df = pd.concat(frames)
mean_df = all_df.drop(columns="seed").groupby(level=0).mean()
std_df  = all_df.drop(columns="seed").groupby(level=0).std()

mean_df.to_csv(base / "summary.csv")
std_df.to_csv(base / "summary_std.csv")
print(f"\nMean across {len(seeds)} seeds:")
print(mean_df.round(3))
print(f"\nStd:")
print(std_df.round(3))

# Per-class F1 bar chart with error bars
class_rows = [r for r in mean_df.index
              if r not in ("accuracy", "macro avg", "weighted avg")]
fig, ax = plt.subplots(figsize=(1.5*len(class_rows)+2, 4))
xs = np.arange(len(class_rows))
ax.bar(xs, mean_df.loc[class_rows, "f1-score"],
       yerr=std_df.loc[class_rows, "f1-score"],
       capsize=4, color="#4477AA", alpha=0.85)
ax.set_xticks(xs); ax.set_xticklabels(class_rows, rotation=20, ha="right")
ax.set_ylabel("F1-score (mean ± std)")
ax.set_title(f"{args.version} / {args.model} — F1 across {len(seeds)} seeds")
ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
for x, m, s in zip(xs, mean_df.loc[class_rows, "f1-score"],
                    std_df.loc[class_rows, "f1-score"]):
    ax.text(x, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(base / "summary.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {base/'summary.csv'} + summary_std.csv + summary.png")
