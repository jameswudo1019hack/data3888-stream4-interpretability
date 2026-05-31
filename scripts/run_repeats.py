"""Run scripts/stream4_pipeline.py multiple times with different seeds.

Each seed produces its own subdir:
  figures/<VERSION>/<MODEL>/seed_<SEED>/

Usage:
  python scripts/run_repeats.py                                # ResNet50 on 100/, seeds 42-46
  python scripts/run_repeats.py --model vit_small_patch16_224  # ViT instead
  python scripts/run_repeats.py --image_folder 50              # cell-only
  python scripts/run_repeats.py --seeds 42 43 44 45 46         # explicit list
  python scripts/run_repeats.py --version v5_grouped --image_folder 50 --seeds 42 43 44 45 46
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "scripts" / "stream4_pipeline.py"
LOG_DIR  = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--version", default="v5_grouped")
ap.add_argument("--model", default="resnet50",
                choices=["resnet50", "vit_small_patch16_224"])
ap.add_argument("--image_folder", default="100", choices=["50", "100"])
ap.add_argument("--max_per_class", default="1000")
ap.add_argument("--seeds", nargs="+", default=["42", "43", "44", "45", "46"])
args = ap.parse_args()

print(f"version={args.version}  model={args.model}  img={args.image_folder}")
print(f"seeds: {args.seeds}\n")

py = "/opt/anaconda3/bin/python"
total_t0 = time.time()
for seed in args.seeds:
    log_path = LOG_DIR / f"{args.version}_{args.model}_img{args.image_folder}_seed{seed}.log"
    print(f"[seed {seed}] -> {log_path.name}")
    env = {**os.environ,
           "SEED":          seed,
           "VERSION":       args.version,
           "MODEL":         args.model,
           "IMAGE_FOLDER":  args.image_folder,
           "MAX_PER_CLASS": args.max_per_class}
    t0 = time.time()
    with open(log_path, "w") as f:
        rc = subprocess.run([py, "-u", str(PIPELINE)],
                            env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[seed {seed}] exit={rc}  ({time.time()-t0:.0f}s)")
    if rc != 0:
        print(f"  see {log_path} for traceback. Continuing to next seed.")

print(f"\nALL DONE  total {time.time()-total_t0:.0f}s")
print(f"results in figures/{args.version}/{args.model}/seed_*/")
print(f"aggregate with: python scripts/aggregate_seeds.py "
      f"--version {args.version} --model {args.model}")
