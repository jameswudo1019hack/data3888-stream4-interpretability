"""Unzip per-class zips, filtering the spurious *_10.png files (Ed #254).
Safe to re-run — already-extracted files are skipped.

Usage:
  python scripts/unzip_all.py                    # default: Images/  -> data/100/
  python scripts/unzip_all.py --size 50          # default 50 paths: Images_50/ -> data/50/
  python scripts/unzip_all.py --src X --dst Y    # custom paths
"""
import argparse, sys
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent  # project root, one above scripts/

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=["50", "100"], default="100",
                help="image size — sets default --src and --dst")
ap.add_argument("--src", help="source folder containing per-class .zip files")
ap.add_argument("--dst", help="destination folder for extracted images")
args = ap.parse_args()

SRC = Path(args.src) if args.src else ROOT / ("Images_50" if args.size == "50" else "Images")
DST = Path(args.dst) if args.dst else ROOT / "data" / args.size
DST.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    sys.exit(f"source folder not found: {SRC}")

print(f"src: {SRC}\ndst: {DST}\n")

skipped_10 = 0
corrupt = []
summary = []

for zpath in sorted(SRC.glob("*.zip")):
    if zpath.stat().st_size == 0:
        corrupt.append((zpath.name, "0 bytes"))
        continue
    try:
        with zipfile.ZipFile(zpath) as z:
            members = [m for m in z.namelist() if m.endswith(".png")]
            keep = [m for m in members if not m.endswith("_10.png")]
            skipped_10 += len(members) - len(keep)
            for m in keep:
                out = DST / m
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as s, open(out, "wb") as d:
                    d.write(s.read())
            class_name = members[0].split("/")[0] if members else zpath.stem
            summary.append((class_name, len(keep)))
    except zipfile.BadZipFile:
        corrupt.append((zpath.name, "bad zip"))

print(f"\nExtracted to {DST}\n")
print(f"{'Class':<32} {'Images':>8}")
print("-" * 42)
total = 0
for name, n in sorted(summary):
    print(f"{name:<32} {n:>8}")
    total += n
print("-" * 42)
print(f"{'TOTAL':<32} {total:>8}")
print(f"\n_10.png files filtered: {skipped_10}")
if corrupt:
    print("\nCorrupt / empty (re-download these):")
    for name, why in corrupt:
        print(f"  {name}: {why}")
