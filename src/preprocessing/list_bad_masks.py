# src/preprocessing/list_bad_masks.py
import cv2, os
from pathlib import Path
import numpy as np

mask_dir = Path("data/train/masks")
out_file = Path("results/bad_masks_report.txt")
os.makedirs("results", exist_ok=True)

rows = []
for p in sorted(mask_dir.iterdir()):
    if p.suffix.lower() not in (".png",".jpg",".jpeg"): continue
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None: continue
    frac = (m>127).mean()
    rows.append((p.name, float(frac)))

rows = sorted(rows, key=lambda x: x[1])
with open(out_file, "w") as f:
    f.write("filename,coverage_fraction\n")
    for name, frac in rows:
        f.write(f"{name},{frac:.6f}\n")

print("Report written to", out_file)
print("Lowest 10 masks (very small coverage):")
for name, frac in rows[:10]:
    print(name, frac)
print("Highest 10 masks (very large coverage):")
for name, frac in rows[-10:]:
    print(name, frac)
