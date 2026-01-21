import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
TRAIN_DIR = "data/train"
OUT_DIR = "data/inspect_no_change"
COVERAGE_THRESHOLD = 0.01  # 1%

MASK_DIR = os.path.join(TRAIN_DIR, "masks")
BEFORE_DIR = os.path.join(TRAIN_DIR, "before")
AFTER_DIR = os.path.join(TRAIN_DIR, "after")

# Creează folderele de output
for sub in ["before", "after", "masks"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

def mask_coverage(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return np.count_nonzero(mask) / mask.size

files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

selected = 0

print(f"🔍 Analizez {len(files)} măști din train...")

for fname in tqdm(files):
    mask_path = os.path.join(MASK_DIR, fname)
    cov = mask_coverage(mask_path)

    if cov is None:
        continue

    if cov < COVERAGE_THRESHOLD:
        # copiem mask + before + after
        shutil.copy(mask_path, os.path.join(OUT_DIR, "masks", fname))
        shutil.copy(os.path.join(BEFORE_DIR, fname), os.path.join(OUT_DIR, "before", fname))
        shutil.copy(os.path.join(AFTER_DIR, fname), os.path.join(OUT_DIR, "after", fname))
        selected += 1

print("\n✅ Gata!")
print(f"📦 Copiate {selected} samples cu mask coverage < {COVERAGE_THRESHOLD*100:.1f}%")
print(f"📁 Verifică vizual în: {OUT_DIR}")
