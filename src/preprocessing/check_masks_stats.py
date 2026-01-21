# src/preprocessing/check_masks_stats.py
import numpy as np
import cv2
import os
from pathlib import Path

def main(mask_folder="data/train/masks"):
    mask_dir = Path(mask_folder)
    files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")])
    areas = []
    missing = 0
    for p in files:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            missing += 1
            continue
        m_bin = (m > 127).astype(np.uint8)
        areas.append(m_bin.mean())   # fraction of pixels foreground

    areas = np.array(areas) if len(areas) > 0 else np.array([0.0])
    print("Folder masks:", mask_folder)
    print("Files found:", len(files))
    print("Files unreadable:", missing)
    print("Samples used:", len(areas))
    print("Mask coverage (fraction) :")
    print(f"  mean   : {areas.mean():.6f}")
    print(f"  median : {np.median(areas):.6f}")
    print(f"  min    : {areas.min():.6f}")
    print(f"  max    : {areas.max():.6f}")
    print(f"Percent masks with <0.1% coverage: {(areas < 0.001).mean()*100:.2f}%")
    print(f"Percent masks with <1% coverage  : {(areas < 0.01).mean()*100:.2f}%")
    print(f"Percent masks with <5% coverage  : {(areas < 0.05).mean()*100:.2f}%")

if __name__ == "__main__":
    # default path is data/train/masks; change cli if needed
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--masks", default="data/train/masks", help="Folder with mask images")
    args = p.parse_args()
    main(args.masks)
