# src/preprocessing/cleanup_masks_batch.py
import cv2, os
from pathlib import Path
import numpy as np

MASK_DIR = Path("data/train/masks")
OUT_DIR = Path("data/train/masks_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

min_area_px = 200   # ajustează (ex: 200 pixeli la 256x256)
top_crop = 10       # dacă vrei elimina artefact sus, crop top rows

for p in MASK_DIR.iterdir():
    if p.suffix.lower() not in (".png",".jpg",".jpeg"): continue
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None: continue
    # binarize
    m_bin = (m>127).astype('uint8')*255
    # optional crop top
    if top_crop>0:
        m_bin[:top_crop,:] = 0
    # open/close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    # remove small components
    contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(m_bin)
    for c in contours:
        if cv2.contourArea(c) >= min_area_px:
            cv2.drawContours(clean, [c], -1, 255, -1)
    cv2.imwrite(str(OUT_DIR/p.name), clean)
print("Saved cleaned masks to", OUT_DIR)
