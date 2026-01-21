#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screenshot Inferență - Etapa 5 Livrabil
Rulează inferență pe sample 91 și salvează overlay
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from pathlib import Path
from scipy import ndimage

# Import din proiect
from dataset import ChangeDetectionDataset
from model import UNet

# ============================================================================
# CONFIGURARE
# ============================================================================

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL_PATH = "../../models/trained_model.pt"
SAMPLE_ID = 91
OUTPUT_DIR = "../../docs/screenshots"
THRESHOLD = 0.55
MIN_PIXELS = 200

print(f"📱 Device: {DEVICE}")

# ============================================================================
# FUNCȚII HELPER
# ============================================================================

def apply_morphological_filter(mask, min_pixels=200):
    """Filtru morfologic: elimină componentele mici"""
    labeled_array, num_features = ndimage.label(mask)
    filtered_mask = np.zeros_like(mask)
    
    for i in range(1, num_features + 1):
        component = (labeled_array == i).astype(np.uint8)
        component_size = np.sum(component)
        if component_size > min_pixels:
            filtered_mask += component
    
    return (filtered_mask > 0).astype(np.uint8)

# ============================================================================
# ÎNCARCĂ MODEL
# ============================================================================

model = UNet(in_channels=6, out_channels=1)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

print(f"✅ Model încărcat: {MODEL_PATH}")

# ============================================================================
# ÎNCARCĂ DATASET
# ============================================================================

test_dataset = ChangeDetectionDataset(root_dir="../../data/test", augment=False)
print(f"✅ Test dataset: {len(test_dataset)} imagini")

# Obține sample-ul
x, y = test_dataset[SAMPLE_ID]
x = x.unsqueeze(0).to(DEVICE)
y_np = y.numpy().astype(np.uint8)

print(f"🔍 Procesez Sample #{SAMPLE_ID}...")

# ============================================================================
# INFERENȚĂ
# ============================================================================

with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)

pred_prob_np = prob[0, 0].cpu().numpy()
pred_binary = (pred_prob_np > THRESHOLD).astype(np.uint8)
pred_filtered = apply_morphological_filter(pred_binary, min_pixels=MIN_PIXELS)

# ============================================================================
# PRELUCREAZĂ IMAGINI
# ============================================================================

x_np = x[0].cpu().numpy()

# Extrage before (RGB) și after (RGB) din 6 canale
before_img = x_np[:3].transpose(1, 2, 0)
before_img = ((before_img + 1) / 2 * 255).astype(np.uint8)

after_img = x_np[3:].transpose(1, 2, 0)
after_img = ((after_img + 1) / 2 * 255).astype(np.uint8)

# ============================================================================
# CREAZĂ OVERLAY - PREDICȚIE FILTRATĂ PE IMAGINEA AFTER
# ============================================================================

overlay = after_img.copy().astype(np.float32)

# Marcheaza predicție în verde
green_mask = np.zeros_like(overlay)
green_mask[:, :, 1] = pred_filtered * 255

# Blanding
alpha = 0.3
overlay = (overlay * (1 - alpha) + green_mask * alpha).astype(np.uint8)

# Contururi în roșu
contours, _ = cv2.findContours(pred_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

# ============================================================================
# SALVEAZĂ REZULTATE
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "inference_real.png")
cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"✅ Screenshot salvat: {output_path}")

# ============================================================================
# METRICI SAMPLE
# ============================================================================

tp = np.sum(pred_filtered & y_np)
fp = np.sum(pred_filtered & ~y_np)
fn = np.sum(~pred_filtered & y_np)

if (tp + fp) > 0:
    precision = tp / (tp + fp)
else:
    precision = 0.0

if (tp + fn) > 0:
    recall = tp / (tp + fn)
else:
    recall = 0.0

if (tp + fp + fn) > 0:
    iou = tp / (tp + fp + fn)
else:
    iou = 0.0

print(f"\n📊 Metrici Sample #{SAMPLE_ID}:")
print(f"   Precision: {precision:.2%}")
print(f"   Recall: {recall:.2%}")
print(f"   IoU: {iou:.2%}")
print(f"   TP: {tp:,} pixeli | FP: {fp:,} | FN: {fn:,}")

# ============================================================================
# SALVEAZĂ COMPARAȚIE LATERALĂ
# ============================================================================

gt_color = cv2.cvtColor((y_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
pred_color = cv2.cvtColor((pred_filtered * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

comparison = np.hstack([gt_color, pred_color, overlay])
comparison_path = os.path.join(OUTPUT_DIR, "inference_comparison.png")
cv2.imwrite(comparison_path, comparison)

print(f"✅ Comparație salvată: {comparison_path}")

print("\n" + "=" * 80)
print("✅ SCREENSHOT COMPLETAT PENTRU ETAPA 5")
print("=" * 80)
