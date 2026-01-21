#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generare screenshot automat pentru Etapa 6
Inferență pe sample 91 cu modelul optimizat
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from scipy import ndimage

# Import din proiect
from src.neural_network.model import UNet
from src.neural_network.dataset import ChangeDetectionDataset

# ============================================================================
# CONFIGURARE
# ============================================================================

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL_PATH = "models/optimized_model.pt"
SAMPLE_ID = 91
THRESHOLD = 0.55
MIN_PIXELS = 200
OUTPUT_DIR = "docs/screenshots"

print(f"📱 Device: {DEVICE}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_morphological_filter(mask, min_pixels=200):
    """Filtru morfologic"""
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
# ÎNCARCĂ DATASET ȘI SAMPLE
# ============================================================================

test_dataset = ChangeDetectionDataset(root_dir="data/test", augment=False)
print(f"✅ Test dataset: {len(test_dataset)} imagini")

x, y = test_dataset[SAMPLE_ID]
y_np = y.numpy().astype(np.uint8)

print(f"🔍 Procesez Sample #{SAMPLE_ID}...")

# ============================================================================
# INFERENȚĂ
# ============================================================================

with torch.no_grad():
    x_batch = x.unsqueeze(0).to(DEVICE)
    logits = model(x_batch)
    prob = torch.sigmoid(logits)

pred_prob_np = prob[0, 0].cpu().numpy()
pred_binary = (pred_prob_np > THRESHOLD).astype(np.uint8)
pred_filtered = apply_morphological_filter(pred_binary, min_pixels=MIN_PIXELS)

# ============================================================================
# PRELUCREAZĂ IMAGINI
# ============================================================================

x_np = x.numpy()

before_img = x_np[:3].transpose(1, 2, 0)
before_img = ((before_img + 1) / 2 * 255).astype(np.uint8)

after_img = x_np[3:].transpose(1, 2, 0)
after_img = ((after_img + 1) / 2 * 255).astype(np.uint8)

# ============================================================================
# CREAZĂ VIZUALIZĂRI
# ============================================================================

# 1. Overlay - predicție pe imagine
overlay = after_img.copy().astype(np.float32)
green_mask = np.zeros_like(overlay)
green_mask[:, :, 1] = pred_filtered * 255

alpha = 0.3
overlay = (overlay * (1 - alpha) + green_mask * alpha).astype(np.uint8)

# Contururi în roșu
contours, _ = cv2.findContours(pred_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

# 2. Comparație laterală: GT | Predicted | Overlay
# Asigură că y_np este 2D
if y_np.ndim > 2:
    y_np = y_np.squeeze()

gt_color = cv2.cvtColor((y_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
pred_color = cv2.cvtColor((pred_filtered * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

print(f"DEBUG shapes:")
print(f"  gt_color: {gt_color.shape}")
print(f"  pred_color: {pred_color.shape}")
print(f"  overlay: {overlay.shape}")

comparison = np.hstack([gt_color, pred_color, overlay])

# ============================================================================
# SALVEAZĂ REZULTATE
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Salvează overlay principal
output_path = os.path.join(OUTPUT_DIR, "inference_optimized.png")
cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# Salvează comparație
comparison_path = os.path.join(OUTPUT_DIR, "inference_optimized_comparison.png")
cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

# ============================================================================
# METRICI
# ============================================================================

tp = np.sum(pred_filtered & y_np)
fp = np.sum(pred_filtered & ~y_np)
fn = np.sum(~pred_filtered & y_np)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

# ============================================================================
# OUTPUT
# ============================================================================

print("\n" + "="*80)
print("✅ SCREENSHOT UI COMPLETAT")
print("="*80)

print(f"\n📸 Fișiere salvate:")
print(f"   - {output_path} (Overlay principal)")
print(f"   - {comparison_path} (Comparație GT|Pred|Overlay)")

print(f"\n📊 Metrici Sample #{SAMPLE_ID}:")
print(f"   Precision: {precision:.2%}")
print(f"   Recall: {recall:.2%}")
print(f"   IoU: {iou:.2%}")
print(f"   TP: {tp:,} pixeli | FP: {fp:,} | FN: {fn:,}")

print(f"\n⚙️  Parametri Configurare:")
print(f"   Threshold: {THRESHOLD}")
print(f"   Min Component Size: {MIN_PIXELS} pixeli")
print(f"   Device: {DEVICE}")
print(f"   Model: {MODEL_PATH}")

print("\n" + "="*80)
