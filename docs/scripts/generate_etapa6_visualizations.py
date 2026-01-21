#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generare vizualizări finale Etapa 6:
1. Confusion Matrix din evaluare test
2. Loss curves din training history
3. Identificare 5 imagini greșite cu cauze
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from pathlib import Path
import os

# Import din proiect
from src.neural_network.dataset import ChangeDetectionDataset
from src.neural_network.model import UNet

# ============================================================================
# CONFIGURARE
# ============================================================================

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL_PATH = "models/optimized_model.pt"
THRESHOLD = 0.55
MIN_PIXELS = 200
OUTPUT_DIR = "docs"

print(f"📱 Device: {DEVICE}")

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
# HELPER FUNCTIONS
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
# 1. GENEREAZĂ CONFUSION MATRIX
# ============================================================================

print("\n" + "="*80)
print("1. GENERARE CONFUSION MATRIX")
print("="*80)

test_dataset = ChangeDetectionDataset(root_dir="data/test", augment=False)
print(f"✅ Test dataset: {len(test_dataset)} imagini")

all_true = []
all_pred = []
error_samples = []

with torch.no_grad():
    for idx in range(len(test_dataset)):
        x, y = test_dataset[idx]
        x = x.unsqueeze(0).to(DEVICE)
        
        logits = model(x)
        prob = torch.sigmoid(logits)
        
        pred_prob_np = prob[0, 0].cpu().numpy()
        pred_binary = (pred_prob_np > THRESHOLD).astype(np.uint8)
        pred_filtered = apply_morphological_filter(pred_binary, min_pixels=MIN_PIXELS)
        
        y_np = y.numpy().astype(np.uint8)
        
        # Binarizare ground truth
        y_binary = (y_np > 0.5).astype(np.uint8)
        
        all_true.append(y_binary)
        all_pred.append(pred_filtered)
        
        # Detectare erori: FP și FN
        tp = np.sum(pred_filtered & y_binary)
        fp = np.sum(pred_filtered & ~y_binary)
        fn = np.sum(~pred_filtered & y_binary)
        
        if fp > 0 or fn > 0:  # Imaginea are erori
            error_samples.append({
                'index': idx,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'pred_pixels': np.sum(pred_filtered),
                'gt_pixels': np.sum(y_binary),
                'error_type': 'FP' if fp > fn else 'FN'
            })
        
        if (idx + 1) % 50 == 0:
            print(f"   Procesat {idx + 1}/{len(test_dataset)}")

# Concatenează și flatten pentru matrice de confuzie
all_true_flat = np.concatenate([m.flatten() for m in all_true])
all_pred_flat = np.concatenate([m.flatten() for m in all_pred])

# Calcul confusion matrix (pe pixeli)
cm = confusion_matrix(all_true_flat, all_pred_flat, labels=[0, 1])

print(f"\n✅ Confusion Matrix calculată din {len(test_dataset)} imagini")
print(f"   TN: {cm[0, 0]:,} | FP: {cm[0, 1]:,}")
print(f"   FN: {cm[1, 0]:,} | TP: {cm[1, 1]:,}")

# Generare vizualizare
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=',.0f', cmap='Blues', 
            xticklabels=['No Change', 'Change'],
            yticklabels=['No Change', 'Change'],
            cbar_kws={'label': 'Pixel Count'})
plt.title('Confusion Matrix - Model Optimizat (Etapa 6)', fontsize=14, fontweight='bold')
plt.ylabel('Ground Truth', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_optimized.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"✅ Confusion matrix salvată: {cm_path}")
plt.close()

# ============================================================================
# 2. GENEREAZĂ LOSS CURVES
# ============================================================================

print("\n" + "="*80)
print("2. GENERARE LOSS CURVES")
print("="*80)

history_df = pd.read_csv("results/training_history_refined.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
ax = axes[0, 0]
ax.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o', linewidth=2)
ax.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', marker='s', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# IoU
ax = axes[0, 1]
ax.plot(history_df['epoch'], history_df['train_iou'], label='Train IoU', marker='o', linewidth=2)
ax.plot(history_df['epoch'], history_df['val_iou'], label='Val IoU', marker='s', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('IoU')
ax.set_title('IoU Evolution')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=19, color='red', linestyle='--', alpha=0.5, label='Best epoch')

# Dice
ax = axes[1, 0]
ax.plot(history_df['epoch'], history_df['train_dice'], label='Train Dice', marker='o', linewidth=2)
ax.plot(history_df['epoch'], history_df['val_dice'], label='Val Dice', marker='s', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Dice')
ax.set_title('Dice Coefficient Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Learning Rate
ax = axes[1, 1]
ax.plot(history_df['epoch'], history_df['lr'], label='Learning Rate', marker='^', linewidth=2, color='green')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.suptitle('Model Optimizat (Etapa 6) - Training History', fontsize=16, fontweight='bold')
plt.tight_layout()

loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
print(f"✅ Loss curves salvate: {loss_curve_path}")
plt.close()

# ============================================================================
# 3. IDENTIFICARE 5 IMAGINI GREȘITE
# ============================================================================

print("\n" + "="*80)
print("3. ANALIZA 5 IMAGINI GREȘITE")
print("="*80)

# Sort error samples by total error (FP + FN)
error_samples_sorted = sorted(error_samples, key=lambda x: x['fp'] + x['fn'], reverse=True)
top_5_errors = error_samples_sorted[:5]

error_analysis = []

for rank, err in enumerate(top_5_errors, 1):
    idx = err['index']
    
    print(f"\n{rank}. Sample #{idx}:")
    print(f"   Error type: {err['error_type']}")
    print(f"   TP: {err['tp']:,} | FP: {err['fp']:,} | FN: {err['fn']:,}")
    print(f"   Pred pixels: {err['pred_pixels']:,} | GT pixels: {err['gt_pixels']:,}")
    
    # Analiză cauza
    if err['fn'] > err['fp']:
        # False Negative - model missed something
        error_type_detail = "False Negative - Model missed change"
        if err['gt_pixels'] < 5000:
            probable_cause = "Obiect mic sub limita 200px sau contrast scăzut"
        elif err['gt_pixels'] > 30000:
            probable_cause = "Obiect mare: posibil contrast scăzut sau iluminare neuniformă"
        else:
            probable_cause = "Obiect mediu: posibil la margine sau partial obscurat"
    else:
        # False Positive - model detected noise
        error_type_detail = "False Positive - Model detected noise"
        probable_cause = "Zgomot senzor sau artefact compresie fișier"
    
    error_analysis.append({
        'rank': int(rank),
        'sample_index': int(idx),
        'error_type': error_type_detail,
        'probable_cause': probable_cause,
        'tp_pixels': int(err['tp']),
        'fp_pixels': int(err['fp']),
        'fn_pixels': int(err['fn']),
        'gt_pixels': int(err['gt_pixels']),
        'pred_pixels': int(err['pred_pixels'])
    })
    
    print(f"   Cauza probabilă: {probable_cause}")

# Salvează analiză în JSON
error_analysis_path = "results/error_analysis_etapa6.json"
with open(error_analysis_path, 'w') as f:
    json.dump(error_analysis, f, indent=2)

print(f"\n✅ Analiză erori salvată: {error_analysis_path}")

# Salvează și în CSV pentru ușor vizualizare
error_df = pd.DataFrame(error_analysis)
error_csv_path = "results/top_5_errors_etapa6.csv"
error_df.to_csv(error_csv_path, index=False)
print(f"✅ CSV erori salvat: {error_csv_path}")

# ============================================================================
# RAPORT FINAL
# ============================================================================

print("\n" + "="*80)
print("✅ VIZUALIZĂRI FINALE COMPLETATE")
print("="*80)
print(f"\n📊 Fișiere generate:")
print(f"   1. {cm_path}")
print(f"   2. {loss_curve_path}")
print(f"   3. {error_analysis_path}")
print(f"   4. {error_csv_path}")
print(f"\n📋 Top 5 imagini greșite:")
for rec in error_analysis:
    print(f"   {rec['rank']}. Sample #{rec['sample_index']:04d} - {rec['error_type']}")
    print(f"      → {rec['probable_cause']}")

print("\n" + "="*80)
