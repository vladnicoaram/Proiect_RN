#!/usr/bin/env python3
"""
HELPER SCRIPT - Verifica dataset si lanseaza antrenare
"""

import os
import sys
from pathlib import Path

BASE_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN"

print("\n" + "="*80)
print("DATASET VALIDATION & TRAINING LAUNCHER")
print("="*80)

# ============================================================================
# VERIFICARE DATASET
# ============================================================================

print("\n[1] Checking dataset structure...")

datasets = {
    'train': 122,
    'test': 34,
    'validation': 34
}

all_ok = True
total_pairs = 0

for dataset_name, expected_count in datasets.items():
    dataset_path = Path(BASE_DIR) / 'data' / dataset_name
    
    before_dir = dataset_path / 'before'
    after_dir = dataset_path / 'after'
    masks_dir = dataset_path / 'masks'
    
    if before_dir.exists():
        before_count = len(os.listdir(before_dir))
        after_count = len(os.listdir(after_dir)) if after_dir.exists() else 0
        masks_count = len(os.listdir(masks_dir)) if masks_dir.exists() else 0
        
        print(f"\n{dataset_name.upper()}:")
        print(f"  Before: {before_count} (expected {expected_count})")
        print(f"  After: {after_count}")
        print(f"  Masks: {masks_count}")
        
        if before_count == after_count == masks_count == expected_count:
            print(f"  ✓ OK")
            total_pairs += before_count
        else:
            print(f"  ✗ MISMATCH!")
            all_ok = False
    else:
        print(f"\n{dataset_name.upper()}: NOT FOUND")
        all_ok = False

print(f"\nTotal pairs: {total_pairs} (expected 190)")

if not all_ok or total_pairs != 190:
    print("\n✗ Dataset validation FAILED")
    sys.exit(1)
else:
    print("\n✓ Dataset validation PASSED")

# ============================================================================
# VERIFICARE MODEL PATHS
# ============================================================================

print("\n[2] Checking model and results directories...")

paths = [
    ('checkpoints', 'Model save directory'),
    ('results', 'Results directory'),
]

for path, desc in paths:
    full_path = Path(BASE_DIR) / path
    if full_path.exists():
        print(f"  ✓ {desc}: {full_path}")
    else:
        print(f"  ! Creating {desc}: {full_path}")
        full_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# AFISARE CONFIGURATIE
# ============================================================================

print("\n[3] Training configuration:")
config = {
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': '1e-4',
    'patience': 15,
    'device': 'mps (Mac M1)',
    'data_augmentation': ['Rotations ±30°', 'H/V Flips', 'Brightness/Contrast', 'Zoom 0.9-1.1x']
}

for key, value in config.items():
    if isinstance(value, list):
        print(f"  {key}:")
        for item in value:
            print(f"    - {item}")
    else:
        print(f"  {key}: {value}")

# ============================================================================
# LANSARE ANTRENARE
# ============================================================================

print("\n[4] Ready to start training!")
print("\nTo run training, execute:")
print("  cd /Users/admin/Documents/Facultatea/Proiect_RN")
print("  python3 src/neural_network/train_final_refined.py")

print("\n" + "="*80)
