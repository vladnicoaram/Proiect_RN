#!/usr/bin/env python3
"""
REORGANIZE DATASET - Pastreaza doar perechi PASSED (190) si reorganizeaza dataset
"""

import os
import shutil
import csv
from pathlib import Path

BASE_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN"
DATA_DIR = Path(BASE_DIR) / 'data'
REJECTED_DIR = DATA_DIR / 'REJECTED_BY_AUDIT'

print("\n" + "="*80)
print("REORGANIZE DATASET - PASTREAZA DOAR PERECHI PASSED (190)")
print("="*80)

# Load audit results
audit_file = Path(BASE_DIR) / 'audit_passed_perechi.csv'
passed_pairs = {}

with open(audit_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row['dataset']
        filename = row['filename']
        if dataset not in passed_pairs:
            passed_pairs[dataset] = set()
        passed_pairs[dataset].add(filename)

print(f"\nPassed pairs loaded:")
for ds, files in passed_pairs.items():
    print(f"  {ds}: {len(files)} pairs")

# ============================================================================
# STERGERE PERECHI RESPINSE DIN FOLDERELE ACTIVE
# ============================================================================

print(f"\n[1] Removing rejected pairs from active dataset...")

for dataset in ['train', 'test', 'validation']:
    if dataset not in passed_pairs:
        continue
    
    dataset_path = DATA_DIR / dataset
    passed_files = passed_pairs[dataset]
    
    for subfolder in ['before', 'after', 'masks'] + (['masks_clean'] if dataset == 'train' else []):
        folder_path = dataset_path / subfolder
        if folder_path.exists():
            all_files = os.listdir(folder_path)
            for filename in all_files:
                if filename not in passed_files:
                    file_path = folder_path / filename
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"  Error removing {file_path}: {e}")
    
    print(f"  ✓ {dataset}: kept only passed pairs")

# ============================================================================
# VERIFICA SINCRONIZARE
# ============================================================================

print(f"\n[2] Verifying synchronization...")

for dataset in ['train', 'test', 'validation']:
    dataset_path = DATA_DIR / dataset
    
    before_dir = dataset_path / 'before'
    after_dir = dataset_path / 'after'
    masks_dir = dataset_path / 'masks'
    
    before_count = len(os.listdir(before_dir)) if before_dir.exists() else 0
    after_count = len(os.listdir(after_dir)) if after_dir.exists() else 0
    masks_count = len(os.listdir(masks_dir)) if masks_dir.exists() else 0
    
    print(f"\n{dataset.upper()}:")
    print(f"  Before: {before_count}")
    print(f"  After: {after_count}")
    print(f"  Masks: {masks_count}")
    
    if before_count == after_count == masks_count:
        print(f"  ✓ Synchronized")
    else:
        print(f"  ✗ MISMATCH!")
    
    if dataset == 'train':
        masks_clean_dir = dataset_path / 'masks_clean'
        masks_clean_count = len(os.listdir(masks_clean_dir)) if masks_clean_dir.exists() else 0
        print(f"  Masks_clean: {masks_clean_count}")
        if before_count == masks_clean_count:
            print(f"  ✓ Masks_clean synchronized")
        else:
            print(f"  ✗ Masks_clean mismatch")

print(f"\n" + "="*80)
print("✓ Dataset reorganization complete!")
print("="*80 + "\n")
