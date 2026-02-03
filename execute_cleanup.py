#!/usr/bin/env python3
"""
Phase 4: EXECUTIE REALA - Stergere simetrica a 1202 perechi mismatched
"""

import os
import csv
from collections import defaultdict

BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"
CSV_FILE = "/Users/admin/Documents/Facultatea/Proiect_RN/mismatched_scenes_full_list.csv"

DATASETS = {
    'train': ['before', 'after', 'masks', 'masks_clean'],
    'test': ['before', 'after', 'masks'],
    'validation': ['before', 'after', 'masks']
}

print("\n" + "="*120)
print("PHASE 4: EXECUTIE REALA - STERGERE SIMETRICA")
print("="*120)

# Incarca CSV
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    mismatched = list(reader)

print(f"\nTotal perechi de sters: {len(mismatched)}")

# Statistici
train_c = len([p for p in mismatched if p['dataset'] == 'train'])
test_c = len([p for p in mismatched if p['dataset'] == 'test'])
val_c = len([p for p in mismatched if p['dataset'] == 'validation'])

print(f"  Train: {train_c}")
print(f"  Test: {test_c}")
print(f"  Validation: {val_c}")

# STERGERE
print(f"\n" + "="*120)
print("EXECUTIE STERGERE...")
print("="*120)

deleted_stats = {
    'before': 0,
    'after': 0,
    'masks': 0,
    'masks_clean': 0,
    'failed': 0
}

for idx, pair in enumerate(mismatched, 1):
    if idx % 200 == 0:
        print(f"Progres: {idx}/{len(mismatched)}")
    
    before_path = pair['before_path']
    after_path = pair['after_path']
    dataset = pair['dataset']
    
    # Sterge BEFORE
    if os.path.exists(before_path):
        try:
            os.remove(before_path)
            deleted_stats['before'] += 1
        except Exception as e:
            deleted_stats['failed'] += 1
    
    # Sterge AFTER
    if os.path.exists(after_path):
        try:
            os.remove(after_path)
            deleted_stats['after'] += 1
        except Exception as e:
            deleted_stats['failed'] += 1
    
    # Sterge MASK
    dataset_path = os.path.dirname(os.path.dirname(before_path))
    filename_png = os.path.basename(before_path).replace('.jpg', '.png')
    mask_path = os.path.join(dataset_path, 'masks', filename_png)
    if os.path.exists(mask_path):
        try:
            os.remove(mask_path)
            deleted_stats['masks'] += 1
        except Exception as e:
            deleted_stats['failed'] += 1
    
    # Sterge MASK_CLEAN (train only)
    if dataset == 'train':
        mask_clean_path = os.path.join(dataset_path, 'masks_clean', filename_png)
        if os.path.exists(mask_clean_path):
            try:
                os.remove(mask_clean_path)
                deleted_stats['masks_clean'] += 1
            except Exception as e:
                deleted_stats['failed'] += 1

print(f"\n" + "="*120)
print("REZULTATE STERGERE:")
print("="*120)
print(f"  Before sters: {deleted_stats['before']}")
print(f"  After sters: {deleted_stats['after']}")
print(f"  Masks sters: {deleted_stats['masks']}")
print(f"  Masks_clean sters: {deleted_stats['masks_clean']}")
print(f"  Erori: {deleted_stats['failed']}")

total_deleted = sum(deleted_stats[k] for k in ['before', 'after', 'masks', 'masks_clean'])
print(f"  TOTAL FISIERE STERSE: {total_deleted}")

# AUDIT
print(f"\n" + "="*120)
print("AUDIT POST-STERGERE: Verificare sincronizare foldere")
print("="*120)

all_synced = True

for dataset_name in DATASETS.keys():
    dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
    
    before_dir = os.path.join(dataset_path, 'before')
    after_dir = os.path.join(dataset_path, 'after')
    masks_dir = os.path.join(dataset_path, 'masks')
    
    before_files = set(os.listdir(before_dir)) if os.path.exists(before_dir) else set()
    after_files = set(os.listdir(after_dir)) if os.path.exists(after_dir) else set()
    masks_files = set(f.replace('.png', '.jpg') for f in os.listdir(masks_dir) if f.endswith('.png')) if os.path.exists(masks_dir) else set()
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  Before: {len(before_files)}")
    print(f"  After: {len(after_files)}")
    print(f"  Masks: {len(masks_files)}")
    
    # Verifica sincronizare
    if len(before_files) == len(after_files) == len(masks_files):
        print(f"  [OK] Sincronizare perfecta")
    else:
        print(f"  [EROARE] Nesincronizare!")
        all_synced = False
        
        # Detecteaza orfani
        orphans_before = before_files - after_files
        orphans_after = after_files - before_files
        orphans_masks = masks_files - before_files
        
        if orphans_before:
            print(f"    Orfani in before: {len(orphans_before)}")
        if orphans_after:
            print(f"    Orfani in after: {len(orphans_after)}")
        if orphans_masks:
            print(f"    Orfani in masks: {len(orphans_masks)}")
    
    # Masks_clean pentru train
    if dataset_name == 'train':
        masks_clean_dir = os.path.join(dataset_path, 'masks_clean')
        masks_clean_files = set(f.replace('.png', '.jpg') for f in os.listdir(masks_clean_dir) if f.endswith('.png')) if os.path.exists(masks_clean_dir) else set()
        print(f"  Masks_clean: {len(masks_clean_files)}")
        
        if len(before_files) == len(masks_clean_files):
            print(f"  [OK] Train masks_clean sincronizat")
        else:
            print(f"  [EROARE] Train masks_clean nesincronizat!")
            all_synced = False

# STATISTICI FINALE
print(f"\n" + "="*120)
print("STATISTICI FINALA - DATASET DUPA STERGERE")
print("="*120)

for dataset_name in DATASETS.keys():
    dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
    after_dir = os.path.join(dataset_path, 'after')
    after_count = len(os.listdir(after_dir)) if os.path.exists(after_dir) else 0
    
    dataset_mm = len([p for p in mismatched if p['dataset'] == dataset_name])
    print(f"\n{dataset_name.upper()}:")
    print(f"  Perechi ramase: {after_count}")
    print(f"  Perechi eliminate: {dataset_mm}")

print(f"\n" + "="*120)
if all_synced:
    print("SUCCESS: Toti fisierul au fost stersi simetric!")
else:
    print("WARNING: Exista inconsistente!")
print("="*120 + "\n")
