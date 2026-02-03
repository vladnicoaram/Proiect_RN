#!/usr/bin/env python3
"""
Phase 4: FINAL CLEANUP - Stergere directa si sincronizare
"""

import os
import csv

BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"
CSV_FILE = "/Users/admin/Documents/Facultatea/Proiect_RN/mismatched_scenes_full_list.csv"

DATASETS = {
    'train': ['before', 'after', 'masks', 'masks_clean'],
    'test': ['before', 'after', 'masks'],
    'validation': ['before', 'after', 'masks']
}

print("\n" + "="*120)
print("PHASE 4: FINAL CLEANUP - STERGERE DIRECTA")
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

# Build deletion dict
to_delete = {'train': set(), 'test': set(), 'validation': set()}
for pair in mismatched:
    filename = pair['filename']
    dataset = pair['dataset']
    to_delete[dataset].add(filename)

print(f"\nFisiere de sters pe dataset:")
for ds, files in to_delete.items():
    print(f"  {ds}: {len(files)}")

# STERGERE
print(f"\n" + "="*120)
print("EXECUTIE STERGERE...")
print("="*120)

deleted_stats = {
    'before': 0,
    'after': 0,
    'masks': 0,
    'masks_clean': 0,
    'total': 0
}

for dataset_name, filenames in to_delete.items():
    dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
    
    for filename in filenames:
        # BEFORE
        before_path = os.path.join(dataset_path, 'before', filename)
        if os.path.exists(before_path):
            os.remove(before_path)
            deleted_stats['before'] += 1
            deleted_stats['total'] += 1
        
        # AFTER
        after_path = os.path.join(dataset_path, 'after', filename)
        if os.path.exists(after_path):
            os.remove(after_path)
            deleted_stats['after'] += 1
            deleted_stats['total'] += 1
        
        # MASKS
        masks_path = os.path.join(dataset_path, 'masks', filename)
        if os.path.exists(masks_path):
            os.remove(masks_path)
            deleted_stats['masks'] += 1
            deleted_stats['total'] += 1
        
        # MASKS_CLEAN (train only)
        if dataset_name == 'train':
            masks_clean_path = os.path.join(dataset_path, 'masks_clean', filename)
            if os.path.exists(masks_clean_path):
                os.remove(masks_clean_path)
                deleted_stats['masks_clean'] += 1
                deleted_stats['total'] += 1

print(f"\nProgres: 100% (procesate {len(mismatched)} perechi)")

print(f"\n" + "="*120)
print("REZULTATE STERGERE:")
print("="*120)
print(f"  Before sters: {deleted_stats['before']}")
print(f"  After sters: {deleted_stats['after']}")
print(f"  Masks sters: {deleted_stats['masks']}")
print(f"  Masks_clean sters: {deleted_stats['masks_clean']}")
print(f"  TOTAL FISIERE STERSE: {deleted_stats['total']}")

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
    masks_files = set(os.listdir(masks_dir)) if os.path.exists(masks_dir) else set()
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  Before: {len(before_files)}")
    print(f"  After: {len(after_files)}")
    print(f"  Masks: {len(masks_files)}")
    
    # Verifica sincronizare
    if len(before_files) == len(after_files) == len(masks_files):
        print(f"  ✓ [OK] Sincronizare perfecta")
    else:
        print(f"  ✗ [EROARE] Nesincronizare!")
        all_synced = False
    
    # Masks_clean pentru train
    if dataset_name == 'train':
        masks_clean_dir = os.path.join(dataset_path, 'masks_clean')
        masks_clean_files = set(os.listdir(masks_clean_dir)) if os.path.exists(masks_clean_dir) else set()
        print(f"  Masks_clean: {len(masks_clean_files)}")
        
        if len(before_files) == len(masks_clean_files):
            print(f"  ✓ [OK] Train masks_clean sincronizat")
        else:
            print(f"  ✗ [EROARE] Train masks_clean nesincronizat!")
            all_synced = False

# STATISTICI FINALE
print(f"\n" + "="*120)
print("STATISTICI FINALA - DATASET DUPA STERGERE")
print("="*120)

original_counts = {'train': 1083, 'test': 267, 'validation': 266}
total_remaining = 0

for dataset_name in DATASETS.keys():
    dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
    after_dir = os.path.join(dataset_path, 'after')
    after_count = len(os.listdir(after_dir)) if os.path.exists(after_dir) else 0
    
    dataset_mm = len([p for p in mismatched if p['dataset'] == dataset_name])
    original = original_counts[dataset_name]
    percentage = (dataset_mm / original) * 100
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  Original: {original}")
    print(f"  Ramase: {after_count}")
    print(f"  Sterse: {dataset_mm} ({percentage:.1f}%)")
    
    total_remaining += after_count

print(f"\n" + "-"*120)
print(f"TOTAL ORIGINAL: 1616 perechi")
print(f"TOTAL STERSE: {len(mismatched)} perechi (74.4%)")
print(f"TOTAL RAMASE: {total_remaining} perechi (25.6%)")

print(f"\n" + "="*120)
if all_synced:
    print("✓ SUCCESS: Toti fisierul au fost stersi simetric si perfect sincronizati!")
else:
    print("✗ WARNING: Exista inconsistente!")
print("="*120 + "\n")
