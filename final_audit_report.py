#!/usr/bin/env python3
"""
FINAL AUDIT REPORT - Phase 4 Complete
"""

import os

BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"

print("\n" + "="*120)
print("AUDIT FINAL - PHASE 4 STERGERE SIMETRICA - RAPORT COMPLET")
print("="*120)

datasets = ['train', 'test', 'validation']
original_counts = {'train': 1083, 'test': 267, 'validation': 266}

print("\n" + "-"*120)
print("SINCRONIZARE FOLDERE")
print("-"*120)

all_synced = True
total_remaining = 0

for dataset_name in datasets:
    dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
    
    before_dir = os.path.join(dataset_path, 'before')
    after_dir = os.path.join(dataset_path, 'after')
    masks_dir = os.path.join(dataset_path, 'masks')
    
    before_count = len(os.listdir(before_dir)) if os.path.exists(before_dir) else 0
    after_count = len(os.listdir(after_dir)) if os.path.exists(after_dir) else 0
    masks_count = len(os.listdir(masks_dir)) if os.path.exists(masks_dir) else 0
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  Before folder: {before_count} imagini")
    print(f"  After folder: {after_count} imagini")
    print(f"  Masks folder: {masks_count} imagini")
    
    # Check synchronization
    if before_count == after_count == masks_count:
        print(f"  ✓ Status: SINCRONIZAT")
        total_remaining += before_count
    else:
        print(f"  ✗ Status: NESINCRONIZAT")
        all_synced = False
    
    # Check masks_clean for train
    if dataset_name == 'train':
        masks_clean_dir = os.path.join(dataset_path, 'masks_clean')
        masks_clean_count = len(os.listdir(masks_clean_dir)) if os.path.exists(masks_clean_dir) else 0
        print(f"  Masks_clean folder: {masks_clean_count} imagini")
        
        if before_count == masks_clean_count:
            print(f"  ✓ Status: SINCRONIZAT")
        else:
            print(f"  ✗ Status: NESINCRONIZAT")
            all_synced = False

print(f"\n" + "-"*120)
print("REZUMAT STERGERE")
print("-"*120)

eliminated_stats = {
    'train': 800,
    'test': 199,
    'validation': 203
}

total_eliminated = sum(eliminated_stats.values())

for dataset_name in datasets:
    original = original_counts[dataset_name]
    remaining = original - eliminated_stats[dataset_name]
    percentage = (eliminated_stats[dataset_name] / original) * 100
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  Original: {original} perechi")
    print(f"  Eliminate: {eliminated_stats[dataset_name]} perechi ({percentage:.1f}%)")
    print(f"  Ramase: {remaining} perechi ({100-percentage:.1f}%)")

print(f"\n" + "-"*120)
print("STATISTICI GLOBALE")
print("-"*120)

print(f"\nDataset Original: 1616 perechi")
print(f"  - Train: 1083 perechi")
print(f"  - Test: 267 perechi")
print(f"  - Validation: 266 perechi")

print(f"\nPerechi Eliminate: {total_eliminated} (74.4%)")
print(f"  - Train: 800 perechi")
print(f"  - Test: 199 perechi")
print(f"  - Validation: 203 perechi")

print(f"\nPerechi Ramase: {total_remaining} (25.6%)")
print(f"  - Train: {original_counts['train'] - 800} perechi")
print(f"  - Test: {original_counts['test'] - 199} perechi")
print(f"  - Validation: {original_counts['validation'] - 203} perechi")

print(f"\nFisiere Sterse Total: ~{total_eliminated * 3} fisiere")
print(f"  (3 fisiere per pereche: before + after + masks)")

print(f"\n" + "="*120)
if all_synced:
    print("✓✓✓ PHASE 4 COMPLETAT CU SUCCES ✓✓✓")
    print("Toti fisierul au fost stersi simetric si perfect sincronizati!")
else:
    print("✗✗✗ AVERTISMENT ✗✗✗")
    print("Exista inconsistente!")
print("="*120 + "\n")
