#!/usr/bin/env python3
import os
import csv

BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"
CSV_FILE = "/Users/admin/Documents/Facultatea/Proiect_RN/mismatched_scenes_full_list.csv"

print("\n" + "="*120)
print("DRY-RUN: PREVIEW STERGERE PERECHI MISMATCHED")
print("="*120)

with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    mismatched = list(reader)

print(f"\nTotal perechi mismatched: {len(mismatched)}")

train_count = len([p for p in mismatched if p['dataset'] == 'train'])
test_count = len([p for p in mismatched if p['dataset'] == 'test'])
val_count = len([p for p in mismatched if p['dataset'] == 'validation'])

print(f"\nDistributie:")
print(f"  Train: {train_count}")
print(f"  Test: {test_count}")
print(f"  Validation: {val_count}")

print(f"\n" + "="*120)
print("PREVIEW: Primele 5 perechi care vor fi sterse:")
print("="*120)

deleted_before = 0
deleted_after = 0
deleted_masks = 0
deleted_masks_clean = 0

for idx, pair in enumerate(mismatched[:5], 1):
    before_path = pair['before_path']
    after_path = pair['after_path']
    dataset = pair['dataset']
    filename = pair['filename']
    
    print(f"\n{idx}. {filename}")
    
    files_count = 0
    
    if os.path.exists(before_path):
        print(f"     [STERS] {before_path}")
        deleted_before += 1
        files_count += 1
    
    if os.path.exists(after_path):
        print(f"     [STERS] {after_path}")
        deleted_after += 1
        files_count += 1
    
    dataset_path = os.path.dirname(os.path.dirname(before_path))
    filename_png = os.path.basename(before_path).replace('.jpg', '.png')
    mask_path = os.path.join(dataset_path, 'masks', filename_png)
    if os.path.exists(mask_path):
        print(f"     [STERS] {mask_path}")
        deleted_masks += 1
        files_count += 1
    
    if dataset == 'train':
        mask_clean_path = os.path.join(dataset_path, 'masks_clean', filename_png)
        if os.path.exists(mask_clean_path):
            print(f"     [STERS] {mask_clean_path}")
            deleted_masks_clean += 1
            files_count += 1
    
    print(f"   Fisiere: {files_count}")

print(f"\n" + "="*120)
print("STATISTICI (din primele 5 perechi):")
print("="*120)
print(f"  Before: {deleted_before}/5")
print(f"  After: {deleted_after}/5")
print(f"  Masks: {deleted_masks}/5")
print(f"  Masks_clean: {deleted_masks_clean}/5")

avg = (deleted_before + deleted_after + deleted_masks + deleted_masks_clean) / 5.0
total_est = int(avg * len(mismatched))

print(f"\nESTIMARE: ~{total_est} fisiere vor fi sterse")

print(f"\n" + "="*120)
print("STAREA DATASET INAINTE:")
print("="*120)

for dset in ['train', 'test', 'validation']:
    dpath = os.path.join(BASE_DATA_DIR, dset)
    before_d = os.path.join(dpath, 'before')
    after_d = os.path.join(dpath, 'after')
    
    before_c = len(os.listdir(before_d)) if os.path.exists(before_d) else 0
    after_c = len(os.listdir(after_d)) if os.path.exists(after_d) else 0
    
    dset_mm = len([p for p in mismatched if p['dataset'] == dset])
    remaining = after_c - dset_mm
    
    print(f"\n{dset.upper()}:")
    print(f"  Curent: {after_c} perechi")
    print(f"  Eliminate: {dset_mm}")
    print(f"  Ramane: {remaining}")

print()
