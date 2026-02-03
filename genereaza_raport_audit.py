#!/usr/bin/env python3
"""
Genereaza raport final - Analiza Audit Certificare
"""

import os
import csv
import pandas as pd

BASE_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN"

print("\n" + "="*140)
print("RAPORT FINAL - AUDIT DE CERTIFICARE")
print("="*140)

# Incarca date
passed_df = pd.read_csv(os.path.join(BASE_DIR, "audit_passed_perechi.csv"))
rejected_df = pd.read_csv(os.path.join(BASE_DIR, "audit_rejected_perechi.csv"))

print(f"\n{'='*140}")
print("REZUMAT AUDIT")
print("="*140)

total_perechi = len(passed_df) + len(rejected_df)
passed_count = len(passed_df)
rejected_count = len(rejected_df)

print(f"\nTotal perechi analizate: {total_perechi}")
print(f"  ✓ PASSED (Gold Standard): {passed_count} ({(passed_count/total_perechi)*100:.1f}%)")
print(f"  ✗ REJECTED: {rejected_count} ({(rejected_count/total_perechi)*100:.1f}%)")

# Analiza PASSED
print(f"\n{'='*140}")
print("STATISTICI - PERECHI PASSED (190)")
print("="*140)

print(f"\nSSIM SCORE:")
print(f"  Mean: {passed_df['ssim'].mean():.4f}")
print(f"  Median: {passed_df['ssim'].median():.4f}")
print(f"  Std Dev: {passed_df['ssim'].std():.4f}")
print(f"  Min: {passed_df['ssim'].min():.4f}")
print(f"  Max: {passed_df['ssim'].max():.4f}")

print(f"\nORB MATCHES:")
print(f"  Mean: {passed_df['orb_matches'].mean():.1f}")
print(f"  Median: {passed_df['orb_matches'].median():.1f}")
print(f"  Std Dev: {passed_df['orb_matches'].std():.1f}")
print(f"  Min: {passed_df['orb_matches'].min():.0f}")
print(f"  Max: {passed_df['orb_matches'].max():.0f}")

print(f"\nHISTOGRAM CORRELATION:")
print(f"  Mean: {passed_df['histogram_corr'].mean():.4f}")
print(f"  Median: {passed_df['histogram_corr'].median():.4f}")
print(f"  Std Dev: {passed_df['histogram_corr'].std():.4f}")
print(f"  Min: {passed_df['histogram_corr'].min():.4f}")
print(f"  Max: {passed_df['histogram_corr'].max():.4f}")

# Analiza REJECTED
print(f"\n{'='*140}")
print("ANALIZA - PERECHI REJECTED (224)")
print("="*140)

# Raiuni de respingere
ssim_fail = len(rejected_df[rejected_df['ssim'] < 0.70])
orb_fail = len(rejected_df[rejected_df['orb_matches'] < 25])
hist_fail = len(rejected_df[rejected_df['histogram_corr'] < 0.60])

print(f"\nRatiuni de respingere:")
print(f"  SSIM < 0.70: {ssim_fail} perechi ({(ssim_fail/rejected_count)*100:.1f}%)")
print(f"  ORB < 25: {orb_fail} perechi ({(orb_fail/rejected_count)*100:.1f}%)")
print(f"  Histogram < 0.60: {hist_fail} perechi ({(hist_fail/rejected_count)*100:.1f}%)")

print(f"\nTop 10 WORST REJECTED:")
worst = rejected_df.nsmallest(10, 'ssim')[['dataset', 'filename', 'ssim', 'orb_matches', 'histogram_corr']]
for idx, row in worst.iterrows():
    print(f"  {row['dataset']}/{row['filename']}: SSIM={row['ssim']:.4f}, ORB={row['orb_matches']:.0f}, Hist={row['histogram_corr']:.4f}")

# Distributie pe dataset
print(f"\n{'='*140}")
print("DISTRIBUTIE PE DATASET - PASSED vs REJECTED")
print("="*140)

for dataset in ['train', 'test', 'validation']:
    passed_ds = len(passed_df[passed_df['dataset'] == dataset])
    rejected_ds = len(rejected_df[rejected_df['dataset'] == dataset])
    total_ds = passed_ds + rejected_ds
    
    print(f"\n{dataset.upper()}:")
    print(f"  Total: {total_ds}")
    print(f"  Passed: {passed_ds} ({(passed_ds/total_ds)*100:.1f}%)")
    print(f"  Rejected: {rejected_ds} ({(rejected_ds/total_ds)*100:.1f}%)")

# Locatie fisiere
rejected_dir = os.path.join(BASE_DIR, "data", "REJECTED_BY_AUDIT")
print(f"\n{'='*140}")
print("FISIERE RESPINSE")
print("="*140)

if os.path.exists(rejected_dir):
    print(f"\nFoldere REJECTED: {rejected_dir}")
    print(f"\nStructura:")
    for dataset in ['train', 'test', 'validation']:
        ds_path = os.path.join(rejected_dir, dataset)
        if os.path.exists(ds_path):
            before_count = len(os.listdir(os.path.join(ds_path, 'before'))) if os.path.exists(os.path.join(ds_path, 'before')) else 0
            after_count = len(os.listdir(os.path.join(ds_path, 'after'))) if os.path.exists(os.path.join(ds_path, 'after')) else 0
            masks_count = len(os.listdir(os.path.join(ds_path, 'masks'))) if os.path.exists(os.path.join(ds_path, 'masks')) else 0
            
            print(f"  {dataset}: before={before_count}, after={after_count}, masks={masks_count}")

# Validare
print(f"\n{'='*140}")
print("CONCLUZIE VALIDARE")
print("="*140)

mean_ssim_passed = passed_df['ssim'].mean()
required_mean = 0.85

print(f"\nMean SSIM (Perechi PASSED): {mean_ssim_passed:.4f}")
print(f"Prag Minim: {required_mean}")

if mean_ssim_passed >= required_mean:
    print(f"\n✓ DATASET-UL VALIDAT - INDEPLINESTE STANDARDUL GOLD")
else:
    print(f"\n⚠️  Dataset-ul NU indeplineste standardul Gold (SSIM {mean_ssim_passed:.4f} < {required_mean})")
    print(f"  Deficit: {(required_mean - mean_ssim_passed):.4f}")
    print(f"  Pentru a atinge {required_mean}, trebuie sa maresti media cu aproximativ {(required_mean - mean_ssim_passed)/mean_ssim_passed * 100:.1f}%")

print(f"\n{'='*140}\n")

# CSV Reports
print(f"Rapoarte generate:")
print(f"  - audit_passed_perechi.csv ({passed_count} perechi)")
print(f"  - audit_rejected_perechi.csv ({rejected_count} perechi)")
print(f"  - Fisiere respinse in: data/REJECTED_BY_AUDIT/")
