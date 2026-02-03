#!/usr/bin/env python3
"""
SCRIPT AUDIT DE CERTIFICARE - Validare Dataset Curat
Verificare stricta: Sincronizare + Calitate (SSIM, ORB, Histogram)
"""

import os
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io

BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"
REJECTED_DIR = os.path.join(BASE_DATA_DIR, "REJECTED_BY_AUDIT")

# Thresholds Gold Standard
SSIM_THRESHOLD = 0.70
ORB_MATCHES_THRESHOLD = 25
HISTOGRAM_CORR_THRESHOLD = 0.60

# Required average for validation
REQUIRED_MEAN_SSIM = 0.85

datasets = ['train', 'test', 'validation']

print("\n" + "="*140)
print("SCRIPT AUDIT DE CERTIFICARE - VALIDARE DATASET CURAT (414 PERECHI)")
print("="*140)

# ============================================================================
# PHASE 1: VERIFICARE SINCRONIZARE (HARD CHECK)
# ============================================================================

print(f"\n{'='*140}")
print("PHASE 1: VERIFICARE SINCRONIZARE (HARD CHECK)")
print("="*140)

sync_check_passed = True

for dataset_name in datasets:
    dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
    
    before_dir = os.path.join(dataset_path, 'before')
    after_dir = os.path.join(dataset_path, 'after')
    masks_dir = os.path.join(dataset_path, 'masks')
    masks_clean_dir = os.path.join(dataset_path, 'masks_clean') if dataset_name == 'train' else None
    
    before_files = set(os.listdir(before_dir)) if os.path.exists(before_dir) else set()
    after_files = set(os.listdir(after_dir)) if os.path.exists(after_dir) else set()
    masks_files = set(os.listdir(masks_dir)) if os.path.exists(masks_dir) else set()
    masks_clean_files = set(os.listdir(masks_clean_dir)) if masks_clean_dir and os.path.exists(masks_clean_dir) else set()
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  Before: {len(before_files)} fisiere")
    print(f"  After: {len(after_files)} fisiere")
    print(f"  Masks: {len(masks_files)} fisiere")
    if dataset_name == 'train':
        print(f"  Masks_clean: {len(masks_clean_files)} fisiere")
    
    # Verificare strict
    if before_files != after_files:
        print(f"  ✗ EROARE CRITICA: Before != After")
        sync_check_passed = False
    elif before_files != masks_files:
        print(f"  ✗ EROARE CRITICA: Before != Masks")
        sync_check_passed = False
    elif dataset_name == 'train' and before_files != masks_clean_files:
        print(f"  ✗ EROARE CRITICA: Before != Masks_clean (train)")
        sync_check_passed = False
    else:
        print(f"  ✓ Sincronizare PERFECTA")

if not sync_check_passed:
    print(f"\n{'='*140}")
    print("✗✗✗ EROARE CRITICA: SINCRONIZARE ESUATA - SCRIPT OPRIT")
    print("="*140 + "\n")
    exit(1)

print(f"\n{'='*140}")
print("✓ PHASE 1 PASSED: Sincronizare PERFECTA pe toate dataset-urile")
print("="*140)

# ============================================================================
# PHASE 2: TEST CALITATE - GOLD STANDARD
# ============================================================================

print(f"\n{'='*140}")
print("PHASE 2: TEST CALITATE - GOLD STANDARD")
print(f"Thresholds: SSIM > {SSIM_THRESHOLD}, ORB Matches > {ORB_MATCHES_THRESHOLD}, Histogram > {HISTOGRAM_CORR_THRESHOLD}")
print("="*140)

# Create REJECTED_BY_AUDIT folder
if not os.path.exists(REJECTED_DIR):
    os.makedirs(REJECTED_DIR)
    for ds in datasets:
        for subfolder in ['before', 'after', 'masks'] + (['masks_clean'] if ds == 'train' else []):
            os.makedirs(os.path.join(REJECTED_DIR, ds, subfolder), exist_ok=True)

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=500)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def extract_orb_features(image_path):
    """Extract ORB features from image"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        keypoints, descriptors = orb.detectAndCompute(img, None)
        return keypoints, descriptors
    except Exception as e:
        return None, None

def calculate_good_matches(desc1, desc2):
    """Calculate number of good matches between two descriptor sets"""
    if desc1 is None or desc2 is None:
        return 0
    try:
        matches = bf_matcher.knnMatch(desc1, desc2, k=2)
        if len(matches) == 0:
            return 0
        # Apply Lowe's ratio test
        good_matches = 0
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches += 1
        return good_matches
    except Exception as e:
        return 0

def calculate_ssim(img1_path, img2_path):
    """Calculate SSIM between two images"""
    try:
        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)
        
        # Resize to same size if needed
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        ssim_score = ssim(img1, img2)
        return ssim_score
    except Exception as e:
        return 0.0

def calculate_histogram_correlation(img1_path, img2_path):
    """Calculate histogram correlation between two images"""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize to same size
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation
    except Exception as e:
        return 0.0

# Collect all results
results = []
rejected = []
total_pairs = 0

for dataset_name in datasets:
    dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
    before_dir = os.path.join(dataset_path, 'before')
    after_dir = os.path.join(dataset_path, 'after')
    masks_dir = os.path.join(dataset_path, 'masks')
    
    before_files = sorted(os.listdir(before_dir))
    
    for idx, filename in enumerate(before_files, 1):
        total_pairs += 1
        
        before_path = os.path.join(before_dir, filename)
        after_path = os.path.join(after_dir, filename)
        mask_path = os.path.join(masks_dir, filename)
        
        # Calculate metrics
        ssim_score = calculate_ssim(before_path, after_path)
        
        # ORB matching
        kp1, desc1 = extract_orb_features(before_path)
        kp2, desc2 = extract_orb_features(after_path)
        orb_matches = calculate_good_matches(desc1, desc2)
        
        # Histogram correlation
        hist_corr = calculate_histogram_correlation(before_path, after_path)
        
        # Check thresholds
        passed = (ssim_score >= SSIM_THRESHOLD and 
                 orb_matches >= ORB_MATCHES_THRESHOLD and 
                 hist_corr >= HISTOGRAM_CORR_THRESHOLD)
        
        result = {
            'dataset': dataset_name,
            'filename': filename,
            'ssim': ssim_score,
            'orb_matches': orb_matches,
            'histogram_corr': hist_corr,
            'passed': passed
        }
        results.append(result)
        
        if not passed:
            rejected.append(result)
            # Move files to REJECTED folder
            rejected_dataset_path = os.path.join(REJECTED_DIR, dataset_name)
            shutil.copy(before_path, os.path.join(rejected_dataset_path, 'before', filename))
            shutil.copy(after_path, os.path.join(rejected_dataset_path, 'after', filename))
            shutil.copy(mask_path, os.path.join(rejected_dataset_path, 'masks', filename))
            if dataset_name == 'train':
                masks_clean_path = os.path.join(dataset_path, 'masks_clean', filename)
                if os.path.exists(masks_clean_path):
                    shutil.copy(masks_clean_path, os.path.join(rejected_dataset_path, 'masks_clean', filename))

print(f"\nProcessare: {total_pairs} perechi analizate")

# ============================================================================
# PHASE 3: RAPORTARE REZULTATE
# ============================================================================

print(f"\n{'='*140}")
print("PHASE 3: RAPORTARE REZULTATE")
print("="*140)

passed_count = len([r for r in results if r['passed']])
rejected_count = len(rejected)

print(f"\nRezumat:")
print(f"  Perechi PASSED: {passed_count}/{total_pairs} ({(passed_count/total_pairs)*100:.1f}%)")
print(f"  Perechi REJECTED: {rejected_count}/{total_pairs} ({(rejected_count/total_pairs)*100:.1f}%)")

if rejected_count > 0:
    print(f"\n⚠️  PERECHI RESPINSE (mutate in {REJECTED_DIR}):")
    
    # Group by reason
    by_ssim = [r for r in rejected if r['ssim'] < SSIM_THRESHOLD]
    by_orb = [r for r in rejected if r['orb_matches'] < ORB_MATCHES_THRESHOLD]
    by_hist = [r for r in rejected if r['histogram_corr'] < HISTOGRAM_CORR_THRESHOLD]
    
    print(f"  - SSIM < {SSIM_THRESHOLD}: {len(by_ssim)} perechi")
    print(f"  - ORB matches < {ORB_MATCHES_THRESHOLD}: {len(by_orb)} perechi")
    print(f"  - Histogram < {HISTOGRAM_CORR_THRESHOLD}: {len(by_hist)} perechi")
    
    # Show worst ones
    print(f"\n  Top 10 WORST (lowest scores):")
    sorted_rejected = sorted(rejected, key=lambda x: (x['ssim'], x['orb_matches'], x['histogram_corr']))
    for i, r in enumerate(sorted_rejected[:10], 1):
        print(f"    {i}. {r['dataset']}/{r['filename']}")
        print(f"       SSIM: {r['ssim']:.4f} | ORB: {r['orb_matches']} | Hist: {r['histogram_corr']:.4f}")

# ============================================================================
# PHASE 4: STATISTICI FINALE
# ============================================================================

print(f"\n{'='*140}")
print("PHASE 4: STATISTICI FINALE - GOLD STANDARD")
print("="*140)

passed_results = [r for r in results if r['passed']]

if len(passed_results) > 0:
    mean_ssim = np.mean([r['ssim'] for r in passed_results])
    std_ssim = np.std([r['ssim'] for r in passed_results])
    min_ssim = np.min([r['ssim'] for r in passed_results])
    max_ssim = np.max([r['ssim'] for r in passed_results])
    
    mean_orb = np.mean([r['orb_matches'] for r in passed_results])
    std_orb = np.std([r['orb_matches'] for r in passed_results])
    min_orb = np.min([r['orb_matches'] for r in passed_results])
    max_orb = np.max([r['orb_matches'] for r in passed_results])
    
    mean_hist = np.mean([r['histogram_corr'] for r in passed_results])
    std_hist = np.std([r['histogram_corr'] for r in passed_results])
    min_hist = np.min([r['histogram_corr'] for r in passed_results])
    max_hist = np.max([r['histogram_corr'] for r in passed_results])
    
    print(f"\nSSIM SCORE:")
    print(f"  Mean: {mean_ssim:.4f} {'✓ PASS' if mean_ssim >= REQUIRED_MEAN_SSIM else '✗ FAIL'} (Threshold: {REQUIRED_MEAN_SSIM})")
    print(f"  Std Dev: {std_ssim:.4f}")
    print(f"  Range: [{min_ssim:.4f}, {max_ssim:.4f}]")
    
    print(f"\nORB MATCHES:")
    print(f"  Mean: {mean_orb:.1f}")
    print(f"  Std Dev: {std_orb:.1f}")
    print(f"  Range: [{min_orb:.0f}, {max_orb:.0f}]")
    
    print(f"\nHISTOGRAM CORRELATION:")
    print(f"  Mean: {mean_hist:.4f}")
    print(f"  Std Dev: {std_hist:.4f}")
    print(f"  Range: [{min_hist:.4f}, {max_hist:.4f}]")
    
    # Overall validation
    print(f"\n{'='*140}")
    if mean_ssim >= REQUIRED_MEAN_SSIM:
        print(f"✓✓✓ DATASET VALIDAT - CALITATE AURIE (GOLD STANDARD)")
        print(f"Mean SSIM: {mean_ssim:.4f} >= {REQUIRED_MEAN_SSIM}")
    else:
        print(f"✗ DATASET NU INDEPLINESTE STANDARDUL")
        print(f"Mean SSIM: {mean_ssim:.4f} < {REQUIRED_MEAN_SSIM}")
    print("="*140)
else:
    print("\n✗ Toate perechile au fost respinse!")

# ============================================================================
# GENEREAZA RAPORT CSV
# ============================================================================

print(f"\n{'='*140}")
print("GENEREAZA RAPOARTE CSV")
print("="*140)

# Raport perechi PASSED
csv_passed = "/Users/admin/Documents/Facultatea/Proiect_RN/audit_passed_perechi.csv"
with open(csv_passed, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'filename', 'ssim', 'orb_matches', 'histogram_corr', 'passed'])
    writer.writeheader()
    for r in sorted(passed_results, key=lambda x: x['ssim'], reverse=True):
        writer.writerow(r)

print(f"\n✓ Raport PASSED: {csv_passed} ({len(passed_results)} perechi)")

# Raport perechi REJECTED
if rejected_count > 0:
    csv_rejected = "/Users/admin/Documents/Facultatea/Proiect_RN/audit_rejected_perechi.csv"
    with open(csv_rejected, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'filename', 'ssim', 'orb_matches', 'histogram_corr', 'passed'])
        writer.writeheader()
        for r in sorted(rejected, key=lambda x: (x['ssim'], x['orb_matches']), reverse=True):
            writer.writerow(r)
    
    print(f"✓ Raport REJECTED: {csv_rejected} ({rejected_count} perechi)")
    print(f"  Locatie foldere: {REJECTED_DIR}/")

print(f"\n{'='*140}\n")
