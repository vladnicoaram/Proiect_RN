#!/usr/bin/env python3
"""
Script pentru detectarea imaginilor corupte/distorsionate din folderul 'after'
Utilizează metoda Laplacian variance pentru a identifica imaginile blurry sau cu zgomot geometric.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Configurare directorio
BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"
AFTER_FOLDERS = [
    "train/after",
    "test/after",
    "validation/after"
]

def laplacian_variance(image_path):
    """
    Calculează Laplacian variance pentru o imagine.
    Valori mici indică blur/distorsiune, valori mari indică claritate.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Cannot read file"
        
        # Calculează Laplacian
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        return variance, "OK"
    except Exception as e:
        return None, str(e)


def get_image_histogram_stats(image_path):
    """
    Analizează histograma unei imagini pentru a detecta artefacte.
    Imaginile corupte au adesea distribuții ciudate ale pixelilor.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # normalize
        
        # Detecta spike-uri neobișnuite (indicator de corrupție)
        max_bin_ratio = np.max(hist) / (np.mean(hist) + 1e-8)
        return max_bin_ratio
    except Exception as e:
        return None


def analyze_corrupted_images(threshold_variance=100, threshold_histogram=5):
    """
    Analizează toate imaginile din folderul 'after' și identifică pe cele corupte.
    
    Args:
        threshold_variance: Laplacian variance sub această valoare = suspect (blur/corupție)
        threshold_histogram: Raport maxim histogram bin - peste aceasta = suspect
    """
    
    corrupted_images = []
    statistics = defaultdict(list)
    
    print("\n" + "="*80)
    print("ANALIZĂ IMAGINI CORUPTE - LAPLACIAN VARIANCE METHOD")
    print("="*80)
    
    for folder in AFTER_FOLDERS:
        folder_path = os.path.join(BASE_DATA_DIR, folder)
        
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder nu există: {folder_path}")
            continue
        
        print(f"\n📁 Scanare: {folder}")
        image_files = sorted([f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"   Total imagini: {len(image_files)}")
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            # Calculează Laplacian variance
            variance, status = laplacian_variance(img_path)
            
            if variance is None:
                continue
            
            # Calculează histogram stats
            hist_ratio = get_image_histogram_stats(img_path)
            
            # Classify as corrupted
            is_corrupted = variance < threshold_variance
            
            statistics[folder].append({
                'path': img_path,
                'filename': img_file,
                'variance': variance,
                'hist_ratio': hist_ratio if hist_ratio else 0,
                'corrupted': is_corrupted
            })
            
            if is_corrupted:
                corrupted_images.append({
                    'path': img_path,
                    'filename': img_file,
                    'folder': folder,
                    'variance': variance,
                    'hist_ratio': hist_ratio if hist_ratio else 0
                })
    
    return corrupted_images, statistics


def display_preview(corrupted_images, max_count=10):
    """
    Afișează o previzualizare a imaginilor corupte detectate.
    """
    print("\n" + "="*80)
    print(f"🔍 IMAGINI CORUPTE DETECTATE (Primele {min(max_count, len(corrupted_images))} din {len(corrupted_images)} total)")
    print("="*80)
    
    if not corrupted_images:
        print("✅ Nu au fost găsite imagini corupte!")
        return
    
    # Sort by variance (cele mai blurry first)
    sorted_images = sorted(corrupted_images, key=lambda x: x['variance'])
    
    for idx, img in enumerate(sorted_images[:max_count], 1):
        print(f"\n{idx}. CALE COMPLETĂ:")
        print(f"   {img['path']}")
        print(f"   Folder: {img['folder']}")
        print(f"   Laplacian Variance: {img['variance']:.2f} (SCĂZUT = blur/corupție)")
        print(f"   Histogram Ratio: {img['hist_ratio']:.2f}")
        
        # Construire nume pereche (before/masks)
        after_folder = os.path.dirname(img['path'])
        base_folder = os.path.dirname(after_folder)  # train/test/validation
        filename = img['filename']
        
        before_path = os.path.join(base_folder, 'before', filename)
        mask_path = os.path.join(base_folder, 'masks', filename)
        mask_clean_path = os.path.join(base_folder, 'masks_clean', filename)
        
        print(f"   Fișiere asociate care vor fi șterse:")
        print(f"     - before: {before_path}")
        if os.path.exists(mask_path):
            print(f"     - masks: {mask_path}")
        if os.path.exists(mask_clean_path):
            print(f"     - masks_clean: {mask_clean_path}")
    
    print("\n" + "="*80)
    print(f"TOTAL IMAGINI DE ȘTERS: {len(corrupted_images)}")
    print(f"  - Foldere afectate:")
    folders_count = defaultdict(int)
    for img in corrupted_images:
        folders_count[img['folder']] += 1
    for folder, count in sorted(folders_count.items()):
        print(f"    * {folder}: {count} imagini")
    print("="*80)
    
    return sorted_images[:max_count]


def get_statistics(statistics):
    """
    Afișează statistici generale despre variante Laplacian.
    """
    print("\n" + "="*80)
    print("📊 STATISTICI GENERALE")
    print("="*80)
    
    for folder, stats_list in sorted(statistics.items()):
        if not stats_list:
            continue
        
        variances = [s['variance'] for s in stats_list]
        corrupted_count = sum(1 for s in stats_list if s['corrupted'])
        
        print(f"\n{folder}:")
        print(f"  Total imagini: {len(stats_list)}")
        print(f"  Imagini corupte: {corrupted_count} ({100*corrupted_count/len(stats_list):.1f}%)")
        print(f"  Laplacian Variance:")
        print(f"    Min: {min(variances):.2f}")
        print(f"    Max: {max(variances):.2f}")
        print(f"    Mean: {np.mean(variances):.2f}")
        print(f"    Median: {np.median(variances):.2f}")


if __name__ == "__main__":
    print("\n🚀 Inițiare detecție imagini corupte...\n")
    
    # Analiza
    corrupted_images, statistics = analyze_corrupted_images(
        threshold_variance=100,  # Sub 100 = suspect
        threshold_histogram=5    # Raport > 5 = suspect
    )
    
    # Statistici
    get_statistics(statistics)
    
    # Previzualizare
    preview = display_preview(corrupted_images, max_count=10)
    
    # Salvare raport
    report_path = os.path.join(BASE_DATA_DIR, "../corrupted_images_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"RAPORT IMAGINI CORUPTE - {len(corrupted_images)} detectate\n")
        f.write("="*80 + "\n\n")
        for img in corrupted_images:
            f.write(f"{img['path']}\n")
    
    print(f"\n✅ Raport salvat în: {report_path}")
    print("\n⏸️  AȘTEPT CONFIRMAREA DVSTRĂ!")
    print("   Verificați previzualizarea și confirmați înainte de ștergere.\n")
