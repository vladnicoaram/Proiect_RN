#!/usr/bin/env python3
"""
Script pentru extragerea și afișarea link-urilor clicabile (file://)
pentru imaginile corupte, cu focus pe top 20 și zona de limită (SSIM 0.38-0.42)
"""

import csv
import os

CSV_FILE = "/Users/admin/Documents/Facultatea/Proiect_RN/corrupted_images_full_list.csv"

def extract_and_display():
    """Extrage și afișează imagini cu link-uri clicabile."""
    
    top_20 = []
    boundary_zone = []  # SSIM între 0.38 și 0.42
    
    # Citește CSV
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ssim = float(row['ssim'])
                
                # Top 20 (cele mai mici SSIM values)
                if len(top_20) < 20:
                    top_20.append(row)
                
                # Boundary zone (SSIM între 0.38 și 0.42)
                if 0.38 <= ssim <= 0.42:
                    boundary_zone.append(row)
            except ValueError:
                continue
    
    # Afișare TOP 20
    print("\n" + "="*100)
    print("🔴 TOP 20 IMAGINI CORUPTE (CELE MAI GRAVE)")
    print("="*100)
    
    for idx, img in enumerate(top_20, 1):
        filename = img['filename']
        dataset = img['dataset']
        ssim = float(img['ssim'])
        hist_corr = float(img['hist_corr'])
        combined = float(img['combined_score'])
        z_score = float(img['z_score'])
        after_path = img['after_path']
        
        print(f"\n{idx}. {filename}")
        print(f"   Dataset: {dataset} | SSIM: {ssim:.4f} | Hist: {hist_corr:.4f} | Z-Score: {z_score:.2f}")
        print(f"\n   🖼️  AFTER (Imagine coruptă):")
        print(f"   file://{after_path}")
        print(f"\n   📋 BEFORE (Referință curată):")
        before_path = img['before_path']
        print(f"   file://{before_path}")
        print("-" * 100)
    
    # Afișare BOUNDARY ZONE
    print("\n" + "="*100)
    print(f"⚠️  ZONA DE LIMITĂ: {len(boundary_zone)} imagini (SSIM 0.38-0.42) - GREU de decizie")
    print("="*100)
    
    for idx, img in enumerate(boundary_zone, 1):
        filename = img['filename']
        dataset = img['dataset']
        ssim = float(img['ssim'])
        hist_corr = float(img['hist_corr'])
        z_score = float(img['z_score'])
        after_path = img['after_path']
        before_path = img['before_path']
        
        print(f"\n{idx}. {filename}")
        print(f"   Dataset: {dataset} | SSIM: {ssim:.4f} | Hist: {hist_corr:.4f} | Z-Score: {z_score:.2f}")
        print(f"\n   🖼️  AFTER:")
        print(f"   file://{after_path}")
        print(f"\n   📋 BEFORE:")
        print(f"   file://{before_path}")
        print("-" * 100)
    
    # Rezumat
    print("\n" + "="*100)
    print("📊 REZUMAT")
    print("="*100)
    print(f"✅ Top 20 imagini corupte: {len(top_20)}")
    print(f"⚠️  Imagini în zona de limită (SSIM 0.38-0.42): {len(boundary_zone)}")
    print(f"\n💡 Instrucțiuni:")
    print(f"   1. Cmd+Click pe link-urile file:// pentru a deschide imaginile în Preview")
    print(f"   2. Compară AFTER (coruptă) cu BEFORE (curată)")
    print(f"   3. Hotărăște dacă aceasta trebuie ștearsă\n")

if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        extract_and_display()
    else:
        print(f"❌ CSV file nu găsit: {CSV_FILE}")
