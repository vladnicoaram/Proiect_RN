import os
import json
import shutil
import random
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append(os.getcwd())
from src.neural_network.dataset import ChangeDetectionDataset
from src.neural_network.model import UNet
from torch.utils.data import DataLoader

def count_mask_pixels(mask_path):
    """Numără pixelii albi (>128) din masca"""
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0
        white_pixels = np.sum(mask > 128)
        return white_pixels
    except:
        return 0

def calculate_iou(pred, target):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()
    return (intersection / (union + 1e-6)).item()

def main():
    print("=" * 80)
    print("🧹 CLEANUP DATASET - Curățare automată & Reanaliza")
    print("=" * 80)
    
    DATA_ROOT = "data/train"
    RESULTS_DIR = "results"
    MODEL_PATH = "models/unet_final.pth"
    
    before_dir = f"{DATA_ROOT}/before"
    after_dir = f"{DATA_ROOT}/after"
    mask_dir = f"{DATA_ROOT}/masks"
    
    # Citește raporturile generate anterior
    print("\n📖 Citire rapoarte...")
    
    with open(f"{RESULTS_DIR}/corrupted_images.json", "r") as f:
        corrupted_list = json.load(f)
        corrupted_files = set(item['filename'] for item in corrupted_list)
    
    print(f"   ✓ Imagini corupte: {len(corrupted_files)}")
    
    # REANALIZA: Procesează TOATE imaginile, nu doar top 50
    print("\n🔄 REANALIZA: Procesez TOATE imaginile din dataset...")
    
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    if os.path.exists(MODEL_PATH):
        model = UNet(6, 1).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
    else:
        print("   ⚠️  Model nu găsit - voi folosi doar pixeli de mască")
        model = None
    
    dataset = ChangeDetectionDataset(root_dir=DATA_ROOT)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_issues = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            filename = dataset.files[i]
            
            if filename in corrupted_files:
                continue  # Skip corupte, vor fi șterse oricum
            
            mask_path = os.path.join(mask_dir, filename)
            white_pixels = count_mask_pixels(mask_path)
            
            if model:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                output = torch.sigmoid(logits)
                iou = calculate_iou(output, y)
            else:
                loss = 0.0
                iou = 0.0
            
            all_issues.append({
                "filename": filename,
                "mask_white_pixels": white_pixels,
                "loss": float(loss) if model else 0.0,
                "iou": float(iou) if model else 0.0,
                "score_error": float(loss) + (1 - iou) if model else 0.0
            })
    
    print(f"   ✓ Analizate {len(all_issues)} imagini valide")
    
    # Sortează după mască (pixeli descrescător)
    all_issues.sort(key=lambda x: x['mask_white_pixels'], reverse=True)
    
    # Categoria 1: Imagini corupte (ȘTERGE INSTANT)
    print("\n🔴 CATEGORIA 1: Imagini corupte fizic...")
    to_delete = set()
    to_delete.update(corrupted_files)
    print(f"   ❌ {len(to_delete)} imagini corupte -> ȘTERGERE IMEDIATĂ")
    
    # Categoria 2: Imagini cu mască mare (>1500 pixeli) - PĂSTREAZĂ
    print("\n🟢 CATEGORIA 2: Imagini cu mască MARE (>1500px)...")
    files_to_keep_big = []
    for item in all_issues:
        if item['mask_white_pixels'] > 1500:
            files_to_keep_big.append(item['filename'])
    
    print(f"   ✓ {len(files_to_keep_big)} imagini MARI (mască >1500px) -> PĂSTREAZĂ")
    
    # Categoria 3: Imagini cu mască goală (<10 pixeli) - selectează 100 aleatorii
    print("\n🟡 CATEGORIA 3: Imagini cu mască GOALĂ (<10px)...")
    files_empty_mask = []
    for item in all_issues:
        if item['mask_white_pixels'] < 10:
            files_empty_mask.append(item['filename'])
    
    print(f"   Găsite {len(files_empty_mask)} imagini cu mască goală")
    
    # Alege 100 aleatorii
    if len(files_empty_mask) > 100:
        random.seed(42)  # Pentru reproducibilitate
        files_keep_empty = random.sample(files_empty_mask, 100)
    else:
        files_keep_empty = files_empty_mask
    
    print(f"   ✓ {len(files_keep_empty)} imagini GOALE aleatorii -> PĂSTREAZĂ (pentru echilibru)")
    print(f"   ❌ {len(files_empty_mask) - len(files_keep_empty)} imagini GOALE -> ȘTERGERE")
    
    # Adaug la lista de ștergere
    to_delete.update(set(files_empty_mask) - set(files_keep_empty))
    
    # Categoria 4: Restul imaginilor cu probleme - ȘTERGE
    print("\n🟠 CATEGORIA 4: Imagini cu modificări mici...")
    files_small_changes = []
    kept_files = set(corrupted_files) | set(files_to_keep_big) | set(files_keep_empty)
    for item in all_issues:
        if item['filename'] not in kept_files:
            files_small_changes.append(item['filename'])
    
    print(f"   ❌ {len(files_small_changes)} imagini cu schimbări mici -> ȘTERGERE")
    to_delete.update(files_small_changes)
    
    # Rezumat
    print("\n" + "=" * 80)
    print("📋 REZUMAT ÎNAINTE DE ȘTERGERE")
    print("=" * 80)
    
    total_before = len(os.listdir(before_dir))
    files_to_keep = total_before - len(to_delete)
    
    print(f"Total imagini ÎNAINTE:  {total_before}")
    print(f"Imagini DE ȘTERS:       {len(to_delete)} ({len(to_delete)/total_before*100:.1f}%)")
    print(f"Imagini DE PĂSTRAT:     {files_to_keep} ({files_to_keep/total_before*100:.1f}%)")
    
    print(f"\n  - Corupte:             {len(corrupted_files)}")
    print(f"  - Mari (>1500px):      {len(files_to_keep_big)}")
    print(f"  - Goale aleatorii:     {len(files_keep_empty)}")
    print(f"  - Cu schimbări mici:   {len(files_small_changes)}")
    
    # Confirmări
    print("\n" + "=" * 80)
    response = input("⚠️  EȘTI SIGUR? Voi ȘTERGE definitiv aceste imagini. (da/nu): ").strip().lower()
    
    if response != "da":
        print("❌ Anulat. Nicio imagine nu a fost ștearsă.")
        return
    
    # ȘTERGERE
    print("\n" + "=" * 80)
    print("🗑️  ȘTERGERE IMAGINI...")
    print("=" * 80)
    
    deleted_count = 0
    for filename in to_delete:
        try:
            before_path = os.path.join(before_dir, filename)
            after_path = os.path.join(after_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            if os.path.exists(before_path):
                os.remove(before_path)
            if os.path.exists(after_path):
                os.remove(after_path)
            if os.path.exists(mask_path):
                os.remove(mask_path)
            
            deleted_count += 1
            if deleted_count % 100 == 0:
                print(f"   ✓ Șters {deleted_count}...")
        except Exception as e:
            print(f"   ⚠️  Eroare la ștergerea {filename}: {e}")
    
    # Verificare finală
    print("\n" + "=" * 80)
    print("✅ ȘTERGERE COMPLETĂ")
    print("=" * 80)
    
    total_after = len(os.listdir(before_dir))
    print(f"Total imagini DUPĂ:     {total_after}")
    print(f"Șterse efectiv:         {total_before - total_after}")
    print(f"Salvare:                {total_after / total_before * 100:.1f}%")
    
    # Salvează raport ștergere
    cleanup_report = {
        "total_before": total_before,
        "total_after": total_after,
        "deleted": len(to_delete),
        "kept_corrupted": len(corrupted_files),
        "kept_big_masks": len(files_to_keep_big),
        "kept_empty_random": len(files_keep_empty),
        "deleted_small_changes": len(files_small_changes),
        "deleted_empty_extra": len(files_empty_mask) - len(files_keep_empty),
    }
    
    with open(f"{RESULTS_DIR}/cleanup_report.json", "w") as f:
        json.dump(cleanup_report, f, indent=4)
    
    print(f"\n📊 Raport salvat: {RESULTS_DIR}/cleanup_report.json")
    
    print("\n" + "=" * 80)
    print("🎉 DATASET CURAT ȘI GATA PENTRU REANTRENARE!")
    print("=" * 80)
    print(f"\n⏭️  Următorul pas: Antrenează modelul cu python src/neural_network/train_clean.py")

if __name__ == "__main__":
    main()
