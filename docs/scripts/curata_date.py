import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import sys
import cv2
import shutil
from pathlib import Path

# Adăugăm folderul curent în path pentru a găsi folderul 'src'
sys.path.append(os.getcwd())

# Importurile tale corectate conform structurii din VS Code
from src.neural_network.dataset import ChangeDetectionDataset 
from src.neural_network.model import UNet

def calculate_iou(pred, target):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()
    return (intersection / (union + 1e-6)).item()

def check_image_corruption(image_path):
    """
    Verifica dacă o imagine poate fi citită corect cu OpenCV
    Returnează: (is_valid: bool, reason: str)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Nu poate fi citita de OpenCV (None)"
        
        if img.size == 0:
            return False, "Imagine goala (size=0)"
        
        # Verific dacă imaginea are dimensiuni rezonabile
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return False, f"Prea mica: {w}x{h}"
        
        # Verific dacă pixelii sunt valid (nu toti = 0 sau 255)
        mean_val = np.mean(img)
        if mean_val < 5 or mean_val > 250:
            return False, f"Imagine uniform (media={mean_val:.1f})"
        
        return True, "OK"
    except Exception as e:
        return False, f"Exceptie: {str(e)[:50]}"

def count_mask_pixels(mask_path):
    """
    Numără pixelii albi (>128) din masca
    """
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0
        white_pixels = np.sum(mask > 128)
        return white_pixels
    except:
        return 0

def get_filename_from_loader(dataset, idx):
    """
    Extrage filename din dataset (pe baza folderului before/)
    """
    return dataset.files[idx]

def main():
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    MODEL_PATH = "models/unet_final.pth"
    DATA_ROOT = "data/train"
    
    print("=" * 80)
    print("🔍 AUDIT DATASET - Identificare date proaste din training set")
    print("=" * 80)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Nu am găsit modelul la {MODEL_PATH}")
        return

    # Încărcare model
    print("\n📦 Încărcare model...")
    model = UNet(6, 1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Analizăm folderul de TRAIN pentru a găsi pozele proaste
    print(f"📂 Citire dataset din {DATA_ROOT}...")
    dataset = ChangeDetectionDataset(root_dir=DATA_ROOT) 
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    
    # Vor stoca: (indice, filename, loss, iou, mask_pixels, motiv)
    issues_list = []
    corruption_issues = []
    
    before_dir = f"{DATA_ROOT}/before"
    after_dir = f"{DATA_ROOT}/after"
    mask_dir = f"{DATA_ROOT}/masks"

    print(f"\n🚀 Analiza celor {len(loader)} imagini din TRAIN...\n")
    
    # Fase 1: Verifica corupția fizică a fișierelor
    print("Fase 1/3: Detectare imagini corupte...")
    for idx, filename in enumerate(dataset.files):
        before_path = os.path.join(before_dir, filename)
        after_path = os.path.join(after_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        # Verifica before
        is_valid, reason = check_image_corruption(before_path)
        if not is_valid:
            corruption_issues.append({
                "index": idx,
                "filename": filename,
                "tip": "before",
                "motiv": reason
            })
            continue
        
        # Verifica after
        is_valid, reason = check_image_corruption(after_path)
        if not is_valid:
            corruption_issues.append({
                "index": idx,
                "filename": filename,
                "tip": "after",
                "motiv": reason
            })
            continue
        
        # Verifica mask
        is_valid, reason = check_image_corruption(mask_path)
        if not is_valid:
            corruption_issues.append({
                "index": idx,
                "filename": filename,
                "tip": "mask",
                "motiv": reason
            })
    
    if corruption_issues:
        print(f"   ⚠️  Găsite {len(corruption_issues)} imagini corupte fizic!")
    else:
        print(f"   ✓ Toate imaginile pot fi citite corect")
    
    # Fase 2: Verifică Loss, IoU și masca de teren
    print("\nFase 2/3: Analiza Loss, IoU și măștilor...")
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            filename = dataset.files[i]
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            logits = model(x)
            loss = criterion(logits, y)
            output = torch.sigmoid(logits)
            
            iou = calculate_iou(output, y)
            
            # Numără pixelii albi din masca (Ground Truth)
            mask_path = os.path.join(mask_dir, filename)
            white_pixels = count_mask_pixels(mask_path)
            
            motiv = []
            
            # Detecta probleme
            if loss.item() > 0.4:
                motiv.append(f"Loss mare ({loss.item():.4f})")
            if iou < 0.05:
                motiv.append(f"IoU scăzut ({iou:.4f})")
            if white_pixels < 100:
                motiv.append(f"Masca prea mica ({white_pixels} px)")
            
            # Dacă are cel puțin o problemă, adaug la lista
            if motiv:
                issues_list.append({
                    "index": i,
                    "filename": filename,
                    "loss": round(loss.item(), 4),
                    "iou": round(iou, 4),
                    "mask_white_pixels": int(white_pixels),
                    "motive": " | ".join(motiv),
                    "score_error": round(loss.item() + (1 - iou), 4)  # Scor combinat
                })
    
    # Sortează după scor de eroare descrescător
    issues_list.sort(key=lambda x: x['score_error'], reverse=True)
    
    print(f"   ✓ Găsite {len(issues_list)} imagini cu probleme potențiale")
    
    # Fase 3: Copiază primele 20 imagini problematice în folder de review
    print("\nFase 3/3: Copiare imagini pentru review manual...")
    review_dir = "results/to_check"
    os.makedirs(review_dir, exist_ok=True)
    
    # Curăță folderul dacă există deja
    for existing_file in Path(review_dir).glob("*"):
        if existing_file.is_file():
            existing_file.unlink()
    
    # Copiază primele 20
    num_to_copy = min(20, len(issues_list))
    for rank, issue in enumerate(issues_list[:num_to_copy], 1):
        filename = issue['filename']
        idx = issue['index']
        
        before_src = os.path.join(before_dir, filename)
        after_src = os.path.join(after_dir, filename)
        mask_src = os.path.join(mask_dir, filename)
        
        # Creează subfolder pentru fiecare pereche
        pair_dir = os.path.join(review_dir, f"{rank:02d}_{filename.split('.')[0]}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Copiază imagini
        if os.path.exists(before_src):
            shutil.copy(before_src, os.path.join(pair_dir, "before.png"))
        if os.path.exists(after_src):
            shutil.copy(after_src, os.path.join(pair_dir, "after.png"))
        if os.path.exists(mask_src):
            shutil.copy(mask_src, os.path.join(pair_dir, "mask.png"))
        
        # Creează fișier info
        info_text = f"""RANK: {rank}
Filename: {filename}
Index: {idx}

PROBLEME:
{issue['motive']}

METRICI:
- Loss: {issue['loss']}
- IoU: {issue['iou']}
- Pixeli albi masca: {issue['mask_white_pixels']}
- Score Error (combinat): {issue['score_error']}
"""
        with open(os.path.join(pair_dir, "INFO.txt"), "w") as f:
            f.write(info_text)
        
        print(f"   {rank:2d}. Copiat: {filename} (Loss={issue['loss']}, IoU={issue['iou']}, Masca={issue['mask_white_pixels']}px)")
    
    # Salvează rapoarte JSON
    print(f"\n📊 Salvare rapoarte...")
    os.makedirs("results", exist_ok=True)
    
    # Raport imagini problematice
    with open("results/problematic_images.json", "w") as f:
        json.dump(issues_list[:50], f, indent=4)  # Top 50
    
    # Raport imagini corupte
    if corruption_issues:
        with open("results/corrupted_images.json", "w") as f:
            json.dump(corruption_issues, f, indent=4)
    
    # Raport rezumat
    summary = {
        "total_images": len(dataset),
        "corrupted_count": len(corruption_issues),
        "problematic_count": len(issues_list),
        "top_20_copied": num_to_copy,
        "statistics": {
            "avg_loss": round(np.mean([x['loss'] for x in issues_list]) if issues_list else 0, 4),
            "avg_iou": round(np.mean([x['iou'] for x in issues_list]) if issues_list else 0, 4),
            "avg_mask_pixels": int(np.mean([x['mask_white_pixels'] for x in issues_list]) if issues_list else 0),
        }
    }
    
    with open("results/audit_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    # Printează rezumat final
    print("\n" + "=" * 80)
    print("📋 REZUMAT AUDIT")
    print("=" * 80)
    print(f"Total imagini analizate:     {summary['total_images']}")
    print(f"Imagini corupte (fizic):     {summary['corrupted_count']}")
    print(f"Imagini cu probleme:         {summary['problematic_count']}")
    print(f"Copiate pentru review:       {summary['top_20_copied']} (în {review_dir})")
    
    if issues_list:
        print(f"\nStatistici problematice:")
        print(f"  - Loss mediu:              {summary['statistics']['avg_loss']}")
        print(f"  - IoU mediu:               {summary['statistics']['avg_iou']}")
        print(f"  - Pixeli masca (mediu):    {summary['statistics']['avg_mask_pixels']}")
    
    print("\n✅ Analiză gata!")
    print(f"📁 Rezultate salvate în: results/")
    print(f"   - results/problematic_images.json (top 50)")
    print(f"   - results/corrupted_images.json")
    print(f"   - results/audit_summary.json")
    print(f"   - results/to_check/ (vizualizare 20 perechi)")

if __name__ == "__main__":
    main()