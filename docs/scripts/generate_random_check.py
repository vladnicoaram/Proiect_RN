import os
import random
import shutil
import cv2
import numpy as np
from pathlib import Path

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

def main():
    print("=" * 80)
    print("🎲 GENERATOR - Random Check Dataset")
    print("=" * 80)
    
    DATA_ROOT = "data/train"
    CHECK_DIR = "results/random_check"
    
    before_dir = f"{DATA_ROOT}/before"
    after_dir = f"{DATA_ROOT}/after"
    mask_dir = f"{DATA_ROOT}/masks"
    
    # Verifică dacă directoarele există
    if not all(os.path.isdir(d) for d in [before_dir, after_dir, mask_dir]):
        print("❌ Directoarele data/train nu sunt complete")
        return
    
    print("\n📂 Citire fișiere...")
    
    # Citește fișierele din fiecare director
    before_files = set(os.listdir(before_dir))
    after_files = set(os.listdir(after_dir))
    mask_files = set(os.listdir(mask_dir))
    
    print(f"   Before: {len(before_files)} fișiere")
    print(f"   After:  {len(after_files)} fișiere")
    print(f"   Masks:  {len(mask_files)} fișiere")
    
    # Găsește fișierele comune în toate 3 directoare
    common_files = before_files & after_files & mask_files
    common_files = sorted(list(common_files))
    
    print(f"   ✓ Fișiere în toate 3 directoare: {len(common_files)}")
    
    if len(common_files) < 50:
        print(f"❌ Doar {len(common_files)} fișiere disponibile, dar trebuie 50!")
        return
    
    # Selectează 50 random
    print("\n🎲 Selecție 50 fișiere aleatorii...")
    random.seed(42)  # Pentru reproducibilitate
    selected_files = random.sample(common_files, 50)
    selected_files = sorted(selected_files)
    
    print(f"   ✓ {len(selected_files)} fișiere selectate")
    
    # Creează folderul principal
    print(f"\n📁 Creare structură foldere...")
    if os.path.exists(CHECK_DIR):
        print(f"   ⚠️  Folderul {CHECK_DIR} există deja - șterg...")
        shutil.rmtree(CHECK_DIR)
    
    os.makedirs(CHECK_DIR, exist_ok=True)
    
    # Copiază fișierele și generează info
    print(f"\n📋 Copiare fișiere și generare info...")
    
    for rank, filename in enumerate(selected_files, 1):
        # Construiește calea subfolder
        name_without_ext = os.path.splitext(filename)[0]
        subfolder_name = f"{rank:02d}_{name_without_ext}"
        subfolder_path = os.path.join(CHECK_DIR, subfolder_name)
        
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Caile sursă
        before_src = os.path.join(before_dir, filename)
        after_src = os.path.join(after_dir, filename)
        mask_src = os.path.join(mask_dir, filename)
        
        # Caile destinație (renumite la .png)
        before_dst = os.path.join(subfolder_path, "before.png")
        after_dst = os.path.join(subfolder_path, "after.png")
        mask_dst = os.path.join(subfolder_path, "mask.png")
        
        # Copiază fișiere
        try:
            shutil.copy(before_src, before_dst)
            shutil.copy(after_src, after_dst)
            shutil.copy(mask_src, mask_dst)
            
            # Numără pixeli din mască
            white_pixels = count_mask_pixels(mask_src)
            
            # Creează info.txt
            info_text = f"""RANDOM CHECK SAMPLE #{rank}
{'='*50}
Filename: {filename}
Subfolder: {subfolder_name}

MASK STATISTICS:
- White pixels (>128): {white_pixels}
- File extension: {os.path.splitext(filename)[1]}

FILES:
- before.png: ✓
- after.png:  ✓
- mask.png:   ✓
"""
            info_path = os.path.join(subfolder_path, "info.txt")
            with open(info_path, "w") as f:
                f.write(info_text)
            
            print(f"   {rank:2d}. {subfolder_name} ({white_pixels} px albi)")
            
        except Exception as e:
            print(f"   ❌ Eroare la copiere {filename}: {e}")
    
    # Rezumat
    print("\n" + "=" * 80)
    print("✅ GENERARE COMPLETĂ")
    print("=" * 80)
    print(f"50 foldere create în: {CHECK_DIR}/")
    print(f"Fiecare folder conține:")
    print(f"  - before.png")
    print(f"  - after.png")
    print(f"  - mask.png")
    print(f"  - info.txt (cu statistici)")
    
    print("\n📊 Statistici finale:")
    print(f"   Total foldere: 50")
    print(f"   Locație: {os.path.abspath(CHECK_DIR)}")
    
    print("\n🎯 Recomandație:")
    print(f"   Deschide {CHECK_DIR} și verifica manual imaginile")
    print(f"   Aceștea sunt DATE ALEATORII din dataset-ul curat!")

if __name__ == "__main__":
    main()
