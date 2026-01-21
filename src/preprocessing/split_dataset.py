import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURARE ---
PROCESSED_DIR = 'data/processed'
DATA_DIR = 'data'
SEED = 42

def split_data():
    # Luăm lista de fișiere procesate
    # (ne uităm în folderul 'before', știind că în 'after' și 'masks' au aceleași nume)
    files = os.listdir(f"{PROCESSED_DIR}/before")
    # Filtrăm să fie doar imagini
    files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(files) == 0:
        print("EROARE: Nu sunt imagini în data/processed. Verifică pasul anterior!")
        return

    print(f"Am găsit {len(files)} perechi procesate. Încep împărțirea...")

    # Pasul 1: Separăm 70% Train și 30% restul (Temp)
    train_files, test_val_files = train_test_split(files, test_size=0.3, random_state=SEED)
    
    # Pasul 2: Din cei 30% rămași, împărțim în jumătate (15% Val, 15% Test)
    val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=SEED)

    splits = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }

    # Copierea efectivă a fișierelor
    for split_name, split_files in splits.items():
        print(f"--> Generez setul '{split_name}' ({len(split_files)} imagini)...")
        
        # Facem folderele (ex: data/train/before, data/train/masks...)
        for subtype in ['before', 'after', 'masks']:
            os.makedirs(f"{DATA_DIR}/{split_name}/{subtype}", exist_ok=True)
        
        for f in tqdm(split_files):
            # Copiem Before
            shutil.copy(f"{PROCESSED_DIR}/before/{f}", f"{DATA_DIR}/{split_name}/before/{f}")
            # Copiem After
            shutil.copy(f"{PROCESSED_DIR}/after/{f}", f"{DATA_DIR}/{split_name}/after/{f}")
            # Copiem Mask
            shutil.copy(f"{PROCESSED_DIR}/masks/{f}", f"{DATA_DIR}/{split_name}/masks/{f}")

    print("\n✅ Gata! Datele au fost împărțite în Train, Validation și Test.")
    print(f"Total: {len(files)} -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

if __name__ == "__main__":
    split_data()