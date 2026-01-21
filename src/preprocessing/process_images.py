import cv2
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURARE ---
RAW_BEFORE_PATH = 'data/raw/before'
RAW_AFTER_PATH = 'data/raw/after'
PROCESSED_PATH = 'data/processed'
IMG_SIZE = (256, 256)

# Creăm folderele necesare dacă nu există
os.makedirs(f"{PROCESSED_PATH}/before", exist_ok=True)
os.makedirs(f"{PROCESSED_PATH}/after", exist_ok=True)
os.makedirs(f"{PROCESSED_PATH}/masks", exist_ok=True)

def align_images(img_ref, img_target):
    """Aliniază img_target peste img_ref folosind ORB."""
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_target, None)

    if des1 is None or des2 is None:
        return img_target # Returnează originalul dacă nu găsește trăsături

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Păstrăm top 15% potriviri
    good_matches = matches[:int(len(matches) * 0.15)]
    
    if len(good_matches) < 4:
        return img_target # Prea puține puncte pentru aliniere

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    h, w, _ = img_ref.shape
    if M is not None:
        aligned_img = cv2.warpPerspective(img_target, M, (w, h))
        return aligned_img
    
    return img_target

def create_diff_mask(img1, img2):
    """Creează masca binară a diferențelor."""
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return thresh

def run_preprocessing():
    files = os.listdir(RAW_BEFORE_PATH)
    valid_extensions = ('.jpg', '.png', '.jpeg')
    files = [f for f in files if f.lower().endswith(valid_extensions)]

    print(f"Caut perechi pentru {len(files)} imagini 'before'...")

    processed_count = 0
    for fname in tqdm(files):
        p_before = os.path.join(RAW_BEFORE_PATH, fname)
        
        # --- LOGICA DE POTRIVIRE A NUMELOR (FIX) ---
        # 1. Încercăm numele identic
        p_after = os.path.join(RAW_AFTER_PATH, fname)

        # 2. Dacă nu există și avem "empty", încercăm să înlocuim cu "full"
        if not os.path.exists(p_after) and 'empty' in fname:
            fname_after = fname.replace('empty', 'full')
            p_after = os.path.join(RAW_AFTER_PATH, fname_after)
            
        # 3. Verificare finală
        if not os.path.exists(p_after):
            # Nu afișăm eroare pentru fiecare fișier ca să nu umplem consola, doar numărăm
            continue

        try:
            img_b = cv2.imread(p_before)
            img_a = cv2.imread(p_after)

            if img_b is None or img_a is None:
                continue

            # Redimensionare
            img_b = cv2.resize(img_b, IMG_SIZE)
            img_a = cv2.resize(img_a, IMG_SIZE)

            # Aliniere
            img_a_aligned = align_images(img_b, img_a)
            
            # Generare Mască
            mask = create_diff_mask(img_b, img_a_aligned)

            # Salvare
            # Folosim numele original din 'before' pentru consistență
            cv2.imwrite(f"{PROCESSED_PATH}/before/{fname}", img_b)
            cv2.imwrite(f"{PROCESSED_PATH}/after/{fname}", img_a_aligned)
            cv2.imwrite(f"{PROCESSED_PATH}/masks/{fname}", mask)
            
            processed_count += 1

        except Exception as e:
            print(f"Eroare: {e}")

    print(f"\n✅ Gata! Am procesat cu succes {processed_count} perechi.")
    if processed_count == 0:
        print("⚠️ Tot 0? Verifică dacă folderul 'data/raw/after' conține imagini cu 'full' în nume.")

if __name__ == "__main__":
    run_preprocessing()