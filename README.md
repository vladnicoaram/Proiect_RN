# 📘 **README – Etapa 3: Analiza și Pregătirea Setului de Date (Vlad-Mihai Nicoară)**

## Proiect: *Compararea și Detectarea Schimbărilor în Imagini – Sala de laborator*

---

# 1. Structura Repository-ului (Etapa 3)

```
project-change-detection/
├── README.md
├── docs/
│   └── datasets/
├── data/
│   ├── raw/
│   │   ├── before/        # imagini "înainte"
│   │   └── after/         # imagini "după"
│   ├── pairs/             # imagini A-B deja formate în perechi
│   ├── processed/         # imagini normalizate, aliniate, 256x256
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── preprocessing/
│   │   ├── align.py
│   │   ├── preprocess_images.py
│   │   └── pair_generator.py
│   └── neural_network/
│       └── siamese_unet.py
├── config/
│   └── preprocessing_config.yaml
└── requirements.txt
```

---

# 2. Descrierea Setului de Date

## 2.1 Originea datelor

Datasetul tău este **generat**, deoarece nu ai încă imagini reale.
Tipul datelor: imagini JPEG/PNG.

### Set constituit astfel:

* Poze simulare/placeholder cu o sală de laborator (descărcate sau generate)
* Pentru fiecare scenă:

  * **Before**: imagine la începutul orei
  * **After**: imagine la finalul orei, cu 1–3 modificări introduse manual
    (obiect adăugat, scaun mutat, monitor deplasat etc.)

## 2.2 Caracteristicile dataset-ului

| Caracteristică         | Descriere                                                          |
| ---------------------- | ------------------------------------------------------------------ |
| Tip date               | imagini RGB                                                        |
| Rezoluție              | variabilă → rescalată la 256×256                                   |
| Perechi                | before/after                                                       |
| Dimensiune recomandată | min. 200–500 perechi                                               |
| Format                 | PNG / JPG                                                          |
| Tip etichetă           | mască diferențe (pentru UNet) / scor de diferență (pentru Siamese) |

## 2.3 Structura unei observații

Fiecare sample = **pair(A, B)**

* A = imagine înainte
* B = imagine după
* y_mask = masca diferențelor
* y_score = un scor ∈ [0,1] reprezentând nivelul schimbării

---

# 3. Analiza Exploratorie a Datelor (EDA)

Imaginile nu au statistici tabulare, deci EDA se face astfel:

### ✔ 3.1 Analiză cantitativă

* număr imagini before / after
* dimensiuni originale
* canale culori
* histograme intensități

### ✔ 3.2 Analiză calitate date

* variații mari de iluminare
* imagini nealiniate
* blur / zgomot
* diferențe de perspectivă

### ✔ 3.3 Probleme identificate

* imaginile before/after pot fi făcute din unghiuri diferite
* iluminarea afectează detectarea schimbărilor
* este necesară **aliniere automată (feature matching + warp)**
* datasetul generat este mic → risc overfitting

---

# 4. Preprocesarea Datelor

## ✔ 4.1 Curățarea

* eliminarea imaginilor corupte
* uniformizarea dimensiunilor (256 × 256)
* corecție iluminare
* conversie RGB

## ✔ 4.2 Aliniere imagini

Folosim OpenCV + ORB/SIFT pentru:

```
detect keypoints → match → homography → warp "after" → aligned_B
```

## ✔ 4.3 Generarea etichetelor

Diferențele se extrag prin:

```
gray(A) – gray(B_aligned) → threshold → mask
```

## ✔ 4.4 Normalizare

* valori pixel → [0,1]
* optional augmentări:

  * flip, rotate, brightness jitter

## ✔ 4.5 Split seturi

```
70% train  
15% validation  
15% test
```

Splitul se aplică **pe perechi**, nu individual pe imagini.

---

# 5. Fișiere Generat În Această Etapă

* `data/raw/before/*.jpg`
* `data/raw/after/*.jpg`
* `data/pairs/*` – perechi aliniate
* `data/processed/*` – imagini normalizate
* `data/train/*`, `data/validation/*`, `data/test/*`
* `preprocessing_config.yaml`
* scripturile Python din `src/preprocessing/`

---

# 6. Cod necesar pentru preprocesare

## **align.py** – alinierea imaginilor

```python
import cv2
import numpy as np

def align_images(imgA, imgB):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(grayA, None)
    kp2, des2 = orb.detectAndCompute(grayB, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    ptsA = np.float32([kp1[m.queryIdx].pt for m in matches[:50]])
    ptsB = np.float32([kp2[m.trainIdx].pt for m in matches[:50]])

    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC)
    alignedB = cv2.warpPerspective(imgB, H, (imgA.shape[1], imgA.shape[0]))

    return alignedB
```

---

## **preprocess_images.py** – generare perechi + imagini procesate

```python
import cv2, os
from align import align_images

def preprocess(before_path, after_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    before_files = sorted(os.listdir(before_path))
    after_files  = sorted(os.listdir(after_path))

    for bf, af in zip(before_files, after_files):
        A = cv2.imread(os.path.join(before_path, bf))
        B = cv2.imread(os.path.join(after_path, af))

        B_aligned = align_images(A, B)

        A = cv2.resize(A, (256,256))
        B_aligned = cv2.resize(B_aligned, (256,256))

        cv2.imwrite(os.path.join(output_path, f"{bf}_A.png"), A)
        cv2.imwrite(os.path.join(output_path, f"{bf}_B.png"), B_aligned)
```

---
