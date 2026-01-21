# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Nicoară Vlad-Mihai
**Data:** `[DE COMPLETEAZĂ]`

---

## 1. Introducere

Acest document prezintă activitățile realizate în **Etapa 3**, care includ analiza și preprocesarea setului de date pentru proiectul *Compararea și Detectarea Schimbărilor din Imagini Aplicate Sălilor de Laborator*. Scopul etapei este pregătirea imaginilor înainte de antrenarea rețelelor neuronale (Siamese + UNet) asigurând calitate, consistență și reproductibilitate.

---

## 2. Structura Repository-ului GitHub (Versiunea Etapei 3)

```
change-detection-lab/
├── README.md
├── docs/
│   └── datasets/          # informații despre dataset + rezultate EDA
├── data/
│   ├── raw/               # imagini brute (neprocesate)
│   │   ├── before/        # imagini înainte
│   │   └── after/         # imagini după
│   ├── pairs/             # perechi before–after generate automat
│   ├── processed/         # imagini aliniate și normalizate
│   ├── train/             # set de antrenare
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # cod pentru preprocesarea imaginilor
│   ├── data_acquisition/  # generare dataset (dacă se extinde)
│   └── neural_network/    # arhitectura RN (Siamese + UNet)
├── config/                # fișiere configurare preprocesare
└── requirements.txt       # dependențe Python
```

---

## 3. Descrierea Setului de Date

### 3.1 Sursa datelor

* **Origine:** Imagini cu o sală de laborator în două momente diferite (before / after).
* **Modul de achiziție:** Imagini colectate manual sau generate în cadrul proiectului.
* **Condițiile colectării:** Imagini surprinse cu aceeași cameră și unghi similar; diferențe introduse manual (obiect mutat, scaun deplasat etc.).

### 3.2 Caracteristicile dataset-ului

* **Număr total perechi:** `[DE COMPLETEAZĂ]`
* **Număr imagini before:** `[DE COMPLETEAZĂ]`
* **Număr imagini after:** `[DE COMPLETEAZĂ]`
* **Tip:** RGB
* **Format:** PNG/JPG
* **Dimensiune finală:** 256×256 px

### 3.3 Componentele unui sample

| Componentă     | Tip     | Descriere                             |
| -------------- | ------- | ------------------------------------- |
| Imagine_before | RGB     | Imaginea capturată la începutul orei  |
| Imagine_after  | RGB     | Imaginea capturată la sfârșitul orei  |
| Mask_diff      | Imagine | Mască binară a zonelor modificate     |
| Score_diff     | Numeric | Scor `0..1` al nivelului de diferență |

---

## 4. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 4.1 Analiză cantitativă

* Număr imagini before/after
* Rezoluții originale
* Histograme R/G/B
* Consistența perechilor A–B

### 4.2 Analiză calitativă

* Iluminare neuniformă
* Diferențe de unghi/perspectivă
* Imagini nealiniate
* Zgomot / blur

### 4.3 Probleme identificate

* Variații de lumină → normalizare
* Aliniere imperfectă → homography
* Dataset mic → augmentări
* Diferențe subtile → filtre post-proces

---

## 5. Preprocesarea Datelor

### 5.1 Curățare

* Eliminare imagini corupte
* Conversie la RGB
* Redimensionare 256×256
* Normalizare iluminare

### 5.2 Aliniere perechi

Metodă: ORB/SIFT → matching → Homography → warp.

### 5.3 Generare etichete

* `Mask_diff`: diferență + threshold + morfologie
* `Score_diff`: proporție pixeli modificați

### 5.4 Normalizare și augmentări

* Scaling `[0,1]`
* Flip, rotație, luminozitate

### 5.5 Split

* 70% train
* 15% validation
* 15% test

### 5.6 Output final

* `processed/`
* `pairs/`
* `train/`, `validation/`, `test/`
* `config/preprocessing_config.yaml`

---

## 6. Fișiere Generate în Etapa 3

* `data/raw/`
* `data/pairs/`
* `data/processed/`
* `docs/datasets/`
* `src/preprocessing/`

---

## 7. Checklist Etapa 3

* [ ] Structură repo configurată
* [ ] Set imagini colectat
* [ ] EDA completă
* [ ] Imagini preprocesate
* [ ] Seturi generate
* [ ] Documentație completă

---

# 📘 README – Etapa 4: Arhitectura Completă SIA

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Nicoară Vlad-Mihai
**Link GitHub:** `[DE COMPLETEAZĂ]`
**Data:** `[DE COMPLETEAZĂ]`

---

## Scopul Etapei 4

Etapa definește arhitectura completă a sistemului cu inteligență artificială (SIA). Modelul este creat și compilat, pipeline-ul complet (date → preprocesare → model → UI) funcționează fără erori.

---

## 1. Tabel Nevoie Reală → Soluție SIA → Modul Software

| Nevoie reală             | Soluție SIA                            | Modul                                                                   |
| ------------------------ | -------------------------------------- | ----------------------------------------------------------------------- |
| Detectarea modificărilor | Pipeline imagini → UNet → scor & mască | `data_acquisition`, `preprocessing`, `neural_network`, `postprocessing` |
| Notificare operator      | Dacă scor > 0.6 → alertă               | `alert_manager`, `api`                                                  |
| Trasabilitate            | Log CSV/DB 6 luni                      | `data_logging`                                                          |

---

## 2. Contribuție originală dataset (≥ 40%)

* **N =** `[DE COMPLETEAZĂ]`
* **M =** `[DE COMPLETEAZĂ]`
* **Procent =** `[DE COMPLETEAZĂ]` %

Tip contribuție:

* [x] Date originale (imagini capturate manual)

Include:

* `docs/acquisition_setup.jpg`
* `docs/data_statistics.csv`
* `docs/generated_vs_public.png`

---

## 3. State Machine

```
IDLE
 → ACQUIRE_IMAGES
 → VALIDATE_IMAGES
 → PREPROCESS_IMAGES
 → GENERATE_MASK_CANDIDATES
 → RN_INFERENCE
 → EVALUATE_CHANGE
    ├─ [OK]    → LOG_RESULT → UPDATE_DASHBOARD → IDLE
    └─ [ALERT] → TRIGGER_ALERT → NOTIFY_OPERATOR → LOG_INCIDENT → IDLE
↓
ERROR_HANDLER → RETRY (x2) → IDLE / ABORT
```

---

## 4. Module Software

### Modul 1 — Data Acquisition & Logging

Fișiere:

* `capture_stub.py`
* `count_dataset.py`

Comenzi:

```
python src/data_acquisition/capture_stub.py --source folder --out data/generated/ --n_pairs 20
python src/data_acquisition/count_dataset.py
```

### Modul 2 — Neural Network

* `model_siamese_unet.py`
* `train_stub.py`

Comenzi:

```
python src/neural_network/model_siamese_unet.py
python src/neural_network/train_stub.py --dry
```

### Modul 3 — Web Service / UI

* `app_fastapi.py`
* `ui_demo.py`

Comenzi:

```
uvicorn src.app.app_fastapi:app --reload --port 8000
streamlit run src.app/ui_demo.py
```

---

## 5. Structura Repository finală Etapa 4

```
proiect-rn-nicoara/
├── data/
├── src/
├── docs/
├── models/
├── config/
├── README_Etapa3.md
├── README_Etapa4_Arhitectura_SIA.md
└── requirements.txt
```

---

## 6. Checklist Etapa 4

* [ ] Capturează date
* [ ] Rulează `count_dataset.py`
* [ ] Rulează modelul & generează `untrained_model.pth`
* [ ] Test UI FastAPI + Streamlit
* [ ] Export diagrama state machine
* [ ] Commit + tag

```
git add .
git commit -m "Etapa 4 completă - Arhitectură SIA funcțională"
git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"
git push --follow-tags
```

---

## 7. Teste Recomandate

* [ ] Test 10–20 perechi reale → `docs/perf_summary.csv`
* [ ] Răspuns FastAPI < 3s
* [ ] Contribuție originală ≥ 40%

---
