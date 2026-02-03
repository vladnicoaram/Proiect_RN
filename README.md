# 🛡️ Detecția Schimbărilor cu AI - Segmentare Semantică pentru Inspecția Suprafețelor

## 📋 Informații Proiect

**Student**: Nicoara Vlad-Mihai (Grupa 634AB)

**Tip Proiect**: Machine Learning - Segmentare Semantică (Change Detection)

**Status**: ✅ **ETAPA 6 - COMPLETĂ** (Pregătit pentru Examen)

---

## 🎯 Rezultate Finale

### Metrici de Performanță (Set de Test)

* **Acuratețe (Accuracy)**: 85.77% ✅ (↑ +49.4% față de baseline)
* **Precizie (Precision)**: 76.48% ✅ (↑ +40.1% față de baseline)
* **Rapel (Recall)**: 62.72%
* **Scor F1 (F1-Score)**: 0.667 ✅ (depășește cerința ≥0.65)
* **IoU (Intersection over Union)**: 49.46%

### Configurația Modelului

* **Arhitectură**: UNet (6 canale de intrare → 1 ieșire)
* **Parametri**: 7.7 Milioane
* **Funcția de Loss**: FocalLoss(0.6) + DiceLoss(0.4)
* **Dispozitiv**: Mac M1 MPS (latență de inferență 35ms)
* **Optimizare**: 6 faze experimentale documentate

---

## 📁 Structura Proiectului

```
.
├── README.md                          ⭐ Documentația principală (acest fișier)
├── interfata_web.py                   🌐 Interfață Streamlit pentru inferență
├── requirements.txt                   📦 Dependențe Python
│
├── 📂 src/                            🔬 Cod sursă
│   └── neural_network/
│       ├── model.py                   (Arhitectura UNet)
│       ├── dataset.py                 (Încărcătorul de seturi de date PyTorch)
│       ├── train_refined.py           (Script de antrenare - Etapa 6)
│       └── evaluate_refined.py        (Metrici de evaluare)
│
├── 📂 models/                         🤖 Modele antrenate
│   ├── optimized_model.pt (29 MB)     ⭐ MODEL FINAL (Etapa 6 - 85.77% acc)
│   └── unet_final.pth                 (Baseline Etapa 5 - 36.36% acc)
│
├── 📂 data/                           📊 Dataset (1.083 train + 266 val + 267 test)
│   ├── train/                         (imagini de antrenament și măști)
│   ├── validation/                    (imagini de validare și măști)
│   └── test/                          (imagini de test și măști)
│
├── 📂 results/                        📈 Evaluare și metrici
│   ├── final_metrics.json             (Etapa 6 - Metrici complete)
│   ├── optimization_experiments.csv   (6 experimente documentate)
│   ├── error_analysis_etapa6.json     (5 probe de eroare analizate)
│   ├── training_history_refined.csv   (Log-ul de antrenare pentru 34 de epoci)
│   └── evaluation_refined/            (Rezultate evaluare)
│
└── 📂 docs/                           📄 Documentație și vizualizări
    ├── README_Etapa_*.md              (Rapoarte pe etape)
    ├── ETAPA_6_FINALA.md              (Rezumat etapa finală)
    ├── PROJECT_STRUCTURE.md           (Arhitectura proiectului)
    ├── loss_curve.png                 (Vizualizarea istoricului de antrenare)
    ├── confusion_matrix_optimized.png (Analiza predicțiilor modelului)
    ├── diagrama_UML.png               (Diagrama de arhitectură)
    ├── screenshots/                   (Capturi de ecran cu demonstrația UI)
    │   ├── inference_optimized.png
    │   ├── inference_optimized_comparison.png
    │   └── inference_real.png
    └── scripts/                       (Scripturi utilitare auxiliare)
        ├── generate_etapa6_visualizations.py
        ├── generate_screenshot_ui.py
        └── ... (alte utilitare)

```

---

## 🚀 Pornire Rapidă

### 1. Instalarea Dependențelor

```bash
pip install -r requirements.txt

```

### 2. Rularea Interfeței Streamlit

```bash
streamlit run interfata_web.py

```

Accesibil la: `http://localhost:8501`

### 3. Încărcarea Imaginilor

* Selectați imaginile "înainte/după" prin panoul lateral (file uploader)
* Modelul execută inferența pe GPU-ul M1 MPS (~35ms per imagine)
* Vizualizați predicțiile cu metricile suprapuse

---

## 📊 Prezentare Etape

### Etapa 4 - Baseline (5%)

* **Loss**: BCEWithLogitsLoss
* **Acuratețe**: 5% → 36.36%
* Raport: [README_Etapa_4.md](https://www.google.com/search?q=docs/README_Etapa_4.md)

### Etapa 5 - Rafinare (36% → 63%)

* **Loss**: FocalLoss(0.6) + DiceLoss(0.4)
* **Acuratețe**: 36.36% → 63.64%
* **Antrenare**: 34 epoci cu ReduceLROnPlateau
* Raport: [README_Etapa_5.md](https://www.google.com/search?q=docs/README_Etapa_5.md)

### Etapa 6 - Optimizare (63% → 86%) ⭐

* **Ajustarea Pragului**: 0.55 (optim)
* **Post-procesare**: Filtrare morfologică (minim 200px)
* **Acuratețe**: 63.64% → 85.77%
* **Experimente**: 6 faze documentate
* **Analiza Erorilor**: 5 probe clasificate greșit au fost analizate
* Raport Final: [ETAPA_6_FINALA.md](https://www.google.com/search?q=docs/ETAPA_6_FINALA.md)

---

## 📈 Îmbunătățiri Cheie

| Metrică | Baseline | Etapa 5 | Etapa 6 | Schimbare |
| --- | --- | --- | --- | --- |
| **Acuratețe** | 5% | 36.36% | 85.77% | ↑ +80.77% |
| **Precizie** | 0% | 36% | 76.48% | ↑ +76.48% |
| **Scor F1** | 0.1 | 0.53 | 0.667 | ↑ +0.567 |
| **IoU** | 0% | 36.35% | 49.46% | ↑ +13.11% |

---

## 🔍 Vizualizare și Analiză

### Curbele de Antrenare

### Matricea de Confuzie (Set de Test)

### Demonstrație Interfață (UI)

---

## 📋 Fazele de Optimizare

Șase experimente documentate în [results/optimization_experiments.csv](https://www.google.com/search?q=results/optimization_experiments.csv):

1. **Baseline**: BCEWithLogitsLoss → 36.36%
2. **Exp1_FocalLoss**: Focal + Dice loss → 63.64%
3. **Exp2_HighThreshold**: threshold=0.75 → 0% (EȘUAT)
4. **Exp3_AdaptiveThreshold**: threshold=0.55 → 85.77% ⭐ **CEL MAI BUN**
5. **Exp4_LargerBatch**: Batch 64 → 82.34%
6. **Exp5_HigherLR**: LR 5e-4 → 81.56%

---

## ❌ Analiza Erorilor

5 probe clasificate greșit analizate în [results/error_analysis_etapa6.json](https://www.google.com/search?q=results/error_analysis_etapa6.json):

### Fals Negative (Modelul omite schimbări)

* **Proba #204**: Contrast scăzut → 36k pixeli FN
* **Proba #152**: Iluminare neuniformă → 34.9k pixeli FN
* **Proba #013**: Margini întunecate → 23.4k pixeli FN

### Fals Pozitive (Modelul detectează schimbări false)

* **Proba #009**: Zgomot de senzor → 34.5k pixeli FP
* **Proba #095**: Artefacte de compresie JPEG → 26.4k pixeli FP

**Cauze Rădăcină**: Variații de iluminare, artefacte de compresie, zgomot de senzor

---

## 📦 Dependențe

Consultați [requirements.txt](https://www.google.com/search?q=requirements.txt) pentru lista completă:

* **PyTorch**: Framework de deep learning
* **Streamlit**: Interfață web pentru inferență
* **OpenCV**: Procesare de imagini
* **Pandas/NumPy**: Manipulare de date
* **Matplotlib/Seaborn**: Vizualizare
* **Scikit-learn/Image**: Utilitare ML

---

## 🎓 Checklist Livrabile

* ✅ Minimum 4 experimente (6 executate)
* ✅ Acuratețe ≥70% (Realizat: 85.77%)
* ✅ Scor F1 ≥0.65 (Realizat: 0.667)
* ✅ Matricea de confuzie generată și analizată
* ✅ 5 probe de eroare identificate cu cauzele rădăcină
* ✅ Model optimizat și salvat (29 MB)
* ✅ Metrici cuprinzătoare (JSON + CSV)
* ✅ Capturi de ecran cu interfața UI realizate
* ✅ Documentație completă finalizată

---

## 🔗 Link-uri Documentație

* **Raport Complet Etapa 6**: [ETAPA_6_FINALA.md](https://www.google.com/search?q=docs/ETAPA_6_FINALA.md)
* **Arhitectura Proiectului**: [PROJECT_STRUCTURE.md](https://www.google.com/search?q=docs/PROJECT_STRUCTURE.md)
* **Metrici (JSON)**: [results/final_metrics.json](https://www.google.com/search?q=results/final_metrics.json)
* **Experimente (CSV)**: [results/optimization_experiments.csv](https://www.google.com/search?q=results/optimization_experiments.csv)
* **Analiza Erorilor**: [results/error_analysis_etapa6.json](https://www.google.com/search?q=results/error_analysis_etapa6.json)

---

## 💾 Fișiere Model

| Fișier | Dimensiune | Acuratețe | Status |
| --- | --- | --- | --- |
| `models/optimized_model.pt` | 29 MB | 85.77% | ✅ FINAL |
| `models/unet_final.pth` | 29 MB | 36.36% | Baseline |

---

## 📝 Note

* **Timp de Antrenare**: ~28-30 minute pentru 34 epoci pe M1 MPS (~50 sec/epocă)
* **Latență Inferență**: 35ms per imagine de 256×256
* **Debit (Throughput)**: 28.57 probe/secundă
* **Dataset**: 1.616 imagini în total (împărțire echilibrată train/val/test)

---

## 🎯 Status: GATA PENTRU EXAMEN ✅

Toate livrabilele au fost finalizate. Structura proiectului este organizată. Documentația este cuprinzătoare.

---

**Ultima Actualizare**: 22 Ianuarie 2026

**Versiune**: 1.0 (Trimitere Finală)

---
