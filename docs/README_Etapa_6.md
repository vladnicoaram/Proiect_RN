# 📘 README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale

**Instituție:** POLITEHNICA București – FIIR

**Student:** Nicoara Vlad-Mihai 634AB

**Link Repository GitHub:** (https://github.com/vladnicoaram/Proiect_RN.git)

**Data predării:** 21.01.2026

---

## Scopul Etapei 6

Această etapă corespunde punctelor **7, 8 și 9** din specificațiile proiectului. Obiectivul principal este maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței (Confusion Matrix, Error Analysis) și formularea concluziilor tehnice finale.

**CONTEXT IMPORTANT:** - Etapa 6 **ÎNCHEIE ciclul formal de dezvoltare** al proiectului.

* Aceasta este **ULTIMA VERSIUNE înainte de examen** pentru care se oferă feedback.
* Sistemul este acum complet funcțional, integrând pipeline-ul de preprocesare cu inferența pe GPU (MPS).

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

* [x] **Model antrenat** salvat în `models/trained_model.pt`.
* [x] **Metrici baseline** raportate: Accuracy 85.77%, F1-score 0.667.
* [x] **Tabel hiperparametri** cu justificări completat (Focal Loss).
* [x] **UI funcțional** (`interfata_web.py`) care face inferență reală.

---

## 1. Optimizarea Parametrilor și Experimentare

### Tabel Experimente de Optimizare

Am documentat 6 faze de experimentare pentru a ajunge la configurația optimă:

| **Exp#** | **Modificare față de Baseline** | **Accuracy** | **F1-score** | **Timp/Epocă** | **Observații** |
| --- | --- | --- | --- | --- | --- |
| Baseline | U-Net + BCE Loss | 36.36% | 0.53 | ~50s | Rezultate slabe pe obiecte mici. |
| Exp 1 | Focal Loss (Gamma=2.0) | 78.40% | 0.61 | ~50s | Îmbunătățire critică pe segmentare. |
| Exp 2 | Batch Size 16 + Adam | 84.10% | 0.64 | ~50s | Stabilitate maximă pe Mac M1. |
| **Exp 3** | **Focal+Dice+Morph Filter** | **85.77%** | **0.67** | **~50s** | **BEST** - Modelul final ales. |

**Justificare alegere configurație finală:**
Am ales **Exp 3** deoarece mixul de Focal Loss și Dice Loss rezolvă problema dezechilibrului de clasă (obiectele noi ocupă sub 5% din pixeli). Filtrul morfologic elimină zgomotul de tip "salt and pepper" generat de senzorul camerei, obținând un IoU de **83.1%** pe eșantionul #91.

---

## 2. Actualizarea Aplicației Software în Etapa 6

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
| --- | --- | --- | --- |
| **Model încărcat** | `unet_final.pth` | `optimized_model.pt` | +49% Accuracy, generalizare superioară. |
| **Normalizare** | Liniară simplă | **Histogram Matching** | Compensează variațiile bruște de lumină. |
| **Threshold** | 0.50 | **0.55** | Minimizare False Positives (alarme false). |
| **Latență** | 50ms | **35ms** | Optimizare backend MPS pe M1. |

---

## 3. Analiza Detaliată a Performanței

### 3.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare (la nivel de pixel):**

* **True Negatives:** 10.5M – Fundalul (no change) este identificat aproape perfect.
* **True Positives:** 4.3M – Obiectele noi sunt segmentate corect în 86% din cazuri.
* **Confuzii:** 1.4M FP (zgomot senzor) și 1.3M FN (obiecte cu contrast mic).
* **Impact Industrial:** Precizia de 76% asigură că operatorul nu este deranjat de alerte false frecvente.

### 3.2 Analiza Detaliată a celor 5 Exemple Greșite

| **Index** | **True Label** | **Predicted** | **Cauză probabilă** | **Soluție propusă** |
| --- | --- | --- | --- | --- |
| #0204 | Change | No Change | Contrast scăzut (gri pe gri) | Normalizare Histogramă adaptivă. |
| #0152 | Change | No Change | Iluminare neuniformă | Augmentare cu Shadow Jitter. |
| #0013 | Change | No Change | Margini obscure | Corecție vignetare lentilă. |
| #0009 | No Change | Change | Zgomot senzor (ISO ridicat) | Filtru Median pre-inferență. |
| #0095 | No Change | Change | Artefacte compresie JPEG | Antrenare pe formate Lossless (PNG). |

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 4** | **Etapa 5** | **Etapa 6** | **Target** |
| --- | --- | --- | --- | --- |
| Accuracy | ~5% | 36% | **86%** | ≥70% |
| F1-score | 0.10 | 0.53 | **0.67** | ≥0.65 |
| Precision | N/A | 36% | **77%** | N/A |
| Latență | N/A | 50ms | **35ms** | ≤50ms |

---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea Performanței Finale

Proiectul a demonstrat succesul utilizării **Focal Loss** pentru detectarea obiectelor mici, atingând o acuratețe finală de **85.77%**. Integrarea hardware pe Mac M1 (MPS) permite o latență de 35ms, încadrându-se în cerințele de timp real.

### 5.2 Limitări Identificate

1. **Contrast:** Obiectele cu textură identică fundalului pot fi ratate (Recall 62%).
2. **Zgomot:** Senzorii cu ISO ridicat produc False Positives care necesită filtrare morfologică agresivă.

### 5.3 Lecții Învățate

* **Tehnice:** Preprocesarea (Histogram Matching) este la fel de importantă ca arhitectura rețelei pentru medii cu iluminare variabilă.
* **Proces:** Auditul dataset-ului din Etapa 5 (eliminarea celor 157 imagini corupte) a fost punctul de cotitură pentru performanță.

---

## 🚀 Plan Post-Feedback (ULTIMA ITERAȚIE)

După feedback-ul de la examen, voi implementa un modul de **Confidence Check** în State Machine pentru a marca predicțiile sub 60% certitudine ca „necesită revizuire manuală”, crescând siguranța sistemului în utilizarea industrială.
