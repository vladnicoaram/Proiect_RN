# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale

**Instituție:** POLITEHNICA București – FIIR

**Student:** Nicoara Vlad-Mihai 634AB

**Link Repository GitHub:** (https://github.com/vladnicoaram/Proiect_RN.git)

**Data predării:** 21.01.2026

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței pe setul de test și integrarea modelului antrenat în aplicația completă, conform instrucțiunilor de la curs.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:

* State Machine definit și justificat
* Cele 3 module funcționale (Data Logging, RN, UI)
* Minimum 40% date originale în dataset (1083 imagini validate)

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, am verificat existența următoarelor elemente din Etapa 4:**

* [x] **State Machine** definit și documentat în `docs/README_Etapa_4.md`
* [x] **Contribuție ≥40% date originale** în `data/generated/` (1083 imagini după eliminarea celor 157 corupte)
* [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
* [x] **Modul 2 (RN)** cu arhitectură U-Net definită
* [x] **Modul 3 (UI/Web Service)** funcțional cu model dummy
* [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

---

## Pregătire Date pentru Antrenare

### Auditul și Preprocesarea Dataset-ului:

Am refăcut preprocesarea pe dataset-ul combinat pentru a asigura eliminarea erorilor de tip "empty mask" detectate inițial.

```bash
# 1. Combinare date
python src/preprocessing/combine_datasets.py

# 2. Refacere preprocesare COMPLETĂ cu parametrii stabiliți
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42

```

**Parametri de preprocesare utilizați:**

* Scaler salvat în `config/preprocessing_params.pkl` (Min-Max Scaling).
* Proporții split: **70% train / 15% validation / 15% test**.
* `random_state=42` pentru reproducibilitate.

---

## Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

1. **Antrenare model:** Model U-Net antrenat pe setul final de 1083 imagini.
2. **Epoci:** 34 epoci rulate, batch size 16.
3. **Împărțire:** Stratificată 70% / 15% / 15%.
4. **Metrici calculate pe test set:**
* **Acuratețe: 85.77%** ✅
* **F1-score (macro): 0.66** ✅


5. **Salvare model antrenat:** `models/trained_model.pt` (format PyTorch).
6. **Integrare în UI:** UI-ul încarcă acum modelul real, realizând inferență pe baza weights-urilor antrenate.

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
| --- | --- | --- |
| Learning rate | 0.001 | Valoare standard pentru Adam; controlată prin scheduler pentru a evita minimul local. |
| Batch size | 16 | Optimizat pentru latența Unified Memory pe chip-ul **Apple M1 (MPS)**. |
| Number of epochs | 60 (34 rulate) | S-a utilizat Early Stopping pentru a opri antrenarea la convergența `val_loss`. |
| Optimizer | Adam | Adaptive learning rate, necesar pentru segmentarea precisă a formelor neregulate. |
| Loss function | **Focal + Dice** | **Critic:** Focal Loss forțează modelul să învețe obiectele mici (haine) care au puțini pixeli în mască. |
| Activation functions | ReLU (hidden), Sigmoid (output) | ReLU pentru evitarea vanishing gradient, Sigmoid pentru output de tip probabilitate pixel. |

**Justificare detaliată batch size:**
Am ales `batch_size=16` deoarece lucrăm pe o arhitectură **Apple Silicon M1**. Un batch mai mare (ex: 32) genera latență în accesul la memoria GPU-ului integrat (Unified Memory), în timp ce un batch de 16 asigură un gradient stabil și o viteză de procesare de aproximativ **91.9% GPU utilization**.

---

### Nivel 2 – Recomandat (85-90% din punctaj)

1. **Early Stopping:** Antrenarea s-a oprit la epoca 34 deoarece performanța pe setul de validare nu s-a mai îmbunătățit timp de 5 epoci consecutive.
2. **Learning Rate Scheduler:** Am utilizat `ReduceLROnPlateau` cu un factor de 0.5.
3. **Augmentări relevante domeniu:**
* **Imagini interioare:** `ColorJitter` pentru a simula variațiile de iluminare (zi/noapte/lumină artificială).
* **Perspective:** `RandomHorizontalFlip` pentru a simula unghiuri diferite de captură ale camerei.


4. **Grafic performanță:** Salvat în `docs/loss_curve.png`.
5. **Analiză erori:** Detaliată în secțiunea dedicată contextului industrial.

**Indicatori țintă Nivel 2 atinși:**

* **Acuratețe: 85.77%** (Target ≥ 75%)
* **F1-score (macro): 0.67** (Target ≥ 0.70 - Aproape de target, optimizat pentru Precision).

---

### Nivel 3 – Bonus (până la 100%)

| **Activitate** | **Livrabil** |
| --- | --- |
| Comparare Arhitecturi | Tabel comparativ între Loss BCE (vechi) și Loss Focal (nou). |
| Analiză Exemple Greșite | Analiza sample-ului #0091 unde IoU a crescut de la 4% la 83%. |

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența respectă fluxul definit în Etapa 4:

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
| --- | --- |
| `ACQUIRE_IMAGES` | Citire batch date din `data/test/` pentru evaluare finală. |
| `PREPROCESS_IMAGES` | Aplicare normalizare  conform parametrilor din Etapa 3. |
| `RN_INFERENCE` | Forward pass pe dispozitivul `mps` cu modelul `trained_model.pt`. |
| `EVALUATE_CHANGE` | Aplicare Threshold de **0.55** și filtrare pete sub 200px. |
| `TRIGGER_ALERT` | Generarea măștii vizuale (verde) în UI pentru utilizator. |

---

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

### 1. Pe ce clase greșește cel mai mult modelul?

Confusion Matrix arată că modelul tinde să ignore (False Negatives) obiectele cu textură similară fundalului (ex: haine de culoare gri pe covor gri). Aceasta este o limitare a contrastului cromatic în dataset-ul original.

### 2. Ce caracteristici ale datelor cauzează erori?

Zgomotul digital (grain) din pozele făcute în lumină slabă induce "artefacte" în predicție. În mediul industrial, acest lucru ar putea fi cauzat de senzori foto de calitate scăzută sau praf pe lentilă.

### 3. Ce implicații are pentru aplicația industrială?

**FALSE POSITIVES (alarmă falsă):** ACCEPTABIL → utilizatorul primește o notificare de schimbare care nu e reală.

**FALSE NEGATIVES (schimbare nedetectată):** CRITIC → un obiect nou (ex. un obstacol sau un furt) nu este detectat.

**Prioritate:** Am prioritizat **Precizia (76.4%)** pentru a evita alarmele false repetitive care ar duce la ignorarea sistemului de către operator.

### 4. Ce măsuri corective propuneți?

1. Colectarea a 500+ imagini adiționale pentru clasa de obiecte mici (haine, genți).
2. Implementarea unui filtru morfologic de tip `Opening` pentru eliminarea zgomotului.
3. Re-antrenarea cu `class weights` mai agresive pentru pixelii de tip "obiect".

---

## Structura Repository-ului la Finalul Etapei 5

```
proiect-rn-nicoara-vlad/
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER
├── docs/
│   ├── state_machine.png              
│   ├── loss_curve.png                 
│   └── screenshots/
│       └── inference_real.png         # Screenshot demonstrație (IoU 83%)
├── models/
│   ├── untrained_model.pt             
│   └── trained_model.pt               # Modelul antrenat (29 MB)
├── results/                            
│   ├── training_history_refined.csv   # Log-urile celor 34 epoci
│   └── test_metrics.json              # Metrici finale

```

---

## Livrabile Obligatorii (Nivel 1)

1. **`docs/etapa5_antrenare_model.md`** (acest fișier completat).
2. **`models/trained_model.pt`** - model antrenat PyTorch (29 MB).
3. **`results/training_history_refined.csv`** - istoric epoci.
4. **`results/test_metrics.json`** - metrici finale: Accuracy 0.857, Precision 0.764.
5. **`docs/screenshots/inference_real.png`** - demonstrație UI cu predicție reală pe sample #0091.

