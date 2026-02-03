# AUDIT DE CERTIFICARE - RAPORT FINAL

**Data Executie**: 22 Ianuarie 2026  
**Dataset**: 414 perechi (dupa stergerea ORB mismatch)  
**Status**: ⚠️ Partial Validated (190/414 passed)

---

## 📊 REZUMAT EXECUTIE

### Perechi Analizate
- **Total**: 414 perechi
- **Passed (Gold Standard)**: 190 perechi (45.9%)
- **Rejected**: 224 perechi (54.1%)

### Motivul Respingerii

| Criteriu | Perechi Respinse |
|----------|------------------|
| SSIM < 0.70 | 177 (79.0%) |
| ORB Matches < 25 | 158 (70.5%) |
| Histogram Corr < 0.60 | 53 (23.7%) |

*Nota: Unele perechi nu trec mai multe criterii simultan.*

---

## ✓ PHASE 1: VERIFICARE SINCRONIZARE

### Status: ✓ PASSED

```
TRAIN (283 perechi):
  Before:       283 ✓
  After:        283 ✓
  Masks:        283 ✓
  Masks_clean:  283 ✓
  Status: PERFECT SINCRONIZAT

TEST (68 perechi):
  Before:   68 ✓
  After:    68 ✓
  Masks:    68 ✓
  Status: PERFECT SINCRONIZAT

VALIDATION (63 perechi):
  Before:   63 ✓
  After:    63 ✓
  Masks:    63 ✓
  Status: PERFECT SINCRONIZAT
```

---

## 📈 PHASE 2: TEST CALITATE - GOLD STANDARD

### Thresholds Aplicati
- **SSIM**: $ > 0.70 $
- **ORB Matches**: $ > 25 $ puncte
- **Histogram Correlation**: $ > 0.60 $

### Statistici Perechi PASSED (190)

#### SSIM Score
```
Mean:      0.8257
Median:    0.8156
Std Dev:   0.0746
Min:       0.7004
Max:       1.0000
Range:     [0.7004, 1.0000]
```

#### ORB Matches
```
Mean:      82.7 matches
Median:    62 matches
Std Dev:   64.7
Min:       25 matches
Max:       414 matches
Range:     [25, 414]
```

#### Histogram Correlation
```
Mean:      0.9424
Median:    0.9639
Std Dev:   0.0670
Min:       0.6008
Max:       1.0000
Range:     [0.6008, 1.0000]
```

---

## ✗ PHASE 3: ANALIZA PERECHI RESPINSE

### Distributie pe Dataset

| Dataset | Total | Passed | Rejected | % Passed |
|---------|-------|--------|----------|----------|
| Train | 283 | 122 | 161 | 43.1% |
| Test | 68 | 34 | 34 | 50.0% |
| Validation | 63 | 34 | 29 | 54.0% |
| **TOTAL** | **414** | **190** | **224** | **45.9%** |

### Top 10 Perechi WORST

| # | Dataset | Filename | SSIM | ORB | Histogram |
|---|---------|----------|------|-----|-----------|
| 1 | train | 3FO4IH7AFVTR_735_empty_1 | 0.2316 | 6 | 0.5427 |
| 2 | train | 3FO4IJM3YPQ1_912_empty_1 | 0.2912 | 10 | 0.6780 |
| 3 | train | 3FO4IN2ELTYX_401_empty_1 | 0.3159 | 13 | 0.4548 |
| 4 | test | 3FO4IMP6G304_844_empty_1 | 0.3291 | 11 | 0.5700 |
| 5 | validation | 3FO4IIOCR1VX_149_empty_1 | 0.3309 | 19 | 0.1873 |
| 6 | train | 3FO4IO3UGYX2_1137_empty_1 | 0.3626 | 4 | 0.8756 |
| 7 | train | 3FO4INFX1K3H_224652_empty_1 | 0.3634 | 8 | 0.8744 |
| 8 | train | 3FO4III2GFNW_182_empty_1 | 0.3704 | 8 | 0.1804 |
| 9 | test | 3FO4IORY3CEK_490_empty_1 | 0.3744 | 16 | 0.2558 |
| 10 | test | 3FO4IM7TTP0E_272_empty_1 | 0.3866 | 9 | 0.6836 |

---

## 📁 FISIERE RESPINSE

### Locatie
```
/Users/admin/Documents/Facultatea/Proiect_RN/data/REJECTED_BY_AUDIT/
```

### Structura
```
REJECTED_BY_AUDIT/
├── train/
│   ├── before/     (161 imagini)
│   ├── after/      (161 imagini)
│   ├── masks/      (161 imagini)
│   └── masks_clean/ (161 imagini)
├── test/
│   ├── before/     (34 imagini)
│   ├── after/      (34 imagini)
│   └── masks/      (34 imagini)
└── validation/
    ├── before/     (29 imagini)
    ├── after/      (29 imagini)
    └── masks/      (29 imagini)
```

### Total Fisiere Respinse
- **Train**: 644 fisiere (161 × 4)
- **Test**: 102 fisiere (34 × 3)
- **Validation**: 87 fisiere (29 × 3)
- **TOTAL**: 833 fisiere

---

## 🎯 CONCLUZIE VALIDARE

### Mean SSIM Analysis

```
Mean SSIM (Perechi PASSED):   0.8257
Prag Minim (Gold Standard):   0.8500
Deficit:                      -0.0243 (-2.9%)

Status: ✗ NU INDEPLINESTE STANDARDUL GOLD
```

### Interpretare

Chiar și perechilevreunei care **trec** pragurile stricte au o medie SSIM de **0.8257**, care este sub pragul Gold Standard de **0.85**.

#### Ce inseamna?
- **190 din 414 perechi** (45.9%) sunt de calitate suficienta
- **Dintre acestea**, media SSIM este 0.8257 (doar **2.9% sub prag**)
- **224 din 414 perechi** (54.1%) au **severi probleme de calitate**
  - 177 cu SSIM prost (< 0.70)
  - 158 cu puține feature matches (ORB < 25)
  - 53 cu histogram slabă (< 0.60)

---

## 🔍 INTERPRETARE REZULTATE

### Ce Se Intamplă

Dataset-ul original de 1.616 perechi a fost redus prin:
1. **ORB Feature Matching** → 1.202 perechi eliminate (74.4%)
2. **Gold Standard Audit** → 224 perechi suplimentar respinse (54.1% din 414 ramase)

Aceasta sugereaza:
- ORB matching a identificat corect mult din problemele scene mismatch
- Dar **24 din 414 perechi ramase inca au probleme grave**
- Dataset-ul are **probleme fundamentale de calitate/aliniere**

### Scenarii Posibile

1. **Dataset neuniform**: Imagini din mai multe surse/experimente cu calitati diferite
2. **Labeling errors**: Perechi nu sunt corect aliniate "before-after"
3. **Varianta de captare**: Unele imagini au perspective diferite intentionately
4. **Probleme de detectie**: Metricele ORB + SSIM sunt prea stricte

---

## 📋 RAPOARTE GENERATE

### CSV Files

1. **`audit_passed_perechi.csv`** (190 intrari)
   - Perechi care trec Gold Standard
   - Coloane: dataset, filename, ssim, orb_matches, histogram_corr, passed

2. **`audit_rejected_perechi.csv`** (224 intrari)
   - Perechi respinse
   - Coloane: dataset, filename, ssim, orb_matches, histogram_corr, passed

### Folder Respinse
- **Locatie**: `data/REJECTED_BY_AUDIT/`
- **Continut**: Copii complete ale imaginilor respinse (before, after, masks)
- **Scopul**: Inspectie manuala si investigare

---

## 💡 RECOMANDARI

### Optiune 1: Continua cu 190 perechi (Gold Standard)
```
Dataset Final: 190 perechi
├── Train: 122 perechi
├── Test: 34 perechi
└── Validation: 34 perechi

Mean SSIM: 0.8257 (sub prag, dar acceptabil)
ORB Mean: 82.7 matches (excelent)
```

**Avantaj**: Dataset foarte curat  
**Dezavantaj**: Foarte mic (88% reducere din original)

### Optiune 2: Relaxa Thresholds
Reduce thresholds la:
- SSIM: 0.65 (in loc de 0.70)
- ORB: 20 (in loc de 25)
- Histogram: 0.55 (in loc de 0.60)

**Avantaj**: Dataset mai mare  
**Dezavantaj**: Calitate mai mica

### Optiune 3: Inspectie Manuala Selectiva
- Verifica manual `data/REJECTED_BY_AUDIT/`
- Salveaza perechi care par corecte
- Reintoarce in dataset activ

---

## 📊 TABEL FINAL - METRICI

| Metrica | Value | Status |
|---------|-------|--------|
| **Perechi Analizate** | 414 | - |
| **Perechi Gold Standard** | 190 | 45.9% ✓ |
| **Perechi Respinse** | 224 | 54.1% ✗ |
| **Mean SSIM (Passed)** | 0.8257 | Sub prag 0.85 ⚠️ |
| **Mean ORB Matches** | 82.7 | Excelent ✓ |
| **Mean Histogram** | 0.9424 | Excelent ✓ |
| **Sincronizare Foldere** | Perfect | ✓ |

---

## ✓ CONCLUZIE GENERALA

**Status Audit**: ⚠️ **Partial Pass**

Dataset-ul:
- ✓ Este **perfect sincronizat** (hard check passed)
- ✓ Contine **190 perechi de calitate inalta** (Gold Standard)
- ✗ Are **224 perechi cu probleme semnificative de calitate**
- ⚠️ Mean SSIM **sub pragul Gold (0.8257 vs 0.85)**

### Recomandare Finala
- Usar manual `REJECTED_BY_AUDIT/` pentru inspectie
- Decide daca esti mulțumit cu 190 perechi sau daca vrei sa relaxezi thresholds
- Pentru antrenare model: puteti incepe cu 190 perechi si observa performanta
