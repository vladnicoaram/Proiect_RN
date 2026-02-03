# 🚀 QUICK START - ANTRENARE MODEL

---

## Setup Verificat ✓

```
Dataset:           190 perechi (122 train + 34 val + 34 test)
Dataset Quality:   Gold Standard (SSIM > 0.70)
Model:             UNet (6 input channels, 1 output channel)
Training Script:   train_final_refined.py (17 KB)
Device:            Mac M1 (MPS)
```

---

## START TRAINING - 3 METODE

### METODA 1: Direct Command
```bash
cd /Users/admin/Documents/Facultatea/Proiect_RN
python3 src/neural_network/train_final_refined.py
```

### METODA 2: Cu Validare Prealabila
```bash
# Verifica dataset
python3 check_and_train.py

# Apoi ruleaza antrenare
python3 src/neural_network/train_final_refined.py
```

### METODA 3: Background (In Alta Termina)
```bash
# Terminal 1 - Ruleaza antrenare
cd /Users/admin/Documents/Facultatea/Proiect_RN
python3 src/neural_network/train_final_refined.py

# Terminal 2 - Monitor progres (dupa ce se incepe)
python3 monitor_training.py
```

---

## EXPECTED OUTPUT

```
================================================================================
TRAIN_FINAL_REFINED - Antrenare Optimizata pentru Dataset Mic (190 perechi)
================================================================================

[1] Loading datasets...
  [TRAIN] Loaded 122 samples
  [VALIDATION] Loaded 34 samples
  [TEST] Loaded 34 samples
  Train batches: 8
  Val batches: 3
  Test batches: 3

[2] Creating model...
  Model parameters: 7,766,721

[3] Starting training...
  Epochs: 100
  Batch size: 16
  Learning rate: 0.0001
  Early stopping patience: 15

Epoch 1/100
  Batch 1/8: loss=0.6931
  Batch 2/8: loss=0.6421
  ...
  Train loss: 0.5234
  Val loss: 0.4876
  LR: 0.000100
  ✓ Best model saved (val_loss: 0.4876)

Epoch 2/100
  ...
```

---

## MONITORING

### In Timp Ce Ruleaza
```bash
# Terminal 2
watch -n 5 "python3 monitor_training.py"
```

### Verifica Loss Curves (Dupa ce se termina)
```bash
open results/training_curves_refined.png
```

### Verifica Metrici Finale
```bash
cat results/training_results_refined.json | python3 -m json.tool
```

---

## FISIERE GENERATE

```
checkpoints/
  └─ best_model_refined.pth     (Model cu cea mai mica validation loss)

results/
  ├─ training_results_refined.json
  └─ training_curves_refined.png

Logs:
  (implicit in console output)
```

---

## CONFIGURATIE HIPERPARAMETRI

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch Size** | 16 | Mic pentru stabilitate |
| **Learning Rate** | 1e-4 | Conservative |
| **Epochs** | 100 | Max, dar early stopping la ~40-50 |
| **Early Stop Patience** | 15 | Opreste daca nu se imbunatateste |
| **LR Scheduler Patience** | 5 | Scade LR dupa 5 epoci |
| **Optimizer** | Adam | Adaptive learning |
| **Weight Decay** | 1e-5 | L2 regularization |
| **Gradient Clipping** | 1.0 | Previne exploding gradients |

---

## DATA AUGMENTATION (TRAINING ONLY)

```
✓ Rotations:    ±30 degrees
✓ H-Flip:       50% probabilitate
✓ V-Flip:       50% probabilitate
✓ Zoom:         0.9x - 1.1x aleatoriu
✓ Brightness:   0.8x - 1.2x aleatoriu
✓ Contrast:     0.8x - 1.2x aleatoriu
```

**Nota**: Augmentarile se aplica CONSISTENT pe before, after, si mask.

---

## TIMELINE ESTIMAT

```
Per Batch:      ~0.6 secunde
Per Epoch:      ~5 secunde (8 batches train + 3 val)
Total Dataset:  190 perechi

Worst Case (100 epoci):
  100 * 5s = 500s ≈ 8 minuten

With Early Stopping (tipic 40-50 epoci):
  45 * 5s = 225s ≈ 4 minuten
```

---

## TROUBLESHOOTING

### ❓ Training e lent
```
- Verifica ps aux | grep python3
- Scade batch_size la 8 in train_final_refined.py
- Verifica ca dataset e intact
```

### ❓ Loss nu descreste
```
- Verifica learning_rate (ar trebui 1e-4)
- Verifica data loading
- Verifica transformari
```

### ❓ Memory issues
```
- Scade batch_size
- Scade num_workers (este deja 0)
```

### ❓ No output / Stuck
```
- Verifica MPS disponibil: python3 -c "import torch; print(torch.backends.mps.is_available())"
- Verifica dataset: python3 check_and_train.py
```

---

## DOPO CE SE TERMINA

### 1. Verifica Rezultate
```bash
python3 monitor_training.py
```

### 2. Vizualeaza Graficele
```bash
open results/training_curves_refined.png
```

### 3. Run Full Evaluation
```bash
python3 src/neural_network/evaluate_final.py
```

### 4. Foloseste Model in Web UI
```bash
streamlit run interfata_web.py
```

---

## EXPECTED PERFORMANCE

```
SSIM (pe perechi GOLD):   0.82+ (mean 0.8257)
ORB Matches:              80+ (mean 82.7)
Histogram:                0.94+ (mean 0.9424)

Model Loss:
  Train:   0.05 - 0.15
  Val:     0.10 - 0.20
  Test:    0.10 - 0.20
```

---

## ✓ READY TO START

```bash
cd /Users/admin/Documents/Facultatea/Proiect_RN
python3 src/neural_network/train_final_refined.py
```

**Happy training! 🚀**
