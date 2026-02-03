# TRAIN_FINAL_REFINED - Documentatie Antrenare

**Data**: 22 Ianuarie 2026  
**Dataset**: 190 perechi certificate (122 train, 34 val, 34 test)  
**Status**: 🚀 Lansat

---

## Configuratie Antrenare

### Dataset
```
Train:       122 perechi (64.2%)
Validation:  34 perechi (17.9%)
Test:        34 perechi (17.9%)
Total:       190 perechi (Gold Standard - SSIM > 0.70)
```

### Hiperparametri
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 16 | Mic pentru stabilitate pe dataset mic |
| **Epochs** | 100 | Putem rula mai repede pe 190 perechi |
| **Learning Rate** | 1e-4 | Conservator, evita overfitting |
| **Optimizer** | Adam | Adaptive, bun pentru dataset mic |
| **Weight Decay** | 1e-5 | Regularizare |
| **Scheduler** | ReduceLROnPlateau | Scade LR cand val_loss plateleaza |
| **Patience (Scheduler)** | 5 | Scade LR dupa 5 epoci fara imbunatare |
| **Early Stopping** | 15 epoci | Opreste antrenarea daca nu se imbunatateste |

---

## Data Augmentation

### Transformari Aplicate (Training Only)

1. **Rotatie Aleatorie**: ±30 grade
   - Imbunatateste robustete la diferite unghiuri de captare
   
2. **Flip Orizontal**: 50% probabilitate
   - Simetriza distributia orientarilor
   
3. **Flip Vertical**: 50% probabilitate
   - Simetriza vertical, relevant pentru detectare schimbari
   
4. **Zoom Aleatoriu**: 0.9x - 1.1x
   - Imita diferente de distanta de captare
   
5. **Ajustari Luminozitate**: 0.8x - 1.2x
   - Rezista la variatie de iluminare
   
6. **Ajustari Contrast**: 0.8x - 1.2x
   - Rezista la variatie de contrast

**Nota**: Toate transformarile se aplica CONSISTENT pe before, after, si mask.

---

## Model Architecture

### UNet Semantic Segmentation
```
Input:  6 channels (3 from before + 3 from after)
Output: 1 channel (binary mask - change detection)

Encoder:
  Conv(6 -> 64) + MaxPool
  Conv(64 -> 128) + MaxPool
  Conv(128 -> 256) + MaxPool

Bottleneck:
  Conv(256 -> 512)

Decoder:
  ConvTranspose(512 -> 256) + Conv -> 256
  ConvTranspose(256 -> 128) + Conv -> 128
  ConvTranspose(128 -> 64) + Conv -> 64

Output:
  Conv(64 -> 1) + Sigmoid
```

**Parameters**: ~7.7M (similar cu versiunea anteriora)

---

## Training Strategy

### Loss Function
- **BCELoss** (Binary Cross Entropy)
- Potrivit pentru problema binara (change/no-change)

### Gradient Clipping
- **Max norm**: 1.0
- Previne exploding gradients pe dataset mic

### Validation
- Ruleaza dupa fiecare epoca
- Calculeaza validation loss pe 34 imagini
- Salveaza best_model daca val_loss se imbunatateste

### Early Stopping
- Monitorizeaza validation loss
- Oprete antrenarea dupa 15 epoci fara imbunatare
- Previne overfitting pe dataset mic

### Learning Rate Scheduler
- **ReduceLROnPlateau**: scade LR cu factor 0.5
- Dupa 5 epoci fara imbunatare pe val set
- Minimum LR: 1e-6

---

## Fisiere Generate

### 1. **best_model_refined.pth** (Checkpoint)
```
Locatie: checkpoints/best_model_refined.pth
Continut: State dict al modelului cu cea mai mica validation loss
Dimensiune: ~30 MB (7.7M parameters * 4 bytes)
```

### 2. **training_results_refined.json** (Metrici)
```json
{
  "timestamp": "2026-01-22T...",
  "config": {...},
  "best_val_loss": 0.1234,
  "test_loss": 0.1456,
  "final_epoch": 45,
  "history": {
    "train_loss": [...],
    "val_loss": [...],
    "learning_rate": [...]
  }
}
```

### 3. **training_curves_refined.png** (Vizualizare)
```
Grafice:
  - Training Loss vs Validation Loss
  - Learning Rate Evolution
```

---

## Comenzi Utile

### Lansare antrenare
```bash
cd /Users/admin/Documents/Facultatea/Proiect_RN
python3 src/neural_network/train_final_refined.py
```

### Monitor progres (in alta termina)
```bash
python3 monitor_training.py
```

### Verifica rezultate
```bash
python3 -c "
import json
with open('results/training_results_refined.json') as f:
    r = json.load(f)
    print(f'Best val loss: {r[\"best_val_loss\"]:.4f}')
    print(f'Test loss: {r[\"test_loss\"]:.4f}')
    print(f'Total epochs: {r[\"final_epoch\"]}')
"
```

---

## Timeline Asteptat

### Pe Mac M1 cu batch_size=16
```
Dataset: 190 perechi (122 train, 34 val, 34 test)
Batch size: 16

Per epoch:
  - Train batches: ~8 (122 / 16)
  - Val batches: ~3 (34 / 16)
  - Timp pe epoca: ~5-10 secunde

Total runtime estimat:
  - Cu early stopping la epoca 40-50: 5-10 minute
  - Worst case (100 epoci): 15-20 minute
```

---

## Expected Results

### Performance Asteptata
```
SSIM (test set):          0.82+
ORB Matches (test set):   80+
Histogram (test set):     0.94+

Loss Target:
  Train loss: 0.05-0.15
  Val loss: 0.10-0.20
  Test loss: 0.10-0.20
```

### Convergence Pattern
```
Epoci 1-5:    Rapid descent
Epoci 5-20:   Moderate descent
Epoci 20+:    Plateau (early stopping activat)
```

---

## Debugging

### Daca antrenarea e lenta
1. Verifica `ps aux | grep python3` pentru alte procese
2. Scade `batch_size` la 8
3. Verifica dataset cu `ls data/train/before | wc -l`

### Daca training loss nu descreste
1. Verifica learning rate: ar trebui sa inceapa cu 1e-4
2. Verifica data: sunt imaginile incarcate corect?
3. Verifica transformarile: sunt deterministe?

### Daca val_loss creste
1. Overfitting - aug mentionez ca este aplicat doar pe train
2. Early stopping ar trebui sa opreasca dupa 15 epoci

---

## Post-Training

### 1. Verifica Rezultate
```bash
python3 monitor_training.py
```

### 2. Vizualeaza Grafice
```bash
open results/training_curves_refined.png
```

### 3. Evaluare Completa
```bash
python3 src/neural_network/evaluate_final.py
```

### 4. Foloseste Model in Web UI
```bash
streamlit run interfata_web.py
```

---

## Optimi Urmatoare (Dupa Antrenare)

1. **Fine-tuning pe full dataset**
   - Dupa ce validezi pe 190 perechi
   - Poti adauga si perechi din REJECTED_BY_AUDIT

2. **Data Augmentation Dynamics**
   - Inverseaza: aplica aug pe val/test deasemenea
   - Compara rezultate

3. **Hiperparametri Fine-tune**
   - Daca overfitting: mai multa augmentare
   - Daca convergenta lenta: mareste learning rate

---

**Status**: 🚀 Training initialized and ready to run
