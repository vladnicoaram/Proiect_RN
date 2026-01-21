import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import csv
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.neural_network.dataset import ChangeDetectionDataset 
from src.neural_network.model import UNet

# ============================================================================
# CONFIGURARE GPU - MAC M1 MPS
# ============================================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("=" * 80)
    print("✅ GPU MPS DETECTAT - Antrenarea va rula pe Apple Metal Performance Shaders")
    print("=" * 80)
else:
    DEVICE = torch.device('cpu')
    print("=" * 80)
    print("⚠️  WARNING: MPS NU DISPONIBIL - Antrenarea va rula pe CPU (MULT MAI LENTĂ!)")
    print("=" * 80)

# ============================================================================
# LOSS FUNCTIONS - FOCAL LOSS PENTRU OBIECTE MICI
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss - special pentru obiecte mici
    Focusează pe hard examples (pixeli dificili de clasificat)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        # Convertește la probabilități
        p = torch.sigmoid(pred)
        
        # Calculează focal weight
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Aplică focal weight
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss - Optimă pentru segmentare"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class RefinedHybridLoss(nn.Module):
    """Loss Hibrid Rafinat: FocalLoss + Dice (ponderi optime pentru obiecte mici)"""
    def __init__(self, focal_weight=0.6, dice_weight=0.4):
        super(RefinedHybridLoss, self).__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice(pred, target)
        
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total_loss

# ============================================================================
# METRICE
# ============================================================================

def calculate_iou(pred, target):
    """Intersection over Union"""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()
    return (intersection / (union + 1e-6)).item()

def calculate_dice(pred, target):
    """Dice Coefficient"""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_bin * target).sum()
    dice = (2.0 * intersection) / (pred_bin.sum() + target.sum() + 1e-6)
    return dice.item()

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Antrenare pe o epocă"""
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculează metrici
        iou = calculate_iou(logits, y)
        dice = calculate_dice(logits, y)
        
        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    avg_dice = total_dice / len(loader)
    
    return avg_loss, avg_iou, avg_dice

def validate(model, loader, criterion, device):
    """Validare"""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            iou = calculate_iou(logits, y)
            dice = calculate_dice(logits, y)
            
            total_loss += loss.item()
            total_iou += iou
            total_dice += dice
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    avg_dice = total_dice / len(loader)
    
    return avg_loss, avg_iou, avg_dice

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("🚀 ANTRENARE RAFINATĂ - Detection Obiecte Mici & Mari")
    print("=" * 80)
    
    # HYPERPARAMETRI
    EPOCHS = 60
    BATCH_SIZE = 16
    LR = 1e-4
    PATIENCE = 5  # Scheduler patience
    GRADIENT_CLIP = 1.0
    
    print(f"\n📋 CONFIGURAȚIE:")
    print(f"   Device: {DEVICE}")
    print(f"   Epoci: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LR}")
    print(f"   Loss Rafinat: FocalLoss (60%) + Dice (40%)")
    print(f"   Scheduler: ReduceLROnPlateau (factor=0.5, patience={PATIENCE})")
    print(f"   Fokus: Detectie obiecte MICI (haine, genți, măsuțe)")
    
    # DIRECTOARE
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # ========================================================================
    # DATASET & DATALOADER
    # ========================================================================
    print(f"\n📂 Încărcare dataset...")
    
    train_ds = ChangeDetectionDataset('data/train', augment=True)
    val_ds = ChangeDetectionDataset('data/validation', augment=False)
    
    print(f"   Train: {len(train_ds)} imagini (cu augmentare)")
    print(f"   Val:   {len(val_ds)} imagini (fără augmentare)")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # ========================================================================
    # MODEL
    # ========================================================================
    print(f"\n🧠 Inițializare model (rafinat)...")
    
    model = UNet(in_channels=6, out_channels=1).to(DEVICE)
    
    # Numără parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parametri: {total_params:,}")
    print(f"   Parametri antrenabili: {trainable_params:,}")
    
    # ========================================================================
    # OPTIMIZER & LOSS
    # ========================================================================
    print(f"\n⚙️  Configurare optimizer & loss (FOCAL + DICE)...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=PATIENCE
    )
    criterion = RefinedHybridLoss(focal_weight=0.6, dice_weight=0.4)
    
    print(f"   Optimizer: Adam (lr={LR})")
    print(f"   Scheduler: ReduceLROnPlateau (factor=0.5, patience={PATIENCE})")
    print(f"   Loss: FocalLoss (60%) + DiceLoss (40%)")
    print(f"   Goal: Precision >60% + Recall >90% (detectie obiecte mici)")
    
    # ========================================================================
    # ANTRENARE
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("🏋️  INCEPE ANTRENAREA RAFINATĂ")
    print("=" * 80 + "\n")
    
    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = datetime.now()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = datetime.now()
        
        # Train
        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        
        # Validate
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Salvează history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_iou': train_iou,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Timp per epocă
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        # Print
        print(f"\n📊 Epoch {epoch}/{EPOCHS} ({epoch_time:.1f}s) | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Salvează best checkpoint
            checkpoint_path = f"checkpoints/best_refined_epoch{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   ✅ BEST MODEL SALVAT ({val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   ⚠️  Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE * 3:  # Triple patience for extended learning
                print(f"\n⏹️  EARLY STOPPING la epocul {epoch}")
                break
    
    # ========================================================================
    # REZULTATE FINALE
    # ========================================================================
    total_time = (datetime.now() - start_time).total_seconds() / 60  # în minute
    
    print("\n" + "=" * 80)
    print("✅ ANTRENARE RAFINATĂ COMPLETĂ")
    print("=" * 80)
    print(f"\n⏱️  Timp total: {total_time:.1f} minute ({total_time/60:.1f} ore)")
    print(f"📈 Epoci antrenate: {len(history)}/{EPOCHS}")
    
    if history:
        best_epoch_idx = np.argmin([h['val_loss'] for h in history])
        best_epoch = history[best_epoch_idx]
        
        print(f"\n🏆 BEST MODEL (Epocul {best_epoch['epoch']}):")
        print(f"   Val Loss: {best_epoch['val_loss']:.4f}")
        print(f"   Val IoU:  {best_epoch['val_iou']:.4f}")
        print(f"   Val Dice: {best_epoch['val_dice']:.4f}")
        print(f"   LR:       {best_epoch['lr']:.6f}")
    
    # ========================================================================
    # SALVARE MODEL & HISTORY
    # ========================================================================
    
    # Salvează model final
    final_model_path = "models/unet_refined_small_objects.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n💾 Model final rafinat salvat: {final_model_path}")
    
    # Salvează history CSV
    history_path = "results/training_history_refined.csv"
    with open(history_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"📊 History salvat: {history_path}")
    
    # ========================================================================
    # STATISTICI FINALE
    # ========================================================================
    
    print(f"\n" + "=" * 80)
    print("📋 STATISTICI FINALE - RAFINARE")
    print("=" * 80)
    
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    val_ious = [h['val_iou'] for h in history]
    val_dices = [h['val_dice'] for h in history]
    
    print(f"\nTrain Loss:")
    print(f"   Initial: {train_losses[0]:.4f}")
    print(f"   Final:   {train_losses[-1]:.4f}")
    print(f"   Scădere: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%")
    
    print(f"\nVal Loss:")
    print(f"   Initial: {val_losses[0]:.4f}")
    print(f"   Final:   {val_losses[-1]:.4f}")
    print(f"   Best:    {min(val_losses):.4f}")
    
    print(f"\nVal Metrici (Best):")
    print(f"   Best IoU:  {max(val_ious):.4f}")
    print(f"   Best Dice: {max(val_dices):.4f}")
    
    print(f"\n" + "=" * 80)
    print("🎉 RAFINARE COMPLETĂ!")
    print("=" * 80)
    print(f"\n⏭️  Următorul pas:")
    print(f"   Evaluează modelul: python src/neural_network/evaluate_refined.py")

if __name__ == "__main__":
    main()
