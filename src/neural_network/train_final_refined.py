#!/usr/bin/env python3
"""
TRAIN_FINAL_REFINED - THRESHOLD OPTIMIZATION
==============================================
Issue: Model has 92% Accuracy but 34% Recall (too conservative at threshold=0.5)

Solution: 
1. Find optimal threshold that maximizes F1-Score on validation set
2. Apply optimized threshold to final test evaluation
3. Save results with optimized metrics

Strategy:
- Test thresholds from 0.1 to 0.5 (step 0.05)
- Select threshold with highest F1-Score on validation
- Final test evaluation uses optimized threshold
- Report both metrics in final_metrics.json
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import csv

# ============================================================================
# CONFIGURARE - PATH-URI RELATIVE (Portabil)
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
RESULTS_DIR = SCRIPT_DIR / "results"
MODELS_DIR = SCRIPT_DIR / "models"

CONFIG = {
    'data_dir': str(DATA_DIR),
    'model_save_dir': str(CHECKPOINTS_DIR),
    'results_dir': str(RESULTS_DIR),
    'pretrained_model': str(MODELS_DIR / "optimized_model_v2.pt"),
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 2e-4,
    'patience': 15,
    'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'threshold': 0.5,  # Default, will be optimized
}

device = torch.device(CONFIG['device'])
print(f"\n📱 Device: {device}")

# ============================================================================
# DATASET (SAME AS BEFORE)
# ============================================================================

class AugmentedChangeDetectionDataset(Dataset):
    """Dataset cu augmentare FULL SPECTRUM"""
    
    def __init__(self, root_dir, split='train', augment=True):
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        self.augment = augment
        
        self.before_dir = self.split_dir / 'before'
        self.after_dir = self.split_dir / 'after'
        self.mask_dir = self.split_dir / 'masks'
        
        self.files = sorted([f.name for f in self.before_dir.iterdir() 
                            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

        if self.augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        
        before = Image.open(self.before_dir / filename)
        after = Image.open(self.after_dir / filename)
        mask = Image.open(self.mask_dir / filename).convert('L')
        
        before_np = np.array(before, dtype=np.float32) / 255.0
        after_np = np.array(after, dtype=np.float32) / 255.0
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        
        x = np.concatenate([before_np, after_np], axis=2)
        
        if self.transform:
            before_pil = Image.fromarray((before_np * 255).astype(np.uint8))
            before_aug = np.array(self.transform(before_pil), dtype=np.float32) / 255.0
            
            after_pil = Image.fromarray((after_np * 255).astype(np.uint8))
            after_aug = np.array(self.transform(after_pil), dtype=np.float32) / 255.0
            
            x = np.concatenate([before_aug, after_aug], axis=2)
        
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        return x, mask


# ============================================================================
# MODEL (SAME AS BEFORE)
# ============================================================================

class UNet(nn.Module):
    """UNet architecture"""
    
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()
        
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1_out = self.enc1(x)
        enc1_pool = self.pool1(enc1_out)
        
        enc2_out = self.enc2(enc1_pool)
        enc2_pool = self.pool2(enc2_out)
        
        enc3_out = self.enc3(enc2_pool)
        enc3_pool = self.pool3(enc3_out)
        
        bottleneck_out = self.bottleneck(enc3_pool)
        
        dec3_up = self.upconv3(bottleneck_out)
        dec3_concat = torch.cat([dec3_up, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_concat)
        
        dec2_up = self.upconv2(dec3_out)
        dec2_concat = torch.cat([dec2_up, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_concat)
        
        dec1_up = self.upconv1(dec2_out)
        dec1_concat = torch.cat([dec1_up, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_concat)
        
        out = self.out(dec1_out)
        
        return out


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate ALL metrics: Accuracy, Precision, Recall, F1, IoU
    """
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


# ============================================================================
# THRESHOLD OPTIMIZATION - CORE FUNCTION
# ============================================================================

def find_optimal_threshold(model, val_loader, device, threshold_range=np.arange(0.1, 0.55, 0.05)):
    """
    Find optimal threshold that maximizes F1-Score on validation set
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: torch device
        threshold_range: List of thresholds to test
    
    Returns:
        dict with optimal threshold and its metrics
    """
    
    print("\n" + "="*90)
    print("🔍 THRESHOLD OPTIMIZATION - Finding Optimal F1-Score")
    print("="*90)
    print(f"\nTesting thresholds: {[f'{t:.2f}' for t in threshold_range]}")
    
    model.eval()
    
    # Store all predictions and targets for threshold testing
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred_prob = torch.sigmoid(outputs)
            
            all_predictions.append(pred_prob.cpu())
            all_targets.append(masks.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Test each threshold
    results = []
    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = None
    
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'IoU':<12}")
    print("-" * 90)
    
    for threshold in threshold_range:
        metrics = calculate_metrics(all_predictions, all_targets, threshold=threshold)
        
        results.append({
            'threshold': float(threshold),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'iou': metrics['iou'],
        })
        
        # Print this threshold's results
        status = "⭐" if metrics['f1'] > best_f1 else ""
        print(f"{threshold:<12.2f} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['iou']:<12.4f} {status}")
        
        # Check if this is the best F1-Score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            best_metrics = metrics
    
    print("-" * 90)
    print(f"\n🏆 OPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"   F1-Score: {best_metrics['f1']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   IoU: {best_metrics['iou']:.4f}")
    
    return {
        'optimal_threshold': best_threshold,
        'metrics': best_metrics,
        'all_results': results
    }


# ============================================================================
# FINAL EVALUATION WITH OPTIMIZED THRESHOLD
# ============================================================================

def evaluate_with_threshold(model, test_loader, device, threshold=0.5):
    """
    Evaluate model on test set with specified threshold
    """
    model.eval()
    all_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'accuracy': []
    }
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred_prob = torch.sigmoid(outputs)
            
            batch_metrics = calculate_metrics(pred_prob, masks, threshold=threshold)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
    
    avg_metrics = {key: np.mean(vals) for key, vals in all_metrics.items()}
    
    return avg_metrics


# ============================================================================
# MAIN - THRESHOLD OPTIMIZATION & FINAL EVALUATION
# ============================================================================

def main():
    print("\n" + "="*90)
    print("🎯 THRESHOLD OPTIMIZATION - FIX RECALL ISSUE")
    print("="*90)
    print("\nProblem: 92% Accuracy but only 34% Recall (threshold=0.5 too conservative)")
    print("Solution: Find optimal threshold that maximizes F1-Score")
    print("="*90)
    
    # Create directories
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # ========================================================================
    # LOAD DATASETS
    # ========================================================================
    print("\n[1] Loading datasets...")
    val_dataset = AugmentedChangeDetectionDataset(CONFIG['data_dir'], 'validation', augment=False)
    test_dataset = AugmentedChangeDetectionDataset(CONFIG['data_dir'], 'test', augment=False)
    
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                           num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'])
    
    print(f"  ✅ Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  ✅ Test: {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    # ========================================================================
    # LOAD PRE-TRAINED MODEL
    # ========================================================================
    print("\n[2] Loading pre-trained model...")
    model = UNet(in_channels=6, out_channels=1).to(device)
    
    best_model_path = os.path.join(CONFIG['model_save_dir'], 'best_model_ultimate.pth')
    
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"  ✅ Best model loaded from {best_model_path}")
        except Exception as e:
            print(f"  ❌ Error loading best model: {e}")
            print(f"  Trying to load from optimized_model_v2.pt...")
            if os.path.exists(CONFIG['pretrained_model']):
                checkpoint = torch.load(CONFIG['pretrained_model'], map_location=device)
                model.load_state_dict(checkpoint)
                print(f"  ✅ Model loaded from {CONFIG['pretrained_model']}")
            else:
                print(f"  ❌ No pretrained model found!")
                return
    else:
        print(f"  ⚠️  Best model checkpoint not found at {best_model_path}")
        if os.path.exists(CONFIG['pretrained_model']):
            checkpoint = torch.load(CONFIG['pretrained_model'], map_location=device)
            model.load_state_dict(checkpoint)
            print(f"  ✅ Loaded from {CONFIG['pretrained_model']}")
        else:
            print(f"  ❌ No model found!")
            return
    
    # ========================================================================
    # STEP 1: FIND OPTIMAL THRESHOLD ON VALIDATION SET
    # ========================================================================
    print("\n[3] Finding optimal threshold...")
    threshold_results = find_optimal_threshold(
        model, val_loader, device, 
        threshold_range=np.arange(0.1, 0.55, 0.05)
    )
    
    optimal_threshold = threshold_results['optimal_threshold']
    
    # ========================================================================
    # STEP 2: EVALUATE ON TEST SET WITH OPTIMIZED THRESHOLD
    # ========================================================================
    print("\n[4] Evaluating on TEST set with optimized threshold...")
    print(f"\nUsing optimal threshold: {optimal_threshold:.2f}")
    
    test_metrics = evaluate_with_threshold(model, test_loader, device, threshold=optimal_threshold)
    
    print(f"\n📊 TEST RESULTS (with optimal threshold {optimal_threshold:.2f}):")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f} ⭐ (was 34%, now should be higher)")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    
    # Check targets
    acc_pass = "✅ PASS" if test_metrics['accuracy'] > 0.70 else "❌ FAIL"
    rec_pass = "✅ PASS" if test_metrics['recall'] > 0.65 else "⚠️ BELOW TARGET"
    f1_pass = "✅ PASS" if test_metrics['f1'] > 0.65 else "⚠️ BELOW TARGET"
    
    print(f"\n📋 TARGET CHECK:")
    print(f"  Accuracy > 70%: {test_metrics['accuracy']:.4f} {acc_pass}")
    print(f"  Recall > 65%: {test_metrics['recall']:.4f} {rec_pass}")
    print(f"  F1-Score > 0.65: {test_metrics['f1']:.4f} {f1_pass}")
    
    # ========================================================================
    # STEP 3: SAVE FINAL METRICS WITH OPTIMIZED THRESHOLD
    # ========================================================================
    
    final_metrics = {
        'timestamp': datetime.now().isoformat(),
        'model_name': 'optimized_model_v2.pt',
        'optimization_phase': 'Threshold Optimization (No Retraining)',
        'threshold_optimization': {
            'method': 'Grid search on validation set (0.1 to 0.5, step 0.05)',
            'optimal_threshold': float(optimal_threshold),
            'threshold_search_results': threshold_results['all_results'],
        },
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1_score': float(test_metrics['f1']),
            'iou': float(test_metrics['iou']),
        },
        'targets_achieved': {
            'accuracy_gt_70': float(test_metrics['accuracy']) > 0.70,
            'recall_gt_65': float(test_metrics['recall']) > 0.65,
            'f1_gt_65': float(test_metrics['f1']) > 0.65,
        },
        'configuration': {
            'threshold': float(optimal_threshold),
            'device': str(device),
            'batch_size': CONFIG['batch_size'],
        },
        'dataset': {
            'validation_samples': len(val_dataset),
            'test_samples': len(test_dataset),
        },
    }
    
    # Save final metrics
    metrics_file = os.path.join(CONFIG['results_dir'], 'final_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n✅ Final metrics saved to {metrics_file}")
    
    # ========================================================================
    # STEP 4: CREATE VISUALIZATION - THRESHOLD COMPARISON
    # ========================================================================
    
    print(f"\n[5] Creating threshold optimization visualization...")
    
    threshold_data = threshold_results['all_results']
    thresholds = [r['threshold'] for r in threshold_data]
    accuracies = [r['accuracy'] for r in threshold_data]
    precisions = [r['precision'] for r in threshold_data]
    recalls = [r['recall'] for r in threshold_data]
    f1_scores = [r['f1'] for r in threshold_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy vs Threshold
    axes[0, 0].plot(thresholds, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    axes[0, 0].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.2f}')
    axes[0, 0].set_xlabel('Threshold', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Accuracy vs Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Precision vs Threshold
    axes[0, 1].plot(thresholds, precisions, marker='s', linewidth=2, markersize=8, color='green')
    axes[0, 1].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.2f}')
    axes[0, 1].set_xlabel('Threshold', fontsize=11)
    axes[0, 1].set_ylabel('Precision', fontsize=11)
    axes[0, 1].set_title('Precision vs Threshold', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Recall vs Threshold - PRIMARY FIX
    axes[1, 0].plot(thresholds, recalls, marker='^', linewidth=2, markersize=8, color='orange')
    axes[1, 0].axhline(y=0.65, color='red', linestyle=':', linewidth=2, label='Target: 0.65')
    axes[1, 0].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.2f}')
    axes[1, 0].set_xlabel('Threshold', fontsize=11)
    axes[1, 0].set_ylabel('Recall', fontsize=11)
    axes[1, 0].set_title('🎯 Recall vs Threshold (PRIMARY FIX)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # F1-Score vs Threshold - OPTIMIZATION METRIC
    axes[1, 1].plot(thresholds, f1_scores, marker='D', linewidth=2.5, markersize=8, color='purple')
    axes[1, 1].axhline(y=0.65, color='red', linestyle=':', linewidth=2, label='Target: 0.65')
    axes[1, 1].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.2f}')
    optimal_f1_idx = thresholds.index(optimal_threshold)
    axes[1, 1].plot(optimal_threshold, f1_scores[optimal_f1_idx], 'r*', markersize=20, label=f'Max F1: {f1_scores[optimal_f1_idx]:.4f}')
    axes[1, 1].set_xlabel('Threshold', fontsize=11)
    axes[1, 1].set_ylabel('F1-Score', fontsize=11)
    axes[1, 1].set_title('🏆 F1-Score vs Threshold (OPTIMIZATION METRIC)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle(f'Threshold Optimization - Optimal Threshold: {optimal_threshold:.2f}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plot_file = os.path.join(CONFIG['results_dir'], 'threshold_optimization.png')
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    print(f"✅ Threshold optimization plot saved to {plot_file}")
    plt.close()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*90)
    print("✅ THRESHOLD OPTIMIZATION COMPLETE")
    print("="*90)
    print(f"\n📊 SUMMARY:")
    print(f"  Original Threshold: 0.50")
    print(f"  Optimal Threshold: {optimal_threshold:.2f}")
    print(f"\n  Test Metrics (with optimal threshold):")
    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall: {test_metrics['recall']:.4f} ⭐ (increased from 34%)")
    print(f"    F1-Score: {test_metrics['f1']:.4f}")
    print(f"    IoU: {test_metrics['iou']:.4f}")
    print(f"\n✅ Results saved to {metrics_file}")
    print(f"✅ Visualization saved to {plot_file}")
    print(f"\n🎯 Ready for final submission with optimized threshold!")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()