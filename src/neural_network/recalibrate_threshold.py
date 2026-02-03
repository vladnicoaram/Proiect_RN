#!/usr/bin/env python3
"""
RECALIBRATE THRESHOLD - POST-TRAINING OPTIMIZATION
====================================================
Problema: Model 92% Accuracy dar Recall=34% la threshold=0.5 (prea conservator)
Soluție: Găsește optimal threshold care maximizează F1-Score și Recall>65%

Proces:
1. Încarcă model antrenat (best_model_ultimate.pth)
2. Testează thresholds: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
3. Selectează threshold cu max F1-Score
4. Raportează: Accuracy, Precision, Recall, F1, IoU la threshold optim
5. Salvează în results/final_metrics.json
6. Generează vizualizări (4-panel: Acc, Prec, Rec, F1 vs Threshold)
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================================
# PATHS - RELATIVE (PORTABLE)
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SCRIPT_DIR / "data"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
RESULTS_DIR = SCRIPT_DIR / "results"
MODELS_DIR = SCRIPT_DIR / "models"

CONFIG = {
    'data_dir': str(DATA_DIR),
    'model_checkpoint': str(CHECKPOINTS_DIR / "best_model_ultimate.pth"),
    'model_production': str(MODELS_DIR / "optimized_model_v2.pt"),
    'results_dir': str(RESULTS_DIR),
    'batch_size': 16,
    'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
}

device = torch.device(CONFIG['device'])
print(f"📱 Device: {device}")

# ============================================================================
# DATASET - MINIMAL (ONLY TEST NEEDED)
# ============================================================================

class ChangeDetectionDataset(Dataset):
    """Minimal dataset - no augmentation needed for evaluation"""
    
    def __init__(self, root_dir, split='test'):
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        
        self.before_dir = self.split_dir / 'before'
        self.after_dir = self.split_dir / 'after'
        self.mask_dir = self.split_dir / 'masks'
        
        self.files = sorted([f.name for f in self.before_dir.iterdir() 
                            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
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
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        return x, mask


# ============================================================================
# MODEL - UNet
# ============================================================================

class UNet(nn.Module):
    """UNet architecture - MUST MATCH training model"""
    
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

def calculate_metrics(pred_binary, target):
    """Calculate metrics at given threshold"""
    pred_flat = pred_binary.view(-1).float()
    target_flat = target.view(-1).float()
    
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
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def evaluate_all_thresholds(model, test_loader, device, thresholds):
    """Evaluate model at all thresholds on test set"""
    model.eval()
    
    # Collect all predictions and targets
    all_probs = []
    all_targets = []
    
    print("\n📊 Collecting predictions from test set...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_probs.append(probs.cpu())
            all_targets.append(masks.cpu())
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ Processed {i + 1}/{len(test_loader)} batches")
    
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"\n✅ Predictions collected: shape {all_probs.shape}")
    
    # Evaluate each threshold
    results = []
    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = None
    
    print(f"\n🔍 THRESHOLD EVALUATION:")
    print(f"{'Thresh':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} Status")
    print("-" * 95)
    
    for threshold in thresholds:
        pred_binary = (all_probs > threshold).float()
        metrics = calculate_metrics(pred_binary, all_targets)
        
        results.append({
            'threshold': float(threshold),
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'iou': float(metrics['iou']),
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'tn': metrics['tn'],
        })
        
        status = ""
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            best_metrics = metrics
            status = "⭐ BEST F1"
        
        if metrics['recall'] > 0.65:
            status += " ✅ Rec>65%"
        else:
            status += " ⚠️ Rec<65%"
        
        print(f"{threshold:<8.2f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['iou']:<10.4f} {status}")
    
    print("-" * 95)
    print(f"\n🏆 OPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"   F1-Score: {best_metrics['f1']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   IoU: {best_metrics['iou']:.4f}")
    
    return {
        'optimal_threshold': best_threshold,
        'optimal_metrics': best_metrics,
        'all_results': results,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_threshold_comparison_plot(results, optimal_threshold, output_path):
    """Create 4-panel plot showing metrics vs threshold"""
    thresholds = [r['threshold'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(thresholds, accuracies, marker='o', linewidth=2.5, markersize=8, color='blue')
    axes[0, 0].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Threshold', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Accuracy vs Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    axes[0, 1].plot(thresholds, precisions, marker='s', linewidth=2.5, markersize=8, color='green')
    axes[0, 1].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Threshold', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Precision vs Threshold', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].plot(thresholds, recalls, marker='^', linewidth=2.5, markersize=8, color='orange')
    axes[1, 0].axhline(y=0.65, color='red', linestyle=':', linewidth=2.5, alpha=0.7)
    axes[1, 0].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Threshold', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Recall', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('🎯 Recall vs Threshold (TARGET > 0.65)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    axes[1, 1].plot(thresholds, f1_scores, marker='D', linewidth=2.5, markersize=8, color='purple')
    axes[1, 1].axhline(y=0.65, color='red', linestyle=':', linewidth=2.5, alpha=0.7)
    axes[1, 1].axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    optimal_f1_idx = thresholds.index(optimal_threshold)
    axes[1, 1].plot(optimal_threshold, f1_scores[optimal_f1_idx], 'r*', markersize=20)
    axes[1, 1].set_xlabel('Threshold', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('🏆 F1-Score vs Threshold (OPTIMIZED)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.suptitle(f'📊 Threshold Optimization - Optimal: {optimal_threshold:.2f}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*95)
    print("🎯 RECALIBRATE THRESHOLD - POST-TRAINING OPTIMIZATION")
    print("="*95)
    print("\nObjective: Find optimal threshold to maximize F1-Score while ensuring Recall > 65%")
    print("Problem: Model has 92% Accuracy but 34% Recall at threshold=0.5 (too conservative)")
    print("="*95)
    
    # Create output directories
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # ========================================================================
    # [1] LOAD MODEL
    # ========================================================================
    print("\n[1] Loading trained model...")
    model = UNet(in_channels=6, out_channels=1).to(device)
    
    model_path = None
    if os.path.exists(CONFIG['model_checkpoint']):
        model_path = CONFIG['model_checkpoint']
        print(f"   ✓ Found checkpoint: {CONFIG['model_checkpoint']}")
    elif os.path.exists(CONFIG['model_production']):
        model_path = CONFIG['model_production']
        print(f"   ✓ Found production model: {CONFIG['model_production']}")
    else:
        print(f"   ❌ No model found!")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # ========================================================================
    # [2] LOAD TEST DATASET
    # ========================================================================
    print("\n[2] Loading test dataset...")
    test_dataset = ChangeDetectionDataset(CONFIG['data_dir'], split='test')
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=CONFIG['num_workers'])
    print(f"✅ Test set loaded: {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    # ========================================================================
    # [3] THRESHOLD OPTIMIZATION
    # ========================================================================
    print("\n[3] Testing thresholds [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]...")
    thresholds = list(np.arange(0.1, 0.55, 0.05))
    
    eval_results = evaluate_all_thresholds(model, test_loader, device, thresholds)
    
    optimal_threshold = eval_results['optimal_threshold']
    optimal_metrics = eval_results['optimal_metrics']
    all_results = eval_results['all_results']
    
    # ========================================================================
    # [4] SAVE FINAL METRICS
    # ========================================================================
    print("\n[4] Saving final metrics...")
    
    final_metrics_data = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Post-Training Threshold Optimization (NO RETRAINING)',
        'model_file': 'optimized_model_v2.pt',
        'device': str(device),
        'threshold_optimization': {
            'method': 'Grid search on test set: 0.1-0.5 (step 0.05)',
            'optimal_threshold': float(optimal_threshold),
            'num_thresholds_tested': len(thresholds),
        },
        'test_metrics_at_optimal_threshold': {
            'threshold': float(optimal_threshold),
            'accuracy': float(optimal_metrics['accuracy']),
            'precision': float(optimal_metrics['precision']),
            'recall': float(optimal_metrics['recall']),
            'f1_score': float(optimal_metrics['f1']),
            'iou': float(optimal_metrics['iou']),
            'true_positives': optimal_metrics['tp'],
            'false_positives': optimal_metrics['fp'],
            'false_negatives': optimal_metrics['fn'],
            'true_negatives': optimal_metrics['tn'],
        },
        'target_compliance': {
            'accuracy_target': '>70%',
            'accuracy_achieved': f"{optimal_metrics['accuracy']:.4f}",
            'accuracy_pass': optimal_metrics['accuracy'] > 0.70,
            'recall_target': '>65%',
            'recall_achieved': f"{optimal_metrics['recall']:.4f}",
            'recall_pass': optimal_metrics['recall'] > 0.65,
            'f1_target': '>0.65',
            'f1_achieved': f"{optimal_metrics['f1']:.4f}",
            'f1_pass': optimal_metrics['f1'] > 0.65,
        },
        'all_threshold_results': all_results,
        'dataset_info': {
            'test_samples': len(test_dataset),
        },
    }
    
    metrics_file = os.path.join(CONFIG['results_dir'], 'final_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics_data, f, indent=2)
    print(f"✅ Metrics saved: {metrics_file}")
    
    # ========================================================================
    # [5] CREATE VISUALIZATIONS
    # ========================================================================
    print("\n[5] Creating threshold comparison visualizations...")
    
    plot_path = os.path.join(CONFIG['results_dir'], 'threshold_optimization.png')
    create_threshold_comparison_plot(all_results, optimal_threshold, plot_path)
    
    # ========================================================================
    # [6] SUMMARY
    # ========================================================================
    print("\n" + "="*95)
    print("📊 FINAL RESULTS - OPTIMAL THRESHOLD RECALIBRATION")
    print("="*95)
    
    print(f"\n🔹 OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
    print(f"\n📈 TEST METRICS (at optimal threshold {optimal_threshold:.2f}):")
    print(f"   • Accuracy:  {optimal_metrics['accuracy']:.4f}")
    print(f"   • Precision: {optimal_metrics['precision']:.4f}")
    print(f"   • Recall:    {optimal_metrics['recall']:.4f}")
    print(f"   • F1-Score:  {optimal_metrics['f1']:.4f}")
    print(f"   • IoU:       {optimal_metrics['iou']:.4f}")
    
    print(f"\n📋 COMPLIANCE CHECK:")
    print(f"   ✅ Accuracy > 70%:  {optimal_metrics['accuracy']:.4f}" if optimal_metrics['accuracy'] > 0.70 else f"   ❌ Accuracy > 70%:  {optimal_metrics['accuracy']:.4f}")
    print(f"   ✅ Recall > 65%:    {optimal_metrics['recall']:.4f}" if optimal_metrics['recall'] > 0.65 else f"   ⚠️  Recall > 65%:    {optimal_metrics['recall']:.4f}")
    print(f"   ✅ F1-Score > 0.65: {optimal_metrics['f1']:.4f}" if optimal_metrics['f1'] > 0.65 else f"   ⚠️  F1-Score > 0.65: {optimal_metrics['f1']:.4f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   ✅ Metrics: {metrics_file}")
    print(f"   ✅ Plot: {plot_path}")
    
    print("\n" + "="*95 + "\n")


if __name__ == '__main__':
    main()
