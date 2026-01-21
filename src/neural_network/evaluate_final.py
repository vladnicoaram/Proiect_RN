import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import json
from pathlib import Path

# Importurile tale specifice
from dataset import ChangeDetectionDataset 
from model import UNet

# ============================================================================
# METRICE
# ============================================================================

def calculate_metrics(pred, target):
    """Calculează metrici: Accuracy, Precision, Recall, F1, IoU"""
    # Threshold la 0.5 pentru a transforma probabilitățile în mască binară
    pred_bin = (pred > 0.5).float()
    
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    
    # Formule pentru metrici
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # IoU (Intersection over Union)
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()
    iou = intersection / (union + 1e-6)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item(),
        'pred_bin': pred_bin
    }

def save_comparison_images(x, y, pred_mask, pred_prob, output_dir, filename_base):
    """Salvează imagini de comparație: before, after, GT, predicted, probability map"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertește tensorii la numpy
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x
    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = y
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.cpu().numpy()
    
    # Before image (primele 3 canale din input)
    before_img = (x_np[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
    before_path = os.path.join(output_dir, f"{filename_base}_01_before.png")
    cv2.imwrite(before_path, cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR))
    
    # After image (ultimele 3 canale din input)
    after_img = (x_np[3:].transpose(1, 2, 0) * 255).astype(np.uint8)
    after_path = os.path.join(output_dir, f"{filename_base}_02_after.png")
    cv2.imwrite(after_path, cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))
    
    # GT Mask
    gt_mask_img = (y_np.squeeze() * 255).astype(np.uint8)
    gt_path = os.path.join(output_dir, f"{filename_base}_03_gt_mask.png")
    cv2.imwrite(gt_path, gt_mask_img)
    
    # Predicted mask (binary)
    pred_mask_img = (pred_mask.squeeze() * 255).astype(np.uint8)
    pred_path = os.path.join(output_dir, f"{filename_base}_04_predicted_mask.png")
    cv2.imwrite(pred_path, pred_mask_img)
    
    # Probability map (heatmap)
    pred_prob_img = (pred_prob.squeeze() * 255).astype(np.uint8)
    pred_prob_color = cv2.applyColorMap(pred_prob_img, cv2.COLORMAP_JET)
    prob_path = os.path.join(output_dir, f"{filename_base}_05_probability_map.png")
    cv2.imwrite(prob_path, pred_prob_color)
    
    # Overlay: GT mask pe after image
    overlay_after = after_img.copy()
    overlay_after[gt_mask_img > 128] = [0, 255, 0]  # Green - GT
    overlay_path = os.path.join(output_dir, f"{filename_base}_06_gt_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_after, cv2.COLOR_RGB2BGR))
    
    # Overlay: Predicted mask pe after image
    overlay_pred = after_img.copy()
    overlay_pred[pred_mask_img > 128] = [0, 0, 255]  # Red - Predicted
    overlay_pred_path = os.path.join(output_dir, f"{filename_base}_07_pred_overlay.png")
    cv2.imwrite(overlay_pred_path, cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))

# ============================================================================
# EVALUARE
# ============================================================================

def evaluate():
    print("=" * 80)
    print("🔍 EVALUARE FINAL - Change Detection Model")
    print("=" * 80)
    
    # Setăm device-ul pe Apple Silicon (MPS)
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n📱 Device: {DEVICE}")
    
    # Calea către modelul antrenat (relativ la root proiect)
    MODEL_PATH = "../../models/unet_final_clean.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Eroare: Nu am găsit modelul la calea: {MODEL_PATH}")
        # Încearcă și alte căi
        if os.path.exists("../../models/unet_final.pth"):
            MODEL_PATH = "../../models/unet_final.pth"
            print(f"   ✓ Găsit model alternativ: {MODEL_PATH}")
        else:
            return

    # Încărcare Model
    print(f"\n📦 Se încarcă modelul din {MODEL_PATH}...")
    model = UNet(6, 1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("   ✓ Model încărcat cu succes")

    # Directoare de salvare
    results_dir = "../../results/evaluation"
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\n📁 Imagini salvate în: {os.path.abspath(vis_dir)}")

    # Încărcare Date Test
    print(f"\n📂 Încărcare test dataset...")
    test_dataset = ChangeDetectionDataset(root_dir="../../data/test", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"   ✓ {len(test_loader)} imagini de test")

    # Metrici
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
    }
    
    detailed_results = []

    print(f"\n🚀 Evaluare pornită...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Inferență
            logits = model(x)
            output = torch.sigmoid(logits)
            
            # Calcul metrici
            metrics = calculate_metrics(output, y)
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            # Salvează imagini pentru fiecare sample
            filename_base = f"{i:04d}"
            save_comparison_images(
                x[0], y[0], metrics['pred_bin'][0], output[0],
                vis_dir, filename_base
            )
            
            detailed_results.append({
                'sample_id': i,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'iou': metrics['iou'],
            })
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ Procesat {i + 1}/{len(test_loader)} imagini...")

    # ========================================================================
    # REZULTATE FINALE
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("📊 REZULTATE FINALE EVALUARE")
    print("=" * 80)
    
    summary = {
        'total_samples': len(test_loader),
        'metrics': {
            'accuracy_mean': float(np.mean(all_metrics['accuracy'])),
            'accuracy_std': float(np.std(all_metrics['accuracy'])),
            'precision_mean': float(np.mean(all_metrics['precision'])),
            'precision_std': float(np.std(all_metrics['precision'])),
            'recall_mean': float(np.mean(all_metrics['recall'])),
            'recall_std': float(np.std(all_metrics['recall'])),
            'f1_mean': float(np.mean(all_metrics['f1'])),
            'f1_std': float(np.std(all_metrics['f1'])),
            'iou_mean': float(np.mean(all_metrics['iou'])),
            'iou_std': float(np.std(all_metrics['iou'])),
        },
        'device': str(DEVICE),
        'model_path': MODEL_PATH,
    }
    
    print(f"\n✅ METRICI (pe {len(test_loader)} imagini):")
    print(f"\n   📈 ACCURACY")
    print(f"      Mean:  {summary['metrics']['accuracy_mean']:.4f}")
    print(f"      Std:   {summary['metrics']['accuracy_std']:.4f}")
    
    print(f"\n   📈 PRECISION")
    print(f"      Mean:  {summary['metrics']['precision_mean']:.4f}")
    print(f"      Std:   {summary['metrics']['precision_std']:.4f}")
    
    print(f"\n   📈 RECALL")
    print(f"      Mean:  {summary['metrics']['recall_mean']:.4f}")
    print(f"      Std:   {summary['metrics']['recall_std']:.4f}")
    
    print(f"\n   📈 F1-SCORE")
    print(f"      Mean:  {summary['metrics']['f1_mean']:.4f}")
    print(f"      Std:   {summary['metrics']['f1_std']:.4f}")
    
    print(f"\n   📈 IoU (Intersection over Union)")
    print(f"      Mean:  {summary['metrics']['iou_mean']:.4f}")
    print(f"      Std:   {summary['metrics']['iou_std']:.4f}")
    
    # Salvează rezumat JSON
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    # Salvează detalii pe fiecare imagine
    details_path = os.path.join(results_dir, "evaluation_details.json")
    with open(details_path, "w") as f:
        json.dump(detailed_results, f, indent=4)
    
    print(f"\n💾 Rezultate salvate:")
    print(f"   - {summary_path}")
    print(f"   - {details_path}")
    print(f"   - {vis_dir}/ (imagini vizualizare)")
    
    print("\n" + "=" * 80)
    print("🎉 EVALUARE COMPLETĂ!")
    print("=" * 80)
    
    print(f"\n📸 Imagini de verificare în:")
    print(f"   {os.path.abspath(vis_dir)}")
    print(f"\n   Fiecare folder conține 7 imagini:")
    print(f"   - 01_before.png")
    print(f"   - 02_after.png")
    print(f"   - 03_gt_mask.png (Ground Truth)")
    print(f"   - 04_predicted_mask.png (Predicție binară)")
    print(f"   - 05_probability_map.png (Heatmap culori)")
    print(f"   - 06_gt_overlay.png (GT pe after - verde)")
    print(f"   - 07_pred_overlay.png (Predicție pe after - roșu)")

if __name__ == "__main__":
    evaluate()