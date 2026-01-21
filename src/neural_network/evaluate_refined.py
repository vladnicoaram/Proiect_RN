import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import json
from scipy import ndimage
from pathlib import Path

# Importurile tale specifice
from dataset import ChangeDetectionDataset 
from model import UNet

# ============================================================================
# METRICE CU FILTRU INTELIGENT
# ============================================================================

def apply_morphological_filter(mask, min_pixels=200):
    """
    Aplică filtru morfologic: elimină componentele mici (<200 pixeli)
    Păstrează componentele mari (care conțin obiecte reale)
    """
    # Label connected components
    labeled_array, num_features = ndimage.label(mask)
    
    # Pentru fiecare component, calculează dimensiunea
    filtered_mask = np.zeros_like(mask)
    
    for i in range(1, num_features + 1):
        component = (labeled_array == i).astype(np.uint8)
        component_size = np.sum(component)
        
        # Păstrează doar componente cu >min_pixels pixeli
        if component_size > min_pixels:
            filtered_mask += component
    
    return (filtered_mask > 0).astype(np.uint8)

def calculate_metrics_refined(pred_prob, target, threshold=0.55, min_pixels=200):
    """
    Calculează metrici cu threshold echilibrat (0.55) și filtru de dimensiune
    
    Args:
        pred_prob: Probabilitățile brute din model (0-1)
        target: Ground truth mask
        threshold: Pragul pentru binarizare (default 0.55 - echilibrat)
        min_pixels: Dimensiune minimă componentă (default 200)
    
    Returns:
        dict cu metrici rafinate
    """
    # Binarizare cu threshold înalt
    pred_bin = (pred_prob > threshold).float().cpu().numpy()
    
    # Aplică filtru morfologic (elimină obiecte mici din noise)
    pred_filtered = apply_morphological_filter(pred_bin, min_pixels=min_pixels)
    
    # Target
    target_np = target.cpu().numpy()
    
    # Metrics
    tp = np.sum(pred_filtered * target_np)
    fp = np.sum(pred_filtered * (1 - target_np))
    fn = np.sum((1 - pred_filtered) * target_np)
    tn = np.sum((1 - pred_filtered) * (1 - target_np))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # IoU
    intersection = tp
    union = np.sum((pred_filtered + target_np) > 0)
    iou = intersection / (union + 1e-6)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'pred_filtered': pred_filtered,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }

def save_comparison_images_refined(x, y, pred_mask_filtered, pred_prob, output_dir, filename_base, metrics):
    """Salvează imagini de comparație cu filtrul morfologic aplicat"""
    
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
    if isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.cpu().numpy()
    
    # Before image
    before_img = (x_np[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
    before_path = os.path.join(output_dir, f"{filename_base}_01_before.png")
    cv2.imwrite(before_path, cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR))
    
    # After image
    after_img = (x_np[3:].transpose(1, 2, 0) * 255).astype(np.uint8)
    after_path = os.path.join(output_dir, f"{filename_base}_02_after.png")
    cv2.imwrite(after_path, cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))
    
    # GT Mask
    gt_mask_img = (y_np.squeeze() * 255).astype(np.uint8)
    gt_path = os.path.join(output_dir, f"{filename_base}_03_gt_mask.png")
    cv2.imwrite(gt_path, gt_mask_img)
    
    # Predicted mask (FILTRAT - doar obiecte >200px)
    pred_mask_img = (pred_mask_filtered * 255).astype(np.uint8)
    # Asigură-te că e 2D pentru cv2.imwrite
    if len(pred_mask_img.shape) == 3:
        pred_mask_img = pred_mask_img.squeeze()
    pred_path = os.path.join(output_dir, f"{filename_base}_04_predicted_filtered.png")
    cv2.imwrite(pred_path, pred_mask_img)
    
    # Probability map (heatmap)
    pred_prob_img = (pred_prob.squeeze() * 255).astype(np.uint8)
    pred_prob_color = cv2.applyColorMap(pred_prob_img, cv2.COLORMAP_JET)
    prob_path = os.path.join(output_dir, f"{filename_base}_05_probability_heatmap.png")
    cv2.imwrite(prob_path, pred_prob_color)
    
    # Overlay GT
    overlay_gt = after_img.copy()
    overlay_gt[gt_mask_img > 128] = [0, 255, 0]
    overlay_gt_path = os.path.join(output_dir, f"{filename_base}_06_gt_overlay.png")
    cv2.imwrite(overlay_gt_path, cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR))
    
    # Overlay Predicted (FILTRAT)
    overlay_pred = after_img.copy()
    pred_mask_overlay = pred_mask_filtered.squeeze() if len(pred_mask_filtered.shape) == 3 else pred_mask_filtered
    overlay_pred[pred_mask_overlay > 0.5] = [0, 0, 255]
    overlay_pred_path = os.path.join(output_dir, f"{filename_base}_07_pred_filtered_overlay.png")
    cv2.imwrite(overlay_pred_path, cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))
    
    # Creează info card cu metrici
    info_text = f"""REFINED EVALUATION
Threshold: 0.75 | Min Component: 200px
{'='*50}
TP: {metrics['tp']} | FP: {metrics['fp']}
FN: {metrics['fn']} | TN: {metrics['tn']}
{'='*50}
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1']:.4f}
IoU:       {metrics['iou']:.4f}
"""
    info_path = os.path.join(output_dir, f"{filename_base}_info.txt")
    with open(info_path, "w") as f:
        f.write(info_text)

# ============================================================================
# EVALUARE RAFINATĂ
# ============================================================================

def evaluate_refined():
    print("=" * 80)
    print("🔍 EVALUARE RAFINATĂ - Small Objects Detection (Threshold=0.75, MinPx=200)")
    print("=" * 80)
    
    # Setăm device-ul
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n📱 Device: {DEVICE}")
    
    # Parametri evaluare
    THRESHOLD = 0.55  # Echilibrat: destul de strict dar nu prea
    MIN_PIXELS = 200  # Elimină noise-ul (obiecte <200px)
    
    print(f"\n⚙️  Parametri Evaluare:")
    print(f"   Threshold: {THRESHOLD} (ridicat pentru precisie)")
    print(f"   Min Component Size: {MIN_PIXELS} pixeli")
    print(f"   Strategie: Păstrează orice >200px, reject noise")
    
    # Calea către modelul rafinat
    MODEL_PATHS = [
        "../../models/unet_refined_small_objects.pth",
        "../../models/unet_final_clean.pth",
        "../../models/unet_final.pth"
    ]
    
    MODEL_PATH = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            MODEL_PATH = path
            break
    
    if MODEL_PATH is None:
        print(f"❌ Eroare: Nu am găsit modelul în niciunul din căile:")
        for path in MODEL_PATHS:
            print(f"   - {path}")
        return

    # Încărcare Model
    print(f"\n📦 Se încarcă modelul din {MODEL_PATH}...")
    model = UNet(6, 1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("   ✓ Model încărcat cu succes")

    # Directoare de salvare
    results_dir = "../../results/evaluation_refined"
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
        'tp': [],
        'fp': [],
        'fn': [],
    }
    
    detailed_results = []

    print(f"\n🚀 Evaluare pornită...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Inferență
            logits = model(x)
            output = torch.sigmoid(logits)
            
            # Calcul metrici rafinate (cu filtru)
            metrics = calculate_metrics_refined(
                output[0], y[0], 
                threshold=THRESHOLD, 
                min_pixels=MIN_PIXELS
            )
            
            for key in ['accuracy', 'precision', 'recall', 'f1', 'iou', 'tp', 'fp', 'fn']:
                all_metrics[key].append(metrics[key])
            
            # Salvează imagini
            filename_base = f"{i:04d}"
            save_comparison_images_refined(
                x[0], y[0], 
                metrics['pred_filtered'], 
                output[0],
                vis_dir, 
                filename_base, 
                metrics
            )
            
            detailed_results.append({
                'sample_id': i,
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'iou': float(metrics['iou']),
                'tp': int(metrics['tp']),
                'fp': int(metrics['fp']),
                'fn': int(metrics['fn']),
            })
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ Procesat {i + 1}/{len(test_loader)} imagini...")

    # ========================================================================
    # REZULTATE FINALE
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("📊 REZULTATE FINALE - EVALUARE RAFINATĂ")
    print("=" * 80)
    
    summary = {
        'total_samples': len(test_loader),
        'threshold': THRESHOLD,
        'min_pixels': MIN_PIXELS,
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
    
    print(f"\n✅ METRICI RAFINATE (pe {len(test_loader)} imagini):")
    print(f"\n   📈 ACCURACY")
    print(f"      Mean:  {summary['metrics']['accuracy_mean']:.4f}")
    print(f"      Std:   {summary['metrics']['accuracy_std']:.4f}")
    
    print(f"\n   📈 PRECISION (CRITICAL - vrem >0.60)")
    print(f"      Mean:  {summary['metrics']['precision_mean']:.4f} {'✅' if summary['metrics']['precision_mean'] > 0.60 else '❌'}")
    print(f"      Std:   {summary['metrics']['precision_std']:.4f}")
    
    print(f"\n   📈 RECALL (vrem >0.90)")
    print(f"      Mean:  {summary['metrics']['recall_mean']:.4f} {'✅' if summary['metrics']['recall_mean'] > 0.90 else '⚠️'}")
    print(f"      Std:   {summary['metrics']['recall_std']:.4f}")
    
    print(f"\n   📈 F1-SCORE")
    print(f"      Mean:  {summary['metrics']['f1_mean']:.4f}")
    print(f"      Std:   {summary['metrics']['f1_std']:.4f}")
    
    print(f"\n   📈 IoU")
    print(f"      Mean:  {summary['metrics']['iou_mean']:.4f}")
    print(f"      Std:   {summary['metrics']['iou_std']:.4f}")
    
    # Salvează rezultate
    summary_path = os.path.join(results_dir, "evaluation_refined_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    details_path = os.path.join(results_dir, "evaluation_refined_details.json")
    with open(details_path, "w") as f:
        json.dump(detailed_results, f, indent=4)
    
    print(f"\n💾 Rezultate salvate:")
    print(f"   - {summary_path}")
    print(f"   - {details_path}")
    print(f"   - {vis_dir}/ (imagini)")
    
    print("\n" + "=" * 80)
    print("🎉 EVALUARE RAFINATĂ COMPLETĂ!")
    print("=" * 80)
    
    print(f"\n📸 Imagini în: {os.path.abspath(vis_dir)}")
    print(f"\n   Fiecare folder:")
    print(f"   - 01_before.png")
    print(f"   - 02_after.png")
    print(f"   - 03_gt_mask.png")
    print(f"   - 04_predicted_filtered.png ← FILTRAT (>200px)")
    print(f"   - 05_probability_heatmap.png")
    print(f"   - 06_gt_overlay.png")
    print(f"   - 07_pred_filtered_overlay.png ← FILTRAT")
    print(f"   - info.txt ← Metrici per imagine")

if __name__ == "__main__":
    evaluate_refined()
