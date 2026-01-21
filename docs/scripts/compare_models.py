import json
import numpy as np

with open("results/evaluation/evaluation_details.json", "r") as f:
    old_results = json.load(f)

with open("results/evaluation_refined/evaluation_refined_details.json", "r") as f:
    new_results = json.load(f)

print("=" * 80)
print("COMPARATIE MODELE - OLD vs NEW")
print("=" * 80)

old_acc = np.array([r['accuracy'] for r in old_results])
old_prec = np.array([r['precision'] if r['precision'] > 0 else 0 for r in old_results])
old_recall = np.array([r['recall'] if r['recall'] > 0 else 0 for r in old_results])

new_acc = np.array([r['accuracy'] for r in new_results])
new_prec = np.array([r['precision'] if r['precision'] > 0 else 0 for r in new_results])
new_recall = np.array([r['recall'] if r['recall'] > 0 else 0 for r in new_results])

print("\nMODEL VECHI (BCE Loss):")
print("   Accuracy: ", round(np.mean(old_acc), 4))
print("   Precision:", round(np.mean(old_prec), 4))
print("   Recall:   ", round(np.mean(old_recall), 4))

print("\nMODEL NOU (Focal Loss - Rafinat):")
print("   Accuracy: ", round(np.mean(new_acc), 4))
print("   Precision:", round(np.mean(new_prec), 4))
print("   Recall:   ", round(np.mean(new_recall), 4))

print("\n" + "=" * 80)
print("TOP 10 IMAGINI CU CELE MAI MARI DETECTII")
print("=" * 80)

detections = []
for i in range(len(new_results)):
    old_iou = old_results[i]['iou']
    new_iou = new_results[i]['iou']
    new_tp = new_results[i]['tp']
    old_tp_approx = old_results[i]['accuracy'] * 262144  # estimare din accuracy
    improvement = new_iou - old_iou
    
    # Căutam obiecte MICI (<15000 px) dar detectate bine de modelul nou
    if 0 < new_tp < 15000 and new_iou > 0.3 and improvement > 0:
        detections.append((i, old_iou, new_iou, improvement, new_tp))

detections.sort(key=lambda x: x[2], reverse=True)

for rank, (idx, old_iou, new_iou, improvement, new_tp) in enumerate(detections[:10], 1):
    print("   {:2d}. Sample {:04d}: IoU Old={:.3f}, New={:.3f} ({:+.3f}), TP={:4d}px".format(rank, idx, old_iou, new_iou, improvement, new_tp))

print("\n" + "=" * 80)
print("TOP 3 EXEMPLE - OBIECTE MICI DETECTATE")
print("=" * 80)
print("\nFoldere pentru inspectie vizuala:")
for rank, (idx, old_iou, new_iou, improvement, new_tp) in enumerate(detections[:3], 1):
    old_path = "results/evaluation/visualizations/{:04d}*".format(idx)
    new_path = "results/evaluation_refined/visualizations/{:04d}*".format(idx)
    print("\nExemplu {}: Sample {:04d}".format(rank, idx))
    print("  - Model VECHI: results/evaluation/visualizations/{:04d}_*".format(idx))
    print("  - Model NOU:   results/evaluation_refined/visualizations/{:04d}_*".format(idx))
    print("  - Improvement IoU: {:.3f} → {:.3f} ({:+.3f})".format(old_iou, new_iou, improvement))
    print("  - True Positives detectate (new): {} pixeli".format(new_tp))
