#!/usr/bin/env python3
"""
TRAINING MONITOR - Urmareste progresul antrenarii
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/Users/admin/Documents/Facultatea/Proiect_RN/results")
RESULTS_FILE = RESULTS_DIR / "training_results_refined.json"
CHECKPOINTS_DIR = Path("/Users/admin/Documents/Facultatea/Proiect_RN/checkpoints")

print("\n" + "="*80)
print("TRAINING MONITOR")
print("="*80)

# Check if training is ongoing
if RESULTS_FILE.exists():
    try:
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        
        print(f"\n✓ Training completed!")
        print(f"\nResults:")
        print(f"  Best validation loss: {results['best_val_loss']:.4f}")
        print(f"  Test loss: {results['test_loss']:.4f}")
        print(f"  Total epochs: {results['final_epoch']}")
        print(f"  Timestamp: {results['timestamp']}")
        
        # Show history
        history = results['history']
        last_epoch = len(history['train_loss']) - 1
        
        print(f"\nLast epoch ({last_epoch+1}):")
        print(f"  Train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Val loss: {history['val_loss'][-1]:.4f}")
        print(f"  LR: {history['learning_rate'][-1]:.6f}")
        
        # Plot file
        plot_file = RESULTS_DIR / "training_curves_refined.png"
        if plot_file.exists():
            print(f"\n✓ Plot saved: {plot_file}")
        
        # Best model
        best_model = CHECKPOINTS_DIR / "best_model_refined.pth"
        if best_model.exists():
            size = os.path.getsize(best_model) / 1024 / 1024
            print(f"✓ Best model saved: {best_model} ({size:.1f} MB)")
    
    except Exception as e:
        print(f"Error reading results: {e}")
else:
    print("\n⏳ Training in progress...")
    print(f"\nCheck back in a few minutes for results.")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print(f"Best model will be saved to: {CHECKPOINTS_DIR}/best_model_refined.pth")

print("\n" + "="*80 + "\n")
