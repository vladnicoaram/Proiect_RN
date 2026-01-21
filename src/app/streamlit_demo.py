#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicație UI Streamlit pentru Etapa 6
Demonstrație inference cu modelul optimizat
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
from scipy import ndimage
from pathlib import Path

# Import model
from src.neural_network.model import UNet
from src.neural_network.dataset import ChangeDetectionDataset

# ============================================================================
# CONFIGURARE STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Change Detection AI - Etapa 6",
    page_icon="🔍",
    layout="wide"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Încarcă modelul optimizat"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = UNet(in_channels=6, out_channels=1)
    checkpoint = torch.load('models/optimized_model.pt', map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device

def apply_morphological_filter(mask, min_pixels=200):
    """Filtru morfologic: elimină componentele mici"""
    labeled_array, num_features = ndimage.label(mask)
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component = (labeled_array == i).astype(np.uint8)
        component_size = np.sum(component)
        if component_size > min_pixels:
            filtered_mask += component
    return (filtered_mask > 0).astype(np.uint8)

def run_inference(model, device, x, threshold=0.55, min_pixels=200):
    """Rulează inferență pe imagine"""
    x = x.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)
    
    pred_prob_np = prob[0, 0].cpu().numpy()
    pred_binary = (pred_prob_np > threshold).astype(np.uint8)
    pred_filtered = apply_morphological_filter(pred_binary, min_pixels=min_pixels)
    
    return pred_prob_np, pred_filtered

# ============================================================================
# UI PRINCIPALE
# ============================================================================

st.title("🔍 Change Detection AI - Model Optimizat (Etapa 6)")

st.markdown("""
**Status Model:** ✅ Optimizat și Finalizat
- **Accuracy:** 85.77%
- **Precision:** 76.48%
- **Recall:** 62.72%
- **IoU:** 49.46%
- **F1-Score:** 66.71%
""")

# Încarcă model
with st.spinner('Încarcă modelul optimizat...'):
    model, device = load_model()

st.success(f'✅ Model încărcat pe device: {device}')

# Opțiuni pentru testare
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Parametri Inferență")
    threshold = st.slider("Threshold", 0.3, 0.9, 0.55, 0.01)
    min_pixels = st.slider("Min Component Size (pixeli)", 50, 500, 200, 50)

with col2:
    st.subheader("📁 Selectare Test Sample")
    # Încarcă test dataset
    test_dataset = ChangeDetectionDataset(root_dir="data/test", augment=False)
    sample_idx = st.selectbox("Sample ID", range(len(test_dataset)), index=91)

# ============================================================================
# INFERENȚĂ
# ============================================================================

if st.button("🚀 Rulează Inferență", type="primary"):
    with st.spinner(f'Procesez sample #{sample_idx}...'):
        x, y = test_dataset[sample_idx]
        y_np = y.numpy().astype(np.uint8)
        
        # Inferență
        pred_prob_np, pred_filtered = run_inference(model, device, x, threshold, min_pixels)
        
        # Extrage imagini originale
        x_np = x.numpy()
        before_img = x_np[:3].transpose(1, 2, 0)
        before_img = ((before_img + 1) / 2 * 255).astype(np.uint8)
        
        after_img = x_np[3:].transpose(1, 2, 0)
        after_img = ((after_img + 1) / 2 * 255).astype(np.uint8)
        
        # Crează overlay
        overlay = after_img.copy().astype(np.float32)
        green_mask = np.zeros_like(overlay)
        green_mask[:, :, 1] = pred_filtered * 255
        
        alpha = 0.3
        overlay = (overlay * (1 - alpha) + green_mask * alpha).astype(np.uint8)
        
        # Contururi în roșu
        contours, _ = cv2.findContours(pred_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        
        # Calcul metrici
        tp = np.sum(pred_filtered & y_np)
        fp = np.sum(pred_filtered & ~y_np)
        fn = np.sum(~pred_filtered & y_np)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # Afișare rezultate
        st.subheader(f"📸 Rezultate Sample #{sample_idx}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR), caption="BEFORE", use_column_width=True)
        
        with col2:
            st.image(cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR), caption="AFTER", use_column_width=True)
        
        with col3:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), caption="PREDICȚIE (Verde=Change, Roșu=Border)", use_column_width=True)
        
        # Metrici
        st.subheader("📊 Metrici Predicție")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Precision", f"{precision:.2%}")
        with metric_col2:
            st.metric("Recall", f"{recall:.2%}")
        with metric_col3:
            st.metric("IoU", f"{iou:.2%}")
        with metric_col4:
            st.metric("TP Pixels", f"{tp:,}")
        
        # Salvează screenshot
        save_path = "docs/screenshots/inference_optimized.png"
        os.makedirs("docs/screenshots", exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        st.success(f"✅ Screenshot salvat: {save_path}")
        
        # Detalii suplimentare
        with st.expander("📋 Detalii Complete"):
            st.write(f"""
            **Configurație Inference:**
            - Model: optimized_model.pt
            - Threshold: {threshold}
            - Min Component Size: {min_pixels} pixeli
            - Device: {device}
            
            **Statistici Predicție:**
            - True Positives: {tp:,} pixeli
            - False Positives: {fp:,} pixeli
            - False Negatives: {fn:,} pixeli
            - Confidence Score: {pred_prob_np.mean():.2%}
            """)

st.markdown("---")
st.caption("🎓 Etapa 6 - Model Optimizat cu Focal Loss + Morphological Filtering")
