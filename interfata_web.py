import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
import sys

sys.path.append(os.getcwd())
from src.neural_network.model import UNet

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
IMG_SIZE = (256, 256)

st.set_page_config(page_title="AI Change Detector Pro", layout="wide")

def match_histograms(source, reference):
    matched = np.zeros_like(source)
    for i in range(3):
        source_hist, _ = np.histogram(source[:,:,i].ravel(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference[:,:,i].ravel(), 256, [0, 256])
        src_cdf = source_hist.cumsum()
        src_cdf = 255 * src_cdf / src_cdf[-1]
        ref_cdf = ref_hist.cumsum()
        ref_cdf = 255 * ref_cdf / ref_cdf[-1]
        lookup_table = np.zeros(256)
        for j in range(256):
            diff = np.abs(ref_cdf - src_cdf[j])
            lookup_table[j] = diff.argmin()
        matched[:,:,i] = cv2.LUT(source[:,:,i].astype(np.uint8), lookup_table.astype(np.uint8))
    return matched

@st.cache_resource
def load_model():
    # Prioritate: optimized_model.pt (Etapa 6) -> fallback: unet_final.pth (Etapa 5)
    model_path = "models/optimized_model.pt"
    if not os.path.exists(model_path):
        model_path = "models/unet_final.pth"
    
    if not os.path.exists(model_path): 
        return None
    
    model = UNet(in_channels=6, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def main():
    st.title("🛡️ Detector AI (Corecție Margini și Podea)")
    
    # Afișare info model și metrici (din Etapa 6)
    with st.sidebar:
        st.markdown("### 📊 Model Info")
        model_info = {
            "Model": "optimized_model.pt (Etapa 6)",
            "Accuracy": "85.77%",
            "Precision": "76.48%",
            "F1-Score": "0.667",
            "Device": "M1 MPS" if torch.backends.mps.is_available() else "CPU"
        }
        
        for key, value in model_info.items():
            st.text(f"{key}: {value}")
        
        # Verifică dacă fișierul metrics JSON există
        metrics_path = "results/final_metrics.json"
        if os.path.exists(metrics_path):
            st.markdown("✅ Full metrics available at: `results/final_metrics.json`")
    
    model = load_model()
    if model is None: 
        st.error("Modelul nu a fost găsit! Verific: models/optimized_model.pt și models/unet_final.pth")
        return

    f1 = st.sidebar.file_uploader("Before", type=["jpg", "png", "jpeg"])
    f2 = st.sidebar.file_uploader("After", type=["jpg", "png", "jpeg"])

    if f1 and f2:
        img_b = Image.open(f1).convert("RGB")
        img_a = Image.open(f2).convert("RGB")
        
        with st.spinner("Analiză în curs..."):
            b_np = np.array(img_b)
            a_np = np.array(img_a)
            a_matched = match_histograms(a_np, b_np)
            
            b_res = cv2.resize(b_np, IMG_SIZE)
            a_res = cv2.resize(a_matched, IMG_SIZE)
            
            x = np.concatenate([b_res/255.0, a_res/255.0], axis=2)
            x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                mask = model(x).squeeze().cpu().numpy()

        # --- REGLAJE PENTRU DETECȚIE COMPLETĂ ---
        mask_u8 = (mask * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(mask_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Facem pragul puțin mai sensibil (coborâm la minim 60 în loc de 80)
        final_thresh = max(otsu_val, 60) 
        _, mask_bin = cv2.threshold(mask_u8, final_thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5,5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        h, w = a_np.shape[:2]
        mask_full = cv2.resize(mask_clean, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        res_img = a_np.copy()
        count = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
            
            # --- MODIFICĂRI CRITICE ---
            # 1. Creștem Aspect Ratio la 8.0 (permite pături/covoare lungi)
            aspect_ratio = max(w_box, h_box) / min(w_box, h_box)
            
            # 2. Relaxăm aria minimă (0.03% din imagine)
            if area > (w * h * 0.0003) and aspect_ratio < 8.0:
                # 3. ELIMINĂM FILTRUL DE MARGINE (acum acceptăm și detecții la poli)
                cv2.rectangle(res_img, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 0, 255), 4)
                count += 1

        # Afișare
        st.subheader(f"Rezultat: {count} obiecte detectate")
        col_left, col_right = st.columns(2)
        
        with col_left:
            m_rgb = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2RGB)
            m_rgb[:, :, 1:] = 0
            st.image(cv2.addWeighted(a_np, 0.7, m_rgb, 0.3, 0), caption="Heatmap", use_container_width=True)
            
        with col_right:
            st.image(res_img, caption="Rezultat Final (Pătura inclusă)", use_container_width=True)

if __name__ == "__main__":
    main()