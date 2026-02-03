#!/usr/bin/env python3
"""
Script AVANSAT pentru detectarea imaginilor corupte/distorsionate.
Combină:
1. SSIM (Structural Similarity) - comparație before vs after
2. Histogram Correlation - analiza distribuției pixelilor
3. Edge Detection - detectare distorsiuni geometrice
4. Z-Score - outlier detection
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import sys

# Configurare directorio
BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"
DATASETS = {
    'train': ['before', 'after', 'masks', 'masks_clean'],
    'test': ['before', 'after', 'masks'],
    'validation': ['before', 'after', 'masks']
}

class CorruptedImageDetector:
    def __init__(self, ssim_threshold=0.4, hist_threshold=0.5):
        self.ssim_threshold = ssim_threshold
        self.hist_threshold = hist_threshold
        self.corrupted_images = []
        self.scores = defaultdict(list)
    
    def calculate_ssim(self, img1_path, img2_path):
        """Calculează SSIM score între două imagini."""
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return None
            
            # Resize img2 to match img1 size if needed
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            score = ssim(img1, img2, data_range=255)
            return score
        except Exception as e:
            return None
    
    def calculate_histogram_correlation(self, img1_path, img2_path):
        """Calculează corelația histogramei între două imagini."""
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return None
            
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            
            # Normalize
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            # Corelație Bhattacharyya
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return correlation
        except Exception as e:
            return None
    
    def detect_geometric_distortion(self, img_path):
        """
        Detectează distorsiuni geometrice prin analiză edge detection.
        Imaginile distorsionate au adesea o densitate de margini ciudată.
        """
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Canny edge detection
            edges = cv2.Canny(img, 50, 150)
            
            # Calculează densitatea de margini
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            
            # Detectează linii lungi repetitive (indicație de distorsiune)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            line_count = len(lines) if lines is not None else 0
            
            # Calculeaza ratio: mai multe linii repetitive = mai probabil distorcionat
            line_density = line_count / (edges.shape[0] / 100)
            
            return {
                'edge_density': edge_density,
                'line_count': line_count,
                'line_density': line_density
            }
        except Exception as e:
            return None
    
    def analyze_dataset(self):
        """Analizează toate paired imaginile din dataset."""
        print("\n" + "="*90)
        print("🔍 DETECȚIE AVANSATĂ IMAGINI CORUPTE - SSIM + HISTOGRAM + GEOMETRIC DISTORTION")
        print("="*90)
        
        for dataset_name, folders in DATASETS.items():
            dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
            before_path = os.path.join(dataset_path, 'before')
            after_path = os.path.join(dataset_path, 'after')
            
            if not os.path.exists(before_path) or not os.path.exists(after_path):
                print(f"⚠️  Dataset folder nu există: {dataset_name}")
                continue
            
            print(f"\n📁 Analizare: {dataset_name}")
            
            # Get paired images
            before_files = set(os.listdir(before_path))
            after_files = set(os.listdir(after_path))
            paired_files = sorted(before_files & after_files)
            
            print(f"   Total perechi: {len(paired_files)}")
            
            for idx, filename in enumerate(paired_files):
                if (idx + 1) % 200 == 0:
                    print(f"   Progres: {idx + 1}/{len(paired_files)}")
                
                before_full = os.path.join(before_path, filename)
                after_full = os.path.join(after_path, filename)
                
                # 1. SSIM Score
                ssim_score = self.calculate_ssim(before_full, after_full)
                if ssim_score is None:
                    continue
                
                # 2. Histogram Correlation
                hist_corr = self.calculate_histogram_correlation(before_full, after_full)
                if hist_corr is None:
                    hist_corr = 0
                
                # 3. Geometric Distortion
                geom_stats = self.detect_geometric_distortion(after_full)
                if geom_stats is None:
                    continue
                
                edge_density = geom_stats['edge_density']
                line_density = geom_stats['line_density']
                
                # Combinate score (lower = more corrupted)
                combined_score = (ssim_score + hist_corr) / 2
                
                # Store scores
                self.scores[dataset_name].append({
                    'filename': filename,
                    'dataset': dataset_name,
                    'before_path': before_full,
                    'after_path': after_full,
                    'ssim': ssim_score,
                    'hist_corr': hist_corr,
                    'edge_density': edge_density,
                    'line_density': line_density,
                    'combined_score': combined_score,
                    'is_outlier': False  # Will be set after z-score
                })
    
    def calculate_z_scores(self):
        """Calculează z-scores și marchează outliers."""
        print(f"\n📊 Calculare Z-Scores pentru outlier detection...")
        
        all_scores = []
        for dataset_scores in self.scores.values():
            all_scores.extend([s['combined_score'] for s in dataset_scores])
        
        if not all_scores:
            print("❌ Nu sunt date suficiente!")
            return
        
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        print(f"\n   Mean Combined Score: {mean_score:.4f}")
        print(f"   Std Dev: {std_score:.4f}")
        print(f"   Outlier Threshold (Z > 2): {mean_score - 2*std_score:.4f}")
        
        # Marchează outliers
        for dataset_scores in self.scores.values():
            for score_dict in dataset_scores:
                z_score = (score_dict['combined_score'] - mean_score) / (std_score + 1e-8)
                score_dict['z_score'] = z_score
                
                # Outlier dacă:
                # 1. SSIM < threshold
                # 2. Histogram correlation prea scăzută
                # 3. Z-score > 2 (mai mult de 2 deviații standard)
                if (score_dict['ssim'] < self.ssim_threshold or 
                    score_dict['hist_corr'] < self.hist_threshold or
                    z_score > 2):
                    score_dict['is_outlier'] = True
                    self.corrupted_images.append(score_dict)
        
        print(f"\n✅ Outliers detectați: {len(self.corrupted_images)}")
    
    def display_results(self, top_n=20):
        """Afișează rezultatele - primele N imagini corupte."""
        print("\n" + "="*90)
        print(f"🔴 TOP {min(top_n, len(self.corrupted_images))} IMAGINI CORUPTE DETECTATE")
        print("="*90)
        
        if not self.corrupted_images:
            print("✅ Nu au fost găsite imagini corupte!")
            return
        
        # Sort by combined score (mai scăzut = mai corupt)
        sorted_corrupted = sorted(self.corrupted_images, key=lambda x: x['combined_score'])
        
        for idx, img in enumerate(sorted_corrupted[:top_n], 1):
            dataset_path = os.path.join(BASE_DATA_DIR, img['dataset'])
            base_folder = dataset_path
            filename = img['filename']
            
            print(f"\n{idx}. IMAGINI DE ȘTERS (TRIPLET):")
            print(f"   After:  {img['after_path']}")
            print(f"   Before: {img['before_path']}")
            
            # Check for masks
            mask_path = os.path.join(base_folder, 'masks', filename)
            mask_clean_path = os.path.join(base_folder, 'masks_clean', filename)
            
            if os.path.exists(mask_path):
                print(f"   Mask:   {mask_path}")
            if os.path.exists(mask_clean_path):
                print(f"   Mask_Clean: {mask_clean_path}")
            
            # Scores
            print(f"\n   📊 METRICI:")
            print(f"      SSIM Score: {img['ssim']:.4f} (baseline: 0.4 = LOW)")
            print(f"      Hist Correlation: {img['hist_corr']:.4f}")
            print(f"      Edge Density: {img['edge_density']:.6f}")
            print(f"      Line Density: {img['line_density']:.2f}")
            print(f"      Combined Score: {img['combined_score']:.4f}")
            print(f"      Z-Score: {img['z_score']:.2f}")
            
            # Motiv de ștergere
            motives = []
            if img['ssim'] < self.ssim_threshold:
                motives.append(f"SSIM scăzut ({img['ssim']:.4f} < {self.ssim_threshold})")
            if img['hist_corr'] < self.hist_threshold:
                motives.append(f"Histogram scăzut ({img['hist_corr']:.4f} < {self.hist_threshold})")
            if img['z_score'] > 2:
                motives.append(f"Outlier Z-Score ({img['z_score']:.2f} > 2)")
            
            print(f"   ⛔ MOTIVE DE ȘTERGERE: {', '.join(motives)}")
    
    def save_report(self, top_n=20):
        """Salvează raportul în fișier."""
        report_path = os.path.join(BASE_DATA_DIR, "../corrupted_images_advanced_report.txt")
        csv_path = os.path.join(BASE_DATA_DIR, "../corrupted_images_full_list.csv")
        
        sorted_corrupted = sorted(self.corrupted_images, key=lambda x: x['combined_score'])
        
        # Salvează raportul text
        with open(report_path, 'w') as f:
            f.write(f"RAPORT IMAGINI CORUPTE (METODA AVANSATĂ) - {len(self.corrupted_images)} DETECTATE\n")
            f.write("="*90 + "\n\n")
            f.write("IMAGINI PENTRU ȘTERGERE SIMETRICĂ:\n")
            f.write("-"*90 + "\n\n")
            
            for img in sorted_corrupted[:top_n]:
                f.write(f"BEFORE: {img['before_path']}\n")
                f.write(f"AFTER:  {img['after_path']}\n")
                
                dataset_path = os.path.join(BASE_DATA_DIR, img['dataset'])
                mask_path = os.path.join(dataset_path, 'masks', img['filename'])
                mask_clean_path = os.path.join(dataset_path, 'masks_clean', img['filename'])
                
                if os.path.exists(mask_path):
                    f.write(f"MASK:   {mask_path}\n")
                if os.path.exists(mask_clean_path):
                    f.write(f"MASK_CLEAN: {mask_clean_path}\n")
                
                f.write(f"SSIM: {img['ssim']:.4f} | HistCorr: {img['hist_corr']:.4f} | ")
                f.write(f"Combined: {img['combined_score']:.4f} | Z-Score: {img['z_score']:.2f}\n")
                f.write("-"*90 + "\n\n")
        
        # Salvează CSV complet
        with open(csv_path, 'w') as f:
            f.write("rank,filename,dataset,ssim,hist_corr,edge_density,line_density,combined_score,z_score,after_path,before_path\n")
            for idx, img in enumerate(sorted_corrupted, 1):
                f.write(f'{idx},"{img["filename"]}",{img["dataset"]},{img["ssim"]:.6f},{img["hist_corr"]:.6f},'
                        f'{img["edge_density"]:.6f},{img["line_density"]:.2f},{img["combined_score"]:.6f},'
                        f'{img["z_score"]:.2f},"{img["after_path"]}","{img["before_path"]}"\n')
        
        print(f"\n✅ Raport text salvat în: {report_path}")
        print(f"✅ CSV complet salvat în: {csv_path}")
    
    def get_statistics(self):
        """Afișează statistici generale."""
        print("\n" + "="*90)
        print("📈 STATISTICI GENERALE")
        print("="*90)
        
        for dataset_name, dataset_scores in self.scores.items():
            if not dataset_scores:
                continue
            
            ssim_scores = [s['ssim'] for s in dataset_scores]
            hist_scores = [s['hist_corr'] for s in dataset_scores]
            combined_scores = [s['combined_score'] for s in dataset_scores]
            corrupted_count = sum(1 for s in dataset_scores if s['is_outlier'])
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Total imagini: {len(dataset_scores)}")
            print(f"  Imagini corupte: {corrupted_count} ({100*corrupted_count/len(dataset_scores):.1f}%)")
            print(f"\n  SSIM Score:")
            print(f"    Min: {min(ssim_scores):.4f}, Max: {max(ssim_scores):.4f}, Mean: {np.mean(ssim_scores):.4f}")
            print(f"\n  Histogram Correlation:")
            print(f"    Min: {min(hist_scores):.4f}, Max: {max(hist_scores):.4f}, Mean: {np.mean(hist_scores):.4f}")
            print(f"\n  Combined Score:")
            print(f"    Min: {min(combined_scores):.4f}, Max: {max(combined_scores):.4f}, Mean: {np.mean(combined_scores):.4f}")


if __name__ == "__main__":
    detector = CorruptedImageDetector(
        ssim_threshold=0.4,      # Sub 0.4 = suspect
        hist_threshold=0.5       # Sub 0.5 = suspect
    )
    
    print("\n🚀 Inițiare detecție AVANSATĂ...\n")
    
    # Analiza
    detector.analyze_dataset()
    
    # Calculează z-scores
    detector.calculate_z_scores()
    
    # Statistici
    detector.get_statistics()
    
    # Previzualizare
    detector.display_results(top_n=20)
    
    # Salvare raport
    detector.save_report(top_n=20)
    
    print("\n⏸️  AȘTEPT CONFIRMAREA DVSTRĂ!")
    print(f"   Total imagini marcate pentru ștergere: {len(detector.corrupted_images)}")
    print("   Verificați previzualizarea de mai sus.\n")
