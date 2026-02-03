#!/usr/bin/env python3
"""
Script pentru validarea integrității scenei - detectare perechi before/after
dintr-o cameră diferită sau unghi diferit.

Metoda: ORB Feature Detection + BFMatcher
- Extrage puncte de interes (keypoints) din ambele imagini
- Face matching între punctele comune
- Dacă sunt prea puține potriviri (< 20) = probabil cameră diferită
- Verifică distribuția spațială a potrivirilor
"""

import os
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
import csv

# Configurare directorio
BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"
DATASETS = {
    'train': ['before', 'after', 'masks', 'masks_clean'],
    'test': ['before', 'after', 'masks'],
    'validation': ['before', 'after', 'masks']
}

class SceneIntegrityValidator:
    def __init__(self, min_matches=20, distribution_threshold=0.4):
        self.min_matches = min_matches
        self.distribution_threshold = distribution_threshold  # % din imagine care trebuie să conțină matches
        
        # Inițializează ORB detector
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.mismatched_pairs = []
        self.scores = defaultdict(list)
    
    def extract_features(self, img_path):
        """Extrage ORB features din imagine."""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None, None
            
            kp, des = self.orb.detectAndCompute(img, None)
            return kp, des
        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
            return None, None
    
    def match_features(self, kp1, des1, kp2, des2):
        """Marchează ORB features între două seturi de descriptori."""
        if des1 is None or des2 is None:
            return None, None, None
        
        try:
            matches = self.bf_matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches, kp1, kp2
        except Exception as e:
            return None, None, None
    
    def calculate_distribution(self, matches, kp1, kp2, img_shape):
        """Calculează distribuția spațială a potrivirilor pe imagine."""
        if not matches or len(matches) < 5:
            return 0, None
        
        # Extrage coordonatele punctelor potrivite
        coords = []
        for match in matches[:self.min_matches]:  # Ia doar top matches
            pt1 = kp1[match.queryIdx].pt
            coords.append(pt1)
        
        if not coords:
            return 0, None
        
        # Calculează bounding box al potrivirilor
        coords = np.array(coords)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        # Aria ocupată de potriviri vs aria totală
        match_area = (x_max - x_min) * (y_max - y_min)
        total_area = img_shape[1] * img_shape[0]
        coverage = match_area / total_area if total_area > 0 else 0
        
        return coverage, (x_min, y_min, x_max, y_max)
    
    def validate_pair(self, before_path, after_path, before_img=None, after_img=None):
        """Validează dacă o pereche before/after sunt din aceeași scenă."""
        try:
            # Extrage features
            kp_before, des_before = self.extract_features(before_path)
            kp_after, des_after = self.extract_features(after_path)
            
            if kp_before is None or kp_after is None:
                return None
            
            # Match features
            matches, _, _ = self.match_features(kp_before, des_before, kp_after, des_after)
            
            if matches is None or len(matches) == 0:
                return {
                    'before_path': before_path,
                    'after_path': after_path,
                    'good_matches': 0,
                    'coverage': 0,
                    'quality_score': 0,
                    'is_mismatched': True,
                    'reason': 'No matches found'
                }
            
            # Contează bune potriviri
            good_matches = len([m for m in matches if m.distance < 50])
            total_matches = len(matches)
            
            # Calculează distribuția
            if before_img is None:
                before_img = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
            
            coverage, bbox = self.calculate_distribution(matches[:good_matches], 
                                                         kp_before, kp_after,
                                                         before_img.shape)
            
            # Calculează quality score (0-1)
            # Iau în considerare: nr de matches, distribuția, și quality matches
            match_quality = min(good_matches / self.min_matches, 1.0) if good_matches > 0 else 0
            distribution_quality = min(coverage / self.distribution_threshold, 1.0)
            quality_score = (match_quality * 0.7 + distribution_quality * 0.3)
            
            # Determina dacă e mismatched
            is_mismatched = (good_matches < self.min_matches or 
                           coverage < self.distribution_threshold * 0.5)
            
            reason = []
            if good_matches < self.min_matches:
                reason.append(f"Prea puține matches ({good_matches} < {self.min_matches})")
            if coverage < self.distribution_threshold * 0.5:
                reason.append(f"Distribuție slabă ({coverage:.2%})")
            
            return {
                'before_path': before_path,
                'after_path': after_path,
                'good_matches': good_matches,
                'total_matches': total_matches,
                'coverage': coverage,
                'quality_score': quality_score,
                'is_mismatched': is_mismatched,
                'reason': ' | '.join(reason) if reason else 'OK'
            }
        
        except Exception as e:
            print(f"Error validating pair: {e}")
            return None
    
    def analyze_dataset(self):
        """Analizează toate perechile din dataset."""
        print("\n" + "="*100)
        print("🔍 VALIDARE INTEGRITATE SCENĂ - ORB FEATURE MATCHING")
        print("="*100)
        
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
                
                # Validează perechea
                result = self.validate_pair(before_full, after_full)
                
                if result is None:
                    continue
                
                self.scores[dataset_name].append(result)
                
                if result['is_mismatched']:
                    self.mismatched_pairs.append({
                        **result,
                        'dataset': dataset_name,
                        'filename': filename
                    })
    
    def display_results(self, top_n=10):
        """Afișează rezultatele - primele N perechi mismatched."""
        print("\n" + "="*100)
        print(f"🔴 TOP {min(top_n, len(self.mismatched_pairs))} PERECHI CU CAMERĂ/UNGHI DIFERIT")
        print("="*100)
        
        if not self.mismatched_pairs:
            print("✅ Toate perechile sunt consistent (din aceeași scenă)!")
            return
        
        # Sort by quality score (worst first)
        sorted_mismatched = sorted(self.mismatched_pairs, key=lambda x: x['quality_score'])
        
        for idx, pair in enumerate(sorted_mismatched[:top_n], 1):
            filename = pair['filename']
            dataset = pair['dataset']
            good_matches = pair['good_matches']
            coverage = pair['coverage']
            quality = pair['quality_score']
            reason = pair['reason']
            
            print(f"\n{idx}. {filename}")
            print(f"   Dataset: {dataset} | Good Matches: {good_matches} | Coverage: {coverage:.2%} | Quality: {quality:.2%}")
            print(f"   ⛔ Motiv: {reason}")
            
            print(f"\n   📷 BEFORE (Referință):")
            print(f"   file://{pair['before_path']}")
            
            print(f"\n   📷 AFTER (Suspect - cameră/unghi diferit):")
            print(f"   file://{pair['after_path']}")
            print("-" * 100)
    
    def save_report(self):
        """Salvează raportul complet."""
        report_path = os.path.join(BASE_DATA_DIR, "../mismatched_scenes_report.txt")
        csv_path = os.path.join(BASE_DATA_DIR, "../mismatched_scenes_full_list.csv")
        
        sorted_mismatched = sorted(self.mismatched_pairs, key=lambda x: x['quality_score'])
        
        # Salvează raportul text
        with open(report_path, 'w') as f:
            f.write(f"RAPORT PERECHI CU CAMERĂ/UNGHI DIFERIT - {len(self.mismatched_pairs)} DETECTATE\n")
            f.write("="*100 + "\n\n")
            
            for pair in sorted_mismatched[:50]:
                f.write(f"Dataset: {pair['dataset']}\n")
                f.write(f"Filename: {pair['filename']}\n")
                f.write(f"Good Matches: {pair['good_matches']} | Coverage: {pair['coverage']:.2%} | Quality: {pair['quality_score']:.2%}\n")
                f.write(f"Motiv: {pair['reason']}\n")
                f.write(f"BEFORE: {pair['before_path']}\n")
                f.write(f"AFTER:  {pair['after_path']}\n")
                f.write("-" * 100 + "\n\n")
        
        # Salvează CSV
        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['dataset', 'filename', 'good_matches', 'total_matches', 
                                                    'coverage', 'quality_score', 'reason', 'before_path', 'after_path'])
            writer.writeheader()
            for pair in sorted_mismatched:
                writer.writerow({
                    'dataset': pair['dataset'],
                    'filename': pair['filename'],
                    'good_matches': pair['good_matches'],
                    'total_matches': pair.get('total_matches', 0),
                    'coverage': pair['coverage'],
                    'quality_score': pair['quality_score'],
                    'reason': pair['reason'],
                    'before_path': pair['before_path'],
                    'after_path': pair['after_path']
                })
        
        print(f"\n✅ Raport text salvat în: {report_path}")
        print(f"✅ CSV complet salvat în: {csv_path}")
    
    def get_statistics(self):
        """Afișează statistici generale."""
        print("\n" + "="*100)
        print("📈 STATISTICI GENERALE")
        print("="*100)
        
        for dataset_name, scores_list in self.scores.items():
            if not scores_list:
                continue
            
            good_matches_list = [s['good_matches'] for s in scores_list]
            quality_scores = [s['quality_score'] for s in scores_list]
            mismatched_count = sum(1 for s in scores_list if s['is_mismatched'])
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Total perechi: {len(scores_list)}")
            print(f"  Perechi mismatched: {mismatched_count} ({100*mismatched_count/len(scores_list):.1f}%)")
            print(f"\n  Good Matches:")
            print(f"    Min: {min(good_matches_list)}, Max: {max(good_matches_list)}, Mean: {np.mean(good_matches_list):.1f}")
            print(f"\n  Quality Score:")
            print(f"    Min: {min(quality_scores):.4f}, Max: {max(quality_scores):.4f}, Mean: {np.mean(quality_scores):.4f}")


if __name__ == "__main__":
    validator = SceneIntegrityValidator(
        min_matches=20,           # Minim 20 potriviri bune
        distribution_threshold=0.4  # Minim 40% din imagine
    )
    
    print("\n🚀 Inițiare validare integritate scenă...\n")
    
    # Analiza
    validator.analyze_dataset()
    
    # Statistici
    validator.get_statistics()
    
    # Previzualizare
    validator.display_results(top_n=10)
    
    # Salvare raport
    validator.save_report()
    
    print("\n⏸️  AȘTEPT CONFIRMAREA DVSTRĂ!")
    print(f"   Total perechi mismatched: {len(validator.mismatched_pairs)}")
    print("   Verificați previzualizarea de mai sus.\n")
