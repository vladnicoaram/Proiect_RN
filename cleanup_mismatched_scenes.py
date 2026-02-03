#!/usr/bin/env python3
"""
Script pentru ștergerea simetrică a perechilor mismatched din dataset.

Logica:
1. Citește mismatched_scenes_full_list.csv
2. Pentru fiecare pereche, șterge:
   - before_path (.jpg)
   - after_path (.jpg)
   - Masca din masks/ (.png)
   - Masca din masks_clean/ (dacă train)
3. Verifică integritatea după ștergere
4. Raportează statistici
"""

import os
import csv
from pathlib import Path
from collections import defaultdict

BASE_DATA_DIR = "/Users/admin/Documents/Facultatea/Proiect_RN/data"
CSV_FILE = "/Users/admin/Documents/Facultatea/Proiect_RN/mismatched_scenes_full_list.csv"

DATASETS = {
    'train': ['before', 'after', 'masks', 'masks_clean'],
    'test': ['before', 'after', 'masks'],
    'validation': ['before', 'after', 'masks']
}

class MismatchedScenesCleaner:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.mismatched_pairs = []
        self.deleted_count = 0
        self.failed_count = 0
        self.orphaned_files = defaultdict(list)
    
    def load_csv(self):
        """Citește CSV-ul cu perechi mismatched."""
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            self.mismatched_pairs = list(reader)
        
        print(f"✅ Citit CSV: {len(self.mismatched_pairs)} perechi mismatched")
        return self.mismatched_pairs
    
    def delete_file(self, filepath):
        """Șterge un fișier și returnează True dacă reușit."""
        if not filepath:
            return False
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except Exception as e:
                print(f"❌ Eroare la ștergere {filepath}: {e}")
                return False
        
        return False  # Fișierul nu există
    
    def get_mask_path(self, before_path, dataset_name, folder_type='masks'):
        """Convertește calea before/after în calea măștii."""
        # before_path: /data/train/before/image.jpg
        # mask_path: /data/train/masks/image.png
        
        dataset_path = os.path.dirname(os.path.dirname(before_path))  # /data/train
        filename = os.path.basename(before_path)  # image.jpg
        filename_png = filename.replace('.jpg', '.png')
        
        mask_path = os.path.join(dataset_path, folder_type, filename_png)
        return mask_path
    
    def preview_deletions(self):
        """Afișează preview al fișierelor care vor fi șterse."""
        print("\n" + "="*120)
        print("🔍 PREVIEW: Fișierele care VOR FI ȘTERSE")
        print("="*120)
        
        total_files_to_delete = 0
        
        for idx, pair in enumerate(self.mismatched_pairs[:5], 1):  # Doar primele 5 pentru preview
            before_path = pair['before_path']
            after_path = pair['after_path']
            dataset = pair['dataset']
            
            files_to_delete = []
            
            # Before
            if os.path.exists(before_path):
                files_to_delete.append(before_path)
            
            # After
            if os.path.exists(after_path):
                files_to_delete.append(after_path)
            
            # Mask
            mask_path = self.get_mask_path(before_path, dataset, 'masks')
            if os.path.exists(mask_path):
                files_to_delete.append(mask_path)
            
            # Mask clean (doar pentru train)
            if dataset == 'train':
                mask_clean_path = self.get_mask_path(before_path, dataset, 'masks_clean')
                if os.path.exists(mask_clean_path):
                    files_to_delete.append(mask_clean_path)
            
            print(f"\n{idx}. {pair['filename']}")
            print(f"   Fișiere de șters: {len(files_to_delete)}")
            for fp in files_to_delete:
                print(f"     - {fp}")
            
            total_files_to_delete += len(files_to_delete)
        
        print(f"\n... și {len(self.mismatched_pairs) - 5} perechi suplimentare\n")
        print(f"📊 TOTAL FIȘIERE DE ȘTERS: ~{total_files_to_delete * len(self.mismatched_pairs) // 5} fișiere\n")
    
    def execute_deletion(self, dry_run=False):
        """Execută ștergerea fișierelor."""
        print("\n" + "="*120)
        if dry_run:
            print("🔬 DRY-RUN: Simulare ștergere (fără a șterge efectiv)")
        else:
            print("⚠️  ȘTERGERE REALĂ: Se șterge acum...")
        print("="*120)
        
        deleted_before = 0
        deleted_after = 0
        deleted_masks = 0
        deleted_masks_clean = 0
        
        for idx, pair in enumerate(self.mismatched_pairs, 1):
            if idx % 200 == 0:
                print(f"Progres: {idx}/{len(self.mismatched_pairs)}")
            
            before_path = pair['before_path']
            after_path = pair['after_path']
            dataset = pair['dataset']
            
            # Șterge before
            if self.delete_file(before_path) if not dry_run else os.path.exists(before_path):
                deleted_before += 1
            
            # Șterge after
            if self.delete_file(after_path) if not dry_run else os.path.exists(after_path):
                deleted_after += 1
            
            # Șterge mask
            mask_path = self.get_mask_path(before_path, dataset, 'masks')
            if self.delete_file(mask_path) if not dry_run else os.path.exists(mask_path):
                deleted_masks += 1
            
            # Șterge mask_clean (doar pentru train)
            if dataset == 'train':
                mask_clean_path = self.get_mask_path(before_path, dataset, 'masks_clean')
                if self.delete_file(mask_clean_path) if not dry_run else os.path.exists(mask_clean_path):
                    deleted_masks_clean += 1
            
            self.deleted_count += 1
        
        print(f"\n✅ Ștergere completă!")
        print(f"   Before: {deleted_before}")
        print(f"   After: {deleted_after}")
        print(f"   Masks: {deleted_masks}")
        print(f"   Masks Clean: {deleted_masks_clean}")
        
        return deleted_before, deleted_after, deleted_masks, deleted_masks_clean
    
    def verify_integrity(self):
        """Verifică integritatea após ștergere - folderele trebuie să aibă același nr de fișiere."""
        print("\n" + "="*120)
        print("🔍 VERIFICARE INTEGRITATE - Foldere orfane și sincronizare")
        print("="*120)
        
        integrity_ok = True
        
        for dataset_name in DATASETS.keys():
            dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
            
            before_dir = os.path.join(dataset_path, 'before')
            after_dir = os.path.join(dataset_path, 'after')
            masks_dir = os.path.join(dataset_path, 'masks')
            
            before_files = set(os.listdir(before_dir)) if os.path.exists(before_dir) else set()
            after_files = set(os.listdir(after_dir)) if os.path.exists(after_dir) else set()
            masks_files = set(f.replace('.png', '.jpg') for f in os.listdir(masks_dir) if os.path.exists(masks_dir)) if os.path.exists(masks_dir) else set()
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Before: {len(before_files)} fișiere")
            print(f"  After:  {len(after_files)} fișiere")
            print(f"  Masks:  {len(masks_files)} fișiere")
            
            # Verifică sincronizare
            if len(before_files) != len(after_files):
                print(f"  ❌ EROARE: Before ({len(before_files)}) != After ({len(after_files)})")
                integrity_ok = False
            
            if len(before_files) != len(masks_files):
                print(f"  ❌ EROARE: Before ({len(before_files)}) != Masks ({len(masks_files)})")
                integrity_ok = False
            
            # Detectează orfani
            orphans_before = before_files - after_files
            orphans_after = after_files - before_files
            orphans_masks = masks_files - before_files
            
            if orphans_before:
                print(f"  ⚠️  Orfani în before: {len(orphans_before)} fișiere")
                self.orphaned_files[f"{dataset_name}_before"] = orphans_before
            
            if orphans_after:
                print(f"  ⚠️  Orfani în after: {len(orphans_after)} fișiere")
                self.orphaned_files[f"{dataset_name}_after"] = orphans_after
            
            if orphans_masks:
                print(f"  ⚠️  Orfani în masks: {len(orphans_masks)} fișiere")
                self.orphaned_files[f"{dataset_name}_masks"] = orphans_masks
            
            if len(before_files) == len(after_files) == len(masks_files):
                print(f"  ✅ Sincronizare OK")
        
        return integrity_ok
    
    def generate_report(self):
        """Generează raport final cu statistici."""
        print("\n" + "="*120)
        print("📊 RAPORT FINAL")
        print("="*120)
        
        # Calculează perechi rămase
        train_remaining = len(os.listdir(os.path.join(BASE_DATA_DIR, 'train', 'after')))
        test_remaining = len(os.listdir(os.path.join(BASE_DATA_DIR, 'test', 'after')))
        validation_remaining = len(os.listdir(os.path.join(BASE_DATA_DIR, 'validation', 'after')))
        
        total_remaining = train_remaining + test_remaining + validation_remaining
        
        print(f"\n📈 Statistici Dataset:")
        print(f"  Train:")
        print(f"    Eliminate: {len([p for p in self.mismatched_pairs if p['dataset'] == 'train'])}")
        print(f"    Rămase: {train_remaining}")
        print(f"\n  Test:")
        print(f"    Eliminate: {len([p for p in self.mismatched_pairs if p['dataset'] == 'test'])}")
        print(f"    Rămase: {test_remaining}")
        print(f"\n  Validation:")
        print(f"    Eliminate: {len([p for p in self.mismatched_pairs if p['dataset'] == 'validation'])}")
        print(f"    Rămase: {validation_remaining}")
        
        print(f"\n🎯 TOTAL:")
        print(f"  Perechi eliminate: {len(self.mismatched_pairs)}")
        print(f"  Perechi rămase: {total_remaining}")
        print(f"  % eliminat: {100*len(self.mismatched_pairs)/(len(self.mismatched_pairs)+total_remaining):.1f}%")
        
        if self.orphaned_files:
            print(f"\n⚠️  PROBLEME DETECTATE:")
            for category, files in self.orphaned_files.items():
                print(f"  {category}: {len(files)} fișiere orfane")
        else:
            print(f"\n✅ Nici un fișier orfan detectat - Dataset consistent!")


if __name__ == "__main__":
    cleaner = MismatchedScenesCleaner(CSV_FILE)
    
    print("\n🚀 Inițiare curățare perechi mismatched...\n")
    
    # Încarcă CSV
    cleaner.load_csv()
    
    # Afișează preview
    cleaner.preview_deletions()
    
    # Întrebare confirmație
    confirm = input("\n⚠️  CONFIRMARE: Doriți să continuați cu ȘTERGEREA REALĂ? (da/nu): ").strip().lower()
    
    if confirm != 'da':
        print("❌ Ștergere anulată de utilizator. Nicio fișier nu a fost șters.")
        exit(0)
    
    # Execută ștergerea
    cleaner.execute_deletion(dry_run=False)
    
    # Verifică integritate
    cleaner.verify_integrity()
    
    # Generează raport
    cleaner.generate_report()
    
    print("\n✅ Script finalizat!\n")
