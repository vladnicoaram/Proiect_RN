import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.before_dir = os.path.join(root_dir, 'before')
        self.after_dir  = os.path.join(root_dir, 'after')
        self.mask_dir   = os.path.join(root_dir, 'masks')

        self.files = sorted(os.listdir(self.before_dir))
        self.augment = augment
        
        # Data augmentation transforms (DOAR pentru imagini, NU pentru măști)
        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        before = cv2.imread(os.path.join(self.before_dir, fname))
        after  = cv2.imread(os.path.join(self.after_dir, fname))
        mask   = cv2.imread(os.path.join(self.mask_dir, fname), 0)

        before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
        after  = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

        before = before / 255.0
        after  = after / 255.0
        mask   = mask / 255.0

        # Aplică augmentare (DOAR imaginile, nu și masca)
        if self.augment:
            # Convertește la tensor pentru transforms
            before_pil = transforms.ToPILImage()(before.astype(np.float32))
            after_pil = transforms.ToPILImage()(after.astype(np.float32))
            
            # Aplică transformări
            before = np.array(self.augmentation(before_pil), dtype=np.float32) / 255.0
            after = np.array(self.augmentation(after_pil), dtype=np.float32) / 255.0

        # concat before + after → 6 canale
        x = np.concatenate([before, after], axis=2)
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)

        y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return x, y
