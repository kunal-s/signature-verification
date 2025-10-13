import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random

class SignaturePairsDataset(Dataset):
    def __init__(self, org_dir, forg_dir, transform=None):
        self.org_dir = org_dir
        self.forg_dir = forg_dir
        self.transform = transform

        self.org_files = [os.path.join(org_dir, f) for f in os.listdir(org_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
        self.forg_files = [os.path.join(forg_dir, f) for f in os.listdir(forg_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]

    def __len__(self):
        return max(len(self.org_files), len(self.forg_files))

    def __getitem__(self, idx):
        # Genuine pair
        img1_path = random.choice(self.org_files)
        img2_path = random.choice(self.org_files)
        label = 1 if os.path.basename(img1_path).split("_")[0] == os.path.basename(img2_path).split("_")[0] else 0

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float)
