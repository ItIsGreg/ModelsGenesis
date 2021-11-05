import torch
from torch.utils.data import Dataset
import os
import numpy as np

class SegDataset(Dataset):
    def __init__(self, originals_dir, segmentations_dir, transform=None, target_transform=None):
        self.originals_dir = originals_dir
        self.segmentations_dir = segmentations_dir
        self.transform = transform
        self.target_transfrom = target_transform
        self.ids = os.listdir(self.originals_dir)
    
    def __len__(self):
        return len(os.listdir(self.originals_dir))
    
    def __getitem__(self, idx):
        original_path = os.path.join(self.originals_dir, self.ids[idx])
        segmentation_path = os.path.join(self.segmentations_dir, self.ids[idx])
        original = torch.from_numpy(np.load(original_path))
        segmentation = torch.from_numpy(np.load(segmentation_path))
        if self.transform:
            original = self.transform(original)
        if self.target_transfrom:
            segmentation = self.target_transfrom(segmentation)
        return original, segmentation