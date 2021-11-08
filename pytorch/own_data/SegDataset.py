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
        
        original = np.load(original_path)
        original = np.expand_dims(original, axis=0)
        original = torch.from_numpy(original)

        segmentation = np.load(segmentation_path)
        segmentation = np.expand_dims(segmentation, axis=0)
        segmentation = torch.from_numpy(segmentation)

        if self.transform:
            original = self.transform(original)
        if self.target_transfrom:
            segmentation = self.target_transfrom(segmentation)
        return original, segmentation