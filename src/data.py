import os
import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

from src.utils import set_seed

class WatermarkDataset(Dataset):
    """
    PyTorch Dataset class that reads data based on splits.json and loads images from the data/raw/ directory.
    """
    def __init__(self, data_dir, splits_file, split='train'):
        self.data_dir = Path(data_dir) / "raw"
        
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
            
        if split not in splits_data:
            raise ValueError(f"Split {split} does not exist in {splits_file}")
            
        self.images = splits_data[split]
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.data_dir / self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        target_size = 128
        w, h = image.size
        
        ratio = target_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        image = F.resize(image, (new_h, new_w))
        
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        
        image = F.to_tensor(image)
        
        return image

def get_dataloaders(data_dir, splits_file, batch_size=32, num_workers=4):
    """
    Instantiates DataLoader classes for training, validation and testing.
    """
    set_seed()
    
    splits = ['train', 'val', 'test']
    dataloaders = {}
    
    for split in splits:
        try:
            dataset = WatermarkDataset(data_dir, splits_file, split=split)
            
            shuffle = True if split == 'train' else False
            drop_last = True if split == 'train' else False
            
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, drop_last=drop_last)
            dataloaders[split] = loader
        except (ValueError, FileNotFoundError):
            dataloaders[split] = None
            
    return dataloaders.get('train'), dataloaders.get('val'), dataloaders.get('test')

def get_dataset_stats(data_dir, splits_file):
    """
    Returns basic dataset statistics.
    """
    stats = {}
    
    try:
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
            
        for split, images in splits_data.items():
            stats[f"Number of {split} images"] = len(images)
            
        dataset = WatermarkDataset(data_dir, splits_file, split='train')
        if len(dataset) > 0:
            sample_shape = list(dataset[0].shape)
            
            min_val = float('inf')
            max_val = float('-inf')

            loader = DataLoader(dataset, batch_size=64, num_workers=4)
            for batch in loader:
                batch_min = batch.min().item()
                batch_max = batch.max().item()
                if batch_min < min_val:
                    min_val = batch_min
                if batch_max > max_val:
                    max_val = batch_max
                
            stats["Image shape"] = str(sample_shape)
            stats["Image value range"] = f"[{min_val:.1f}, {max_val:.1f}]"

    except (FileNotFoundError, ValueError) as e:
        stats["Warning"] = f"Failed to load dataset stats: {e}"

    return stats
