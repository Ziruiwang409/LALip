import os
from dataset.dataset import GRIDDataset
from torch.utils.data import DataLoader
import torch


def get_video_dirs(path):
    video_dirs = []

    speaker_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for dir in speaker_dirs:
        dir_path = os.path.join(path, dir)
        child_dirs = [os.path.join(dir, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        video_dirs.extend(child_dirs)

    return video_dirs



def get_dataloaders(root_path,
                    batch_size, 
                    split=0.8,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True):
    
    dataset = GRIDDataset(root_path)
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader