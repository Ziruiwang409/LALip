import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def custom_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))

    if not batch:
        return None, None
    
    return torch.utils.data.dataloader.default_collate(batch)

class GRIDDataset(Dataset):
    '''
    Dataset class for one word videos from GRID corpus Dataset
    '''
    def __init__(self, path):
        '''
        Args:
            path (str): path to the dataset folder
        '''
        self.root = Path(path).__str__()
        self.video_dirs = GRIDDataset._get_video_dirs(path)

    def word2idx(self, word):
        chars = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
        chars = chars.split(' ')

        words = ['bin', 'lay', 'place', 'set', 'blue', 'green', 'red', 'white', 'at', 'by', 'in', 'with', 'a', 'an', 'the', 'no', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'again', 'now', 'please', 'soon']

        words.extend(chars)

        return words.index(word)

    def _get_video_dirs(path):
        video_dirs = []

        speaker_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        for dir in speaker_dirs:
            dir_path = os.path.join(path, dir)
            child_dirs = [os.path.join(dir, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            video_dirs.extend(child_dirs)

        return video_dirs
    
    def __getitem__(self, index):
        video_dir = os.path.join(self.root, self.video_dirs[index])
        frame_paths = GRIDDataset._get_paths(video_dir)
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path,0)
            frame = torch.from_numpy(frame)
            frame = frame.to(device)
            frames.append(frame)

        if len(frames) != 75:
            return None

        with open(os.path.join(video_dir, 'words.txt')) as f:
            words = f.read()

        words = words.split(' ')
        
        word_idx = [self.word2idx(word) for word in words]

        frames = torch.stack(frames)
        frames = frames.float()
        frames = frames/255.0
        frames = frames.unsqueeze(0)
        return frames, torch.tensor(word_idx)
    
    def _get_paths(data_path, file_type='.png'):
        paths = []

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(file_type):
                    paths.append(os.path.join(root, file))
        return paths
    
    def __len__(self):
        return len(self.video_dirs)
    

def get_dataloaders(root_path,
                    batch_size, 
                    split=0.8,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True):
    
    dataset = GRIDDataset(root_path)
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate)
    return train_loader, val_loader

    