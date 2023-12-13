import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

from LALip.misc import text2idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def custom_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))

    if not batch:
        return None, None,None, None
    
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
        self.video_dirs = self._get_video_dirs(path)
    
    def __getitem__(self, index):
        video_dir = os.path.join(self.root, self.video_dirs[index])
        frame_paths = self._get_paths(video_dir)

        # 1. get video
        video = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path,0)
            frame = torch.from_numpy(frame)
            frame = frame.to(device)
            video.append(frame)

        # skip if number of video is not 75
        if len(video) != 75:
            return None

        video = torch.stack(video)
        video = video.float()
        video = video/255.0
        video = video.unsqueeze(0)
        video_length = video.shape[1]
        
        # 2. get text
        with open(os.path.join(video_dir, 'words.txt')) as f:
            text = f.read()
        label = text2idx(text.lower())
        label_length = label.shape[0]

        # add padding   NOTE: length to be added to config file
        label = self._add_padding(label, length=50)
        
        # convert to tensor
        label = torch.LongTensor(label)
 
        return video, label, video_length, label_length
    
    def __len__(self):
        return len(self.video_dirs)
    
    def _get_video_dirs(self, path):
        video_dirs = []

        speaker_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        for dir in speaker_dirs:
            dir_path = os.path.join(path, dir)
            child_dirs = [os.path.join(dir, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            video_dirs.extend(child_dirs)

        return video_dirs
    
    def _get_paths(self, data_path, file_type='.png'):
        paths = []

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(file_type):
                    paths.append(os.path.join(root, file))
        return paths
    
    def _add_padding(self, text_array, length=50):
        # convert to array if not already
        text_array = np.array(text_array)
        # add padding
        if len(text_array) < length:
            pad = np.zeros(length-len(text_array))
            text_array = np.concatenate((text_array, pad), axis=0)
        return text_array
        

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

    