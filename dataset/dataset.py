import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRIDDataset(Dataset):
    '''
    Dataset class for one word videos from GRID corpus Dataset
    '''
    def __init__(self, path, split):
        '''
        Args:
            path (str): path to the dataset folder
            split (str): train/val/test
        '''
        self.root = path
        self.split = split
        self.word2idx = {'bin': 0, 'lay': 1, 'place': 2, 'set': 3, 'blue': 4, 'green': 5, 'red': 6, 'white': 7, 'at': 8, 'by': 9, 'in': 10, 'with': 11, 'a': 12, 'an': 13, 'the': 14, 'no': 15, 'zero': 16, 'one': 17, 'two': 18, 'three': 19, 'four': 20, 'five': 21, 'six': 22, 'seven': 23, 'eight': 24, 'nine': 25, 'again': 26, 'now': 27, 'please': 28, 'soon': 29, 'tomorrow': 30, 'morning': 31, 'monday': 32, 'tuesday': 33, 'wednesday': 34, 'thursday': 35, 'friday': 36, 'saturday': 37, 'sunday': 38}