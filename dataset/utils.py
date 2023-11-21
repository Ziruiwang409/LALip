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