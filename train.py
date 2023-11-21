# torch 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torch.autograd import Variable

# utils
import os
import numpy as np
import argparse

# statistics & visualization
import matplotlib.pyplot as plt
import statistics

# model & dataset
from model.densenet_3d import DenseNet3D

from dataset import dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Lip Reading')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save path')
    #parser.add_argument('--train_data_path', type=str, default='data/train', help='train data path')
    #parser.add_argument('--val_data_path', type=str, default='data/val', help='val data path')
    parser.add_argument('--data_path', type=str, default='data', help='train data path')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize error curve')

    args = parser.parse_args()
    return args

def train():
    # parse arguments
    args = parse_args()
    print(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load data
    # TODO: Dataset to be implemented
    train_loader, valid_loader = dataset.get_dataloaders(root_path=args.data_path,
                                                         batch_size=args.batch_size,
                                                         split=0.8,
                                                         shuffle=True,
                                                         num_workers=args.num_workers,
                                                         pin_memory=True)

    print("data loaded")
    # load model
    model = DenseNet3D()
    model.to(device)
    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("model loaded")
    # set to train mode
    model.train()
    iteration = 0

    # training
    for epoch in range(args.epochs):
        correct_train = 0
        total_train = 0
        losses = []

        for i, (video, label) in enumerate(train_loader):
            if video is None: # some videos have incorrect number of frames, skip them (see collate_fn in dataset.py)
                continue

            # empty cache   
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            print("video shape:", video.shape)
            #video = pad_sequence(video, batch_first=True)
            
            # forward
            output = model(video)

            _, predicted = torch.max(output.data, 1)

            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()    

            # loss
            label = label.to(device)
            loss = criterion(output, label)

            # backward
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            iteration += 1

            if iteration % 10 == 0: # validate every 10 iterations
                model.eval()
                with torch.no_grad():
                    correct_val = 0
                    total_val = 0
                    for i, (video, label) in enumerate(valid_loader):
                        if video is None: # some videos have incorrect number of frames, skip them (see collate_fn in dataset.py)
                            continue
                        video = pad_sequence(video, batch_first=True)
                        output = model(video)
                        _, predicted = torch.max(output.data, 1)
                        total_val += label.size(0)
                        correct_val += (predicted == label).sum().item()
                    print("----------------- Validation ----------------")
                    print('Epoch: {}, Iter: {},  Val_Acc: {}'.format(epoch, i, 100 * correct_val//total_val))
                    print("----------------------------------------------")
                model.train()

        # print training information
        print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, 100 * correct_train//total_train))

    # save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model-{}.pth'.format(epoch)))

    #if args.visualize:
        # TODO: not implemented


if __name__ == '__main__':
    train()