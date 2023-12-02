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

# evaluation
from evaluation.WER_CER_metric import CalculateErrorRate
from dataset import dataset
from misc import word2idx, ctc_idx2word, idx2word, gt_label



def parse_args():
    parser = argparse.ArgumentParser(description='Lip Reading')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save path')
    #parser.add_argument('--train_data_path', type=str, default='data/train', help='train data path')
    #parser.add_argument('--val_data_path', type=str, default='data/val', help='val data path')
    parser.add_argument('--data_path', type=str, default='frames', help='train data path')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize error curve')

    args = parser.parse_args()
    return args

def text_decoder(output):
    output = output.argmax(-1) # (B, T=75, Emb=56) -> (B, T=75)
    length = output.size(0)
    text_list = []
    for _ in range(length):
        text_list.append(ctc_idx2word(output[_]))
    return text_list

def train():
    # parse arguments
    args = parse_args()
    print(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load data
    # TODO: Dataset to be implemented
    train_loader, valid_loader = dataset.get_dataloaders(root_path=args.data_path,
                                                         batch_size=1,
                                                         split=0.8,
                                                         shuffle=True,
                                                         num_workers=args.num_workers,
                                                         pin_memory=False)

    print("data loaded")
    # load model
    model = DenseNet3D()
    model.to(device)
    # loss function
    criterion = torch.nn.CTCLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("model loaded")
    # set to train mode
    model.train()
    iteration = 0

    WER = []
    CER = []

    # training
    for epoch in range(args.epochs):
        correct_train = 0
        total_train = 0
        losses = []

        for i, (video, label, video_length, label_length) in enumerate(train_loader):
            if video is None: # some videos have incorrect number of frames, skip them (see collate_fn in dataset.py)
                continue

            model.train()
            # zero the parameter gradients
            optimizer.zero_grad()

            video, label,video_length, label_length = video.to(device), label.to(device), video_length.to(device), label_length.to(device)

            
            # forward
            output = model(video)   #(T, B, C, H, W) -> (T, B, Emb=56)
            # update model
            loss = criterion(output.transpose(0, 1).log_softmax(-1), label, video_length.view(-1), label_length.view(-1))
            loss.backward()

            # decode prediction
            pred_text = text_decoder(output)
            print("prediction text:",pred_text)
            gt_text = [idx2word(label[i]) for i in range(label.size(0))]
            print("ground truth text:",gt_text)
            # calculate WER and CER
            for i in range(len(pred_text)):
                # print(gt_text[i])
                # print(pred_text[i])
                wer = CalculateErrorRate(gt_text[i], pred_text[i], method='WER')
                cer = CalculateErrorRate(gt_text[i], pred_text[i], method='CER')
                # print(wer, cer)
                WER.append(wer)
                CER.append(cer)
            # print(WER)
            # print(CER)
            exit()
            mean_WER = np.mean(WER)
            mean_CER = np.mean(CER)

            total_iter = epoch * len(train_loader) + i

            optimizer.step()

            # statistics
            print('Epoch: {}, Iteration: {}, Loss: {:.3f}, WER: {:.3f}, CER: {:.3f}'.format(epoch, iteration, loss.item(), mean_WER, mean_CER))


    # save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model-{}.pth'.format(epoch)))

    #if args.visualize:
        # TODO: not implemented


if __name__ == '__main__':
    train()