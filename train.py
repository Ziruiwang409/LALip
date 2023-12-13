# torch 
import torch

# utils
import os
import numpy as np
import argparse

# model & dataset
from model.densenet_3d import DenseNet3D
from dataset import dataset

# evaluation
from metric.WERandCERmetrics import CalculateErrorRate
from LALip.misc import idx2text, ctc_decoder, plot_error_curves_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Lip Reading')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save path')
    parser.add_argument('--data', type=str, default='/data/ziruiw3/lip_reading/frames/', help='train data path')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize error curve')

    args = parser.parse_args()
    return args

def train():
    # prepare log 
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    # parse args
    args = parse_args()
    log.write(str(args)+'\n')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.write('Using device: {}\n'.format(device))

    # load data
    train_loader, valid_loader = dataset.get_dataloaders(root_path=args.data,
                                                         batch_size=64,
                                                         split=0.8,
                                                         shuffle=True,
                                                         num_workers=args.num_workers,
                                                         pin_memory=False)
    print("data loaded")

    # output directory 
    outputs = {}

    # ---- training 3D CNN + GRU -----
    # load model
    model = DenseNet3D(rnn='gru')
    model.to(device)
    criterion = torch.nn.CTCLoss()                                  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    # optimizer
    print("model loaded")
    model.train() 

    # training
    WER = []
    CER = []
    iteration = 0
    print("start training...")

    for epoch in range(args.epochs):
        model.train()
        for _, (video, label, video_length, label_length) in enumerate(train_loader):

            if video is None: # some videos have incorrect number of frames, skip them (see collate_fn in dataset.py)
                continue

            # reset gradients
            optimizer.zero_grad()

            # load data to GPU
            video, label,video_length, label_length = video.to(device), label.to(device), video_length.to(device), label_length.to(device)

            # forward 
            output = model(video)   #(T, B, C, H, W) -> (B, T, Emb=28) (n, 75, 28)

            # backward
            loss = criterion(output.log_softmax(-1).transpose(0,1), label, video_length.view(-1), label_length.view(-1))
            loss.backward()

            # decode text output
            pred_text = ctc_decoder(output)

            gt_text = []
            for _ in range(len(label)):
                gt_text.append(idx2text(label[_]))

            # calculate WER and CER
            wer = CalculateErrorRate(gt_text[0], pred_text[0], method='WER')
            cer = CalculateErrorRate(gt_text[0], pred_text[0], method='CER')
            WER.append(wer)
            CER.append(cer)


            mean_WER = np.mean(np.array(WER))
            mean_CER = np.mean(np.array(CER))

            # update parameters
            optimizer.step()

            # statistics
            iteration += 1
            if iteration % 50 == 0:
                # print statistics and write to log
                print('Epoch [{}/{}], Iteration: {}, Loss: {:.3f}, WER: {:.3f}, CER: {:.3f}'.format(epoch + 1, args.epochs, iteration, loss.item(), mean_WER, mean_CER))
                log.write('Epoch [{}/{}], Iteration: {}, Loss: {:.3f}, WER: {:.3f}, CER: {:.3f}\n'.format(epoch + 1, args.epochs, iteration, loss.item(), mean_WER, mean_CER))

        # validation after each epoch
        evaluate(model, valid_loader, device, epoch,log)

        # save model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'densenet3d-gru-ep{}.pth'.format(epoch)))

        # flush log
        log.flush()

    # close log
    log.close()

def evaluate(model, valid_loader, device, epoch, log):
    model.eval()

    with torch.no_grad():
        # initialize
        eval_loss = []
        eval_WER = []
        eval_CER = []
        
        criterion = torch.nn.CTCLoss()

        for _, (video, label, video_length, label_length) in enumerate(valid_loader):
            if video is None:
                continue
            video, label, video_length, label_length = video.to(device), label.to(device), video_length.to(device), label_length.to(device)
            output = model(video)

            # calculate loss
            loss = criterion(output.log_softmax(-1).transpose(0,1), label, video_length.view(-1), label_length.view(-1))
            eval_loss.append(loss.detach().cpu().numpy())

            pred_text = ctc_decoder(output)
            gt_text = []
            for _ in range(len(label)):
                gt_text.append(idx2text(label[_]))
            
            # calculate WER and CER
            wer = CalculateErrorRate(gt_text[0], pred_text[0], method='WER')
            cer = CalculateErrorRate(gt_text[0], pred_text[0], method='CER')
            eval_WER.append(wer)
            eval_CER.append(cer)

            mean_WER = np.mean(np.array(eval_WER))
            mean_CER = np.mean(np.array(eval_CER))
        
        # print statistics
        print('Epoch {} Evaluation Loss: {:.3f}, eval WER: {:.3f}, eval CER: {:.3f}'.format(epoch+1, np.mean(np.array(eval_loss)), mean_WER, mean_CER))
        log.write('Epoch {} Evaluation Loss: {:.3f}, eval WER: {:.3f}, eval CER: {:.3f}\n'.format(epoch+1, np.mean(np.array(eval_loss)), mean_WER, mean_CER))


if __name__ == '__main__':
    train()