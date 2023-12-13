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
from misc import idx2text, ctc_decoder, plot_error_curves_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Lip Reading')
    parser.add_argument('--data', type=str, default='/data/ziruiw3/lip_reading/frames', help='path to dataset')
    parser.add_argument('--rnn', type=str, default='gru', help=' support gru and lstm')
    parser.add_argument('--visualize',type=bool, default=False, help='visualize error curve')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='path to save model')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    
    args = parser.parse_args()
    return args

def test(rnn):
    args = parse_args()

    # set up cuda
    device = torch.device('cuda:0' if  torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # load model
    model = DenseNet3D(rnn).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'densenet3d-{}.pth'.format(rnn))))
    print("model loaded")

    # load data
    train_loader, valid_loader = dataset.get_dataloaders(root_path=args.data,
                                                         batch_size=1,
                                                         split=0.8,
                                                         shuffle=True,
                                                         num_workers=args.num_workers,
                                                         pin_memory=False)
    
    model.eval()

    with torch.no_grad():
        # initialize
        eval_loss = []
        eval_WER = []
        eval_CER = []
        
        criterion = torch.nn.CTCLoss()
        iteration = 0
        for _, (video, label, video_length, label_length) in enumerate(valid_loader):
            if video is None:
                continue
            video, label, video_length, label_length = video.to(device), label.to(device), video_length.to(device), label_length.to(device)
            output = model(video)

            # calculate loss
            loss = criterion(output.log_softmax(-1).transpose(0,1), label, video_length.view(-1), label_length.view(-1))
            eval_loss.append(loss.detach().cpu().numpy())

            pred_text = ctc_decoder(output)
            print("[prediction]: ", str(pred_text))
            gt_text = []
            for _ in range(len(label)):
                gt_text.append(idx2text(label[_]))
            print("ground truth]: ", str(gt_text))
            # calculate WER and CER
            wer = CalculateErrorRate(gt_text[0], pred_text[0], method='WER')
            cer = CalculateErrorRate(gt_text[0], pred_text[0], method='CER')
            eval_WER.append(wer)
            eval_CER.append(cer)

            mean_WER = np.mean(np.array(eval_WER))
            mean_CER = np.mean(np.array(eval_CER))
        
            # print statistics
            iteration += 1
            if iteration % 20 == 0:
                print('Eval: Iteration: {}, Loss: {:.3f}, WER: {:.3f}, CER: {:.3f}'.format(iteration,np.mean(np.array(eval_loss)), mean_WER, mean_CER))

        return np.mean(np.array(eval_loss)), eval_WER, eval_CER
    
if __name__ == '__main__':
    args = parse_args()
    outputs = {}
    loss, wer, cer = test('gru')
    outputs['gru'] = {'train_loss': loss, 'train_wer': wer, 'train_cer': cer}
    loss, wer, cer = test('lstm')
    outputs['lstm'] = {'train_loss': loss, 'train_wer': wer, 'train_cer': cer}
    
    plot_error_curves_comparison(outputs, mode='test')