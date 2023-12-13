# torch 
import torch

# utils
import os
import numpy as np
import argparse
import time

# model & dataset
from LALip.model.densenet_3d import DenseNet3D
from LALip.llm import LLM_Inference
from dataset import dataset

# evaluation
from metric.WERandCERmetrics import CalculateErrorRate
from LALip.misc import idx2text, ctc_decoder

# API key
API_KEY = "your-api-key"

def parse_args():
    parser = argparse.ArgumentParser(description='Lip Reading')
    parser.add_argument('--data', type=str, default='/data/ziruiw3/lip_reading/frames', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--rnn', type=str, default='gru', help=' support gru and lstm')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='path to save model')
    parser.add_argument('--enable_llm', type=bool, default=True, help='enable LLM correction')
    
    args = parser.parse_args()
    return args

def test(rnn='gru'):
    print("inferencing using {}".format(rnn.upper()))

    args = parse_args()
    print(args)

    # set up cuda
    device = torch.device('cuda:0' if  torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # load model
    model = DenseNet3D(rnn).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'densenet3d-{}.pth'.format(rnn))))

    LLM = LLM_Inference(API_KEY)
    print("model loaded")

    # load data
    train_loader, valid_loader = dataset.get_dataloaders(root_path=args.data,
                                                         batch_size=args.batch_size,
                                                         split=0.8,
                                                         shuffle=True,
                                                         num_workers=0,
                                                         pin_memory=False)
    
    model.eval()

    with torch.no_grad():
        # initialize
        eval_WER = []
        eval_CER = []
        iter = 0    
        for _, (video, label, video_length, label_length) in enumerate(valid_loader):
            if video is None:
                continue
            video, label, video_length, label_length = video.to(device), label.to(device), video_length.to(device), label_length.to(device)
            
            # inference
            t1 = time.time()
            # I.1 DenseNet prediction            
            output = model(video)
            pred_text = ctc_decoder(output)
            # print("Initial prediction: {}".format(pred_text))

            # I.2 LLM correction
            if args.enable_llm:
                for i in range(len(pred_text)):
                    pred_text[i] = pred_text[i].split(' ')
                pred_text = LLM.get_response(np.array(pred_text))
            # print("LLM correction: {}".format(pred_text))
            t2 = time.time()

            gt_text = []
            for _ in range(len(label)):
                gt_text.append(idx2text(label[_]))
            # calculate WER and CER
            for i in range(len(gt_text)):
                wer = CalculateErrorRate(gt_text[i], pred_text[i], method='WER')
                cer = CalculateErrorRate(gt_text[i], pred_text[i], method='CER')
                eval_WER.append(wer)
                eval_CER.append(cer)

            iter += 1
            if iter % 10 == 0:
                print("Iter: {}, WER: {}, CER: {}, avg inf time (per query): {}s".format(iter, np.mean(np.array(eval_WER)), np.mean(np.array(eval_CER)), (t2-t1)/args.batch_size))


        print("Evaluation results: ")
        print('WER: %.3f' % (np.mean(np.array(eval_WER))))
        print('CER: %.3f' % (np.mean(np.array(eval_CER))))


    
if __name__ == '__main__':
    args = parse_args()
    test(args.rnn)
