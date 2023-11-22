import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# reference: 
# https://github.com/VIPL-Audio-Visual-Speech-Understanding/Lipreading-DenseNet3D/blob/master/models/Dense3D.py#L35
# https://github.com/VIPL-Audio-Visual-Speech-Understanding/Lipreading-DenseNet3D/blob/master/main.py


def validate(model_output, length, labels, total=None, wrong=None):
    averageEnergies = torch.sum(model_output.data, 1)
    for i in range(model_output.size(0)):
        averageEnergies[i] = model_output[i,:length[i]].sum(0)

    _, maxindices = torch.max(averageEnergies, 1)
    count = 0

    for i in range(0, labels.squeeze(1).size(0)):
        l = int(labels.squeeze(1)[i].cpu())
        if total is not None:
            if l not in total:
                total[l] = 1
            else:
                total[l] += 1 
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
        else:
            if wrong is not None:
               if l not in wrong:
                   wrong[l] = 1
               else:
                   wrong[l] += 1

    return (averageEnergies, count)


class Evaluator():
    def __init__(self, mode, data_loader, device='cuda'):
        self.mode = mode
        assert self.mode in ['validation', 'test']
        self.data_loader = data_loader      # TODO: when dataloader is ready, change it
        self.device = device
    
    def __call__(self, model):
        with torch.no_grad():
            print(f"Starting {self.mode}...")
            count = np.zeros((len(self.validationdataset.pinyins)))     # TODO: change this when dataset is ready
            model.eval()
            if self.device == 'cuda':
                net = nn.DataParallel(model).cuda()
                
            num_samples = 0            
            for i_batch, sample_batched in enumerate(self.data_loader):
                input = Variable(sample_batched['temporalvolume']).cuda()   # TODO: change these when dataloader is ready
                labels = Variable(sample_batched['label']).cuda()
                length = Variable(sample_batched['length']).cuda()
                
                model = model.cuda()

                outputs = net(input)
                (vector, _) = validate(outputs, length, labels)

                argmax = (-vector.cpu().numpy()).argsort()
                for i in range(input.size(0)):
                    p = list(argmax[i]).index(labels[i])
                    count[p:] += 1                    
                num_samples += input.size(0)
                
                print(
                    f'i_batch/tot_batch:{i_batch}/{len(self.data_loader)},
                    corret/tot:{count[0]}/{len(self.data_loader)},
                    current_acc:{1.0*count[0]/num_samples}'
                )                

        return count/num_samples