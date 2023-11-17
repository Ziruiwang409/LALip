import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# Credit: 
# revised by Zirui Wang, 2023-11-17
# original code from DenseNet3D 
# see reference: https://github.com/VIPL-Audio-Visual-Speech-Understanding/Lipreading-DenseNet3D/blob/master/models/Dense3D.py


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        kernel_size = (1,2,2)
        stride = (1,2,2)
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=kernel_size, stride=stride))

class Dense3D(nn.Module):
    """
    Implementation of a densely connected convolutionnal network
    """
    def __init__(self, growth=8, number_initial_features=32, batch_norm_size=4, dropout=0):
        super(Dense3D, self).__init__()


        configuration = (4, 4)

        kernel = (1, 2, 2)
        stride = (1, 2, 2)

        # Initialise the network with a single layer
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, number_initial_features, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))),
            ('norm0', nn.BatchNorm3d(number_initial_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=kernel, stride=stride)),
        ]))


        number_features = number_initial_features
        for k, number_layers in enumerate(configuration):
            # add a dense block
            block = _DenseBlock(num_layers=number_layers, num_input_features=number_features,
                                bn_size=batch_norm_size, growth_rate=growth, drop_rate=dropout)

            self.features.add_module('denseblock%d' % (k + 1), block)

            number_features = number_features + number_layers * growth
            # add a transition layer (except last layer)
            if k != len(configuration) - 1:
                trans = _Transition(num_input_features=number_features, num_output_features=number_features)
                self.features.add_module('transition%d' % (k + 1), trans)

        self.features.add_module('norm%d' % (len(configuration)), nn.BatchNorm3d(number_features))
        self.features.add_module('pool', nn.AvgPool3d(kernel_size=kernel, stride=stride))

    def forward(self, x):
        return self.features(x)