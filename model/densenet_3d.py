import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Dense3D import Dense3D

class DenseNet3D(nn.Module):
    def __init__(self):
        super(DenseNet3D, self).__init__()

        self.head = Dense3D()

        self.bi_gru_1 = nn.GRU(3072, 256, 1, bidirectional=True)
        self.bi_gru_2 = nn.GRU(512, 256, 1, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

        # classifier
        self.fc1 = nn.Linear(512, 28)
    
    def forward(self, x):
        

        # Pass the input tensor containing the frames through the 3D dense network
        # (B, C=3, T, H=64, W=128) --> (B, C=96, T, H=4, W=8)
        x = self.head(x)
        print("output shape from dense3d head:", x.shape)

        # Rearrange the input before encoder
        # (B, C, T, H, W) -> (T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4)
        x = x.contiguous()

        # Concatenate Channel/Height/Width (B, C, T, H, W) -> (T, B, C*H*W)
        B = x.size(0)   # batch size
        C = x.size(1)   # channel size
        x = x.view(B, C, -1)


        x, _ = self.bi_gru_1(x)
        x = self.dropout(x)
        x, _ = self.bi_gru_2(x)
        x = self.dropout(x)

        # Fully connected layer
        #(T, B, Emb=512) --> (T, B, Emb=28)
        x = self.fc1(x)

        # (T, B, Emb) -> (B, T, Emb)
        x = x.permute(1, 0, 2)
        x = x.contiguous()

        return x
    
if __name__ == '__main__':
    model = DenseNet3D()
    print(model)

    # Test the model with a random input
    x = torch.rand(2, 3, 16, 64, 128)
    print("Input shape:", x.shape)

    output = model(x)
    print("Output shape:", output.shape)




