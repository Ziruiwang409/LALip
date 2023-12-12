import torch
import torch.nn as nn

from model.Dense3D import Dense3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DenseNet3D(nn.Module):
    def __init__(self, rnn='gru'):
        super(DenseNet3D, self).__init__()

        self.head = Dense3D()
        self.rnn = rnn
        if self.rnn == 'gru':
            self.bi_gru_1 = nn.GRU(192 , 256, 1, bidirectional=True)    # NOTE: 192 = 64 * 3 (3D DenseNet output channels)
            self.bi_gru_2 = nn.GRU(512, 256, 1, bidirectional=True)
            self.dropout = nn.Dropout(0.5)
        if self.rnn == 'lstm':
            self.bi_lstm_1 = nn.LSTM(192 , 256, 1, bidirectional=True)
            self.bi_lstm_2 = nn.LSTM(512, 256, 1, bidirectional=True)
            self.dropout = nn.Dropout(0.5)

        # classifier
        self.fc1 = nn.Linear(512, 28)   # NOTE: number of classes  = 28 (26 letters + blank + space)


    def forward(self, x):
        # Encoder
        x = self.head(x)


        x = x.permute(2, 0, 1, 3, 4)  # Rearrange dimensions: [num_frames, batch_size, channels, height, width]
        x = x.contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        if self.rnn == 'gru':
            # Pass through the GRU layers
            x, _ = self.bi_gru_1(x)
            x = self.dropout(x)
            x, _ = self.bi_gru_2(x)
            x = self.dropout(x)
        if self.rnn == 'lstm':
            # Pass through the LSTM layers
            x, _ = self.bi_lstm_1(x)
            x = self.dropout(x)
            x, _ = self.bi_lstm_2(x)
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




