import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.Dense3D import Dense3D

class DenseNet3D(nn.Module):
    def __init__(self):
        super(DenseNet3D, self).__init__()

        self.head = Dense3D()
        self.bi_gru_1 = nn.GRU(192 , 256, 1, bidirectional=True)
        self.bi_gru_2 = nn.GRU(512, 256, 1, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

        # classifier
        self.decoder_gru = nn.GRU(input_size=56, hidden_size=256, num_layers=1, bidirectional=False)
        self.fc1 = nn.Linear(256, 56)   # NOTE: number of classes  = 56 (26 characters +30 words)
        self.hidden_size_adjust = nn.Linear(512, 256)


    def forward(self, x, target_length=6):
        # Encoder
        x = self.head(x)
        x = x.permute(2, 0, 1, 3, 4)  # Rearrange dimensions: [num_frames, batch_size, channels, height, width]
        x = x.contiguous()
        x = x.view(x.size(1), x.size(0), -1)

        # Pass through the GRU layers
        x, hidden = self.bi_gru_1(x)
        x = self.dropout(x)
        x, hidden = self.bi_gru_2(x)
        x = self.dropout(x)

        # Global Average Pooling over the sequence dimension
        x = torch.mean(x, dim=1, keepdim=False)

        decoder_hidden = hidden[0, :, :] + hidden[1, :, :]
        decoder_hidden = decoder_hidden.unsqueeze(0)

        forward_hidden = hidden[0, :, :]
        backward_hidden = hidden[1, :, :]
        combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        decoder_hidden = self.hidden_size_adjust(combined_hidden).unsqueeze(0)

        # Decoder
        outputs = []
        for b in range(x.size(0)):  # Iterate over each element in the batch
            decoder_input = torch.zeros(1, 1, 56)
            current_hidden = decoder_hidden[:, b, :].unsqueeze(1)

            batch_outputs = []
            for i in range(target_length):
                # Decoder step for each time step
                decoder_output, current_hidden = self.decoder_gru(decoder_input, current_hidden)
                decoder_output = self.fc1(decoder_output)
                batch_outputs.append(decoder_output)

                # Update the input for the next step with current output
                decoder_input = decoder_output

            # Concatenate outputs for the current batch element
            batch_outputs = torch.cat(batch_outputs, dim=1)
            outputs.append(batch_outputs)

        # Concatenate outputs for all batch elements
        outputs = torch.cat(outputs, dim=0)  # (batch_size, target_length, 56)

        return outputs


if __name__ == '__main__':
    model = DenseNet3D()
    print(model)

    # Test the model with a random input
    x = torch.rand(2, 3, 16, 64, 128)
    print("Input shape:", x.shape)

    output = model(x)
    print("Output shape:", output.shape)




