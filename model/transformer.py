import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model.Dense3D import Dense3D

class VideoToTextTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=512, nhead=8, num_encoder_layers=6, seq_length=6):
        super(VideoToTextTransformer, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model

        self.input_transform = nn.Linear(75*25*45, 6 * d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 6, d_model))

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.output_layer = nn.Linear(d_model, num_tokens)

    def forward(self, video_frames):
        batch_size = video_frames.size(0)

        video_frames = video_frames.view(batch_size, -1)
        video_frames = self.input_transform(video_frames)
        video_frames = video_frames.view(batch_size, 6, -1)
        video_frames = video_frames + self.positional_encoding
        transformer_output = self.transformer_encoder(video_frames)

        text_output = self.output_layer(transformer_output)
        return text_output
