import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model.Dense3D import Dense3D

class VideoToTextTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=512, nhead=8, num_encoder_layers=6, seq_length=6):
        super(VideoToTextTransformer, self).__init__()
        self.feature_extractor = Dense3D()
        self.seq_length = seq_length
        self.d_model = d_model

        self.feature_transform = nn.Linear(14400, self.seq_length * self.d_model)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Output layer for text generation
        self.output_layer = nn.Linear(d_model, num_tokens)

    def forward(self, video_frames):
        # Extract features from video frames
        features = self.feature_extractor(video_frames)
        features = features.view(features.size(0), -1)  # Flatten
        features = self.feature_transform(features)
        features = features.view(-1, self.seq_length, self.d_model) 

        transformer_output = self.transformer_encoder(features)
        text_output = self.output_layer(transformer_output)
        return text_output
