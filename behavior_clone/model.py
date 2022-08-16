import math
from collections import deque

import torch
import torch.nn as nn
from torchinfo import summary


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return token_embedding + self.pos_embedding[:token_embedding.size(0), :]



class EyePredModel1(nn.Module):
    def __init__(self, img_size=(224, 224), token_size=128, max_len=5000, frozen_backbone=False) -> None:
        super().__init__()
        self.img_size = img_size
        self.token_size = token_size
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        if frozen_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
        self.cnn_encoder = nn.Sequential(*(list(resnet.children())[:-2])) #Remove last layers from resnet
        self.cnn_feature_reduction = nn.Conv2d(512, 32, 1) # Reduces the number of feature maps of the resnet
        self.image_embedding = nn.Linear(32 * 5 * 4, token_size) # Maps resnet output to embeddings for the transformer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_size, nhead=4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.position_encoding = PositionalEncoding(token_size, max_len)
        self.decoder = nn.Linear(self.token_size, 5)
        self.activation = nn.LeakyReLU(inplace=True)
        self.src_mask = None
        self.embedding_cache = deque(maxlen=max_len)

    def reset_history(self):
        """
        Reset the history token cache if a new sequence begins 
        """
        self.embedding_cache.clear()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor):
        # Check if we are in training mode 
        # In this case all images in the sequence are provided at the same time
        # and all resnet instances run in parralel 
        # In eval only one image is provided and we cache the resnet embeddings
        if self.training:
            # x is in shape, [batch_position, sequence_position, channel, img_x, img_y]
            batch_size, sequence_length, channels, img_size_x, img_size_y = x.size()
            # Combine sequence_position with batch_position, to process all images in the sequence with shared weights
            x = x.view(batch_size * sequence_length, channels, img_size_x, img_size_y)
            # Apply CNN reduction to all images in all batches
            x = self.cnn_encoder(x)
            # Reduce the resnet output to the token size
            x = self.image_embedding(self.activation(self.cnn_feature_reduction(x)).flatten(start_dim=1))
            # Recreate the sequence demension
            x = x.view(batch_size, sequence_length, self.token_size)
            # Bring input from shape [batch_size, sequence, token] in the shape [sequence, batch_size, token]
            x = x.transpose(0, 1)
        else:
            # Get frame dimensions
            batch_size, channels, img_size_x, img_size_y = x.size()
            # Calculate embedding for the new frame
            x = self.cnn_encoder(x)
            # Reduce the resnet output to the token size
            x = self.image_embedding(self.activation(self.cnn_feature_reduction(x)).flatten(start_dim=1))
            # Add the token of the new frame to the embedding cache
            self.embedding_cache.append(x)
            # Get sequence dimensions
            sequence_length = len(self.embedding_cache)
            # Create token sequence tensor
            x = torch.stack(tuple(self.embedding_cache)) 
        # Mask future in transformer
        if self.src_mask is None or self.src_mask.size(0) != sequence_length:
            device = x.device
            mask = self._generate_square_subsequent_mask(sequence_length).to(device)
            self.src_mask = mask
        # Run transformer
        x = self.decoder(self.transformer_encoder(self.position_encoding(x), self.src_mask))
        # Return input to shape [batch_size, sequence, token]
        x = x.transpose(0, 1)
        return x
    
    def __str__(self):
        return str(summary(self, (1, 3, self.img_size[0], self.img_size[1]), verbose=0))


class MLPSeq(nn.Module):
    def __init__(self, img_size=(224, 224), token_size=128, max_len=5, frozen_backbone=False) -> None:
        super().__init__()
        self.img_size = img_size
        self.token_size = token_size
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        if frozen_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
        self.cnn_encoder = nn.Sequential(*(list(resnet.children())[:-2])) #Remove last layers from resnet
        self.cnn_feature_reduction = nn.Conv2d(512, 32, 1) # Reduces the number of feature maps of the resnet
        self.image_embedding = nn.Linear(32 * 5 * 4, token_size) # Maps resnet output to embeddings for the transformer
        self.aggregation = nn.Linear(self.token_size * max_len, self.token_size)
        self.decoder = nn.Linear(self.token_size, 5)
        self.activation = nn.LeakyReLU(inplace=True)
        self.embedding_cache = deque([torch.zeros((1, token_size))] * max_len, maxlen=max_len)

    def reset_history(self):
        """
        Reset the history token cache if a new sequence begins 
        """
        self.embedding_cache.clear()

    def forward(self, x: torch.Tensor):
        # Check if we are in training mode 
        # In this case all images in the sequence are provided at the same time
        # and all resnet instances run in parralel 
        # In eval only one image is provided and we cache the resnet embeddings
        if self.training:
            # x is in shape, [batch_position, sequence_position, channel, img_x, img_y]
            batch_size, sequence_length, channels, img_size_x, img_size_y = x.size()
            # Combine sequence_position with batch_position, to process all images in the sequence with shared weights
            x = x.view(batch_size * sequence_length, channels, img_size_x, img_size_y)
            # Apply CNN reduction to all images in all batches
            x = self.cnn_encoder(x)
            # Reduce the resnet output to the token size
            x = self.image_embedding(self.activation(self.cnn_feature_reduction(x)).flatten(start_dim=1))
            # Recreate the sequence demension
            x = x.view(batch_size, sequence_length * self.token_size)
        else:
            # Get frame dimensions
            batch_size, channels, img_size_x, img_size_y = x.size()
            # Calculate embedding for the new frame
            x = self.cnn_encoder(x)
            # Reduce the resnet output to the token size
            x = self.image_embedding(self.activation(self.cnn_feature_reduction(x)).flatten(start_dim=1))
            # Add the token of the new frame to the embedding cache
            self.embedding_cache.append(x)
            # Get sequence dimensions
            sequence_length = len(self.embedding_cache)
            # Create token sequence tensor
            x = torch.cat(tuple(self.embedding_cache)) 
        # Run transformer
        x = self.decoder(self.activation(self.aggregation(self.activation(x))))
        return x
    
    def __str__(self):
        return str(summary(self, (1, 3, self.img_size[0], self.img_size[1]), verbose=0))
