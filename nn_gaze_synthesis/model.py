import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])



class EyePredModel1(nn.Module):
    def __init__(self, img_size=224, token_size=128, max_len=5000) -> None:
        super().__init__()
        self.token_size = token_size
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.cnn_encoder = torch.nn.Sequential(*(list(resnet.children())[:-2])) #Remove last layers from resnet
        self.image_embedding = nn.Linear(512 * 7 * 7, token_size) # Maps resnet output to embeddings for the transformer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_size, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.position_encoding = PositionalEncoding(token_size, 0.1, max_len)
        self.decoder = nn.Linear(self.token_size, 2)
        self.activation = nn.LeakyReLU(inplace=True)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor):
        # x is in shape, [batch_position, sequence_position, channel, img_x, img_y]
        batch_size, sequence_length, channels, img_size_x, img_size_y = x.size()
        # Combine sequence_position with batch_position, to process all images in the sequence with shared weights
        x = x.view(batch_size * sequence_length, channels, img_size_x, img_size_y)
        # Apply CNN reduction to all images in all batches
        x = self.cnn_encoder(x)
        # Reduce the resnet output to the token size
        x = self.image_embedding(x.flatten(start_dim=1))
        # Recreate the sequence demension
        x = x.view(batch_size, sequence_length, self.token_size)
        # Mask future in transformer
        if self.src_mask is None or self.src_mask.size(1) != sequence_length:
            device = x.device
            mask = self._generate_square_subsequent_mask(sequence_length).to(device)
            self.src_mask = mask
        # Run transformer
        x = self.decoder(self.transformer_encoder(self.position_encoding(x)))
        return x
