import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class SequentialSkip(nn.Module):
    def __init__(self, sequential: nn.Sequential, dropout: int = 0):
        super().__init__()
        self.sequential = sequential
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.sequential(x)
        x = self.dropout_layer(x)
        x = x + residual
        return x

class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        layer_dims,
        dropout = 0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_dims = layer_dims
        self.dropout = dropout
        
        layer_dims = [in_dim] + layer_dims + [out_dim]
        
        layers = []
        for i in range(1, len(layer_dims)):
            linear_in = layer_dims[i-1]
            linear_out = layer_dims[i]
            layers.append(self._get_linear_layer(linear_in, linear_out))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        x = self.layers(x)
    
    def _get_linear_layer(self, in_dim, out_dim):
        return SequentialSkip(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            ),
            dropout=self.dropout
            
        )    


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim,
        inner_dim,
        dropout = 0,      
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.dropout = dropout
    
        self.op = SequentialSkip(
            nn.Sequential(
                nn.Linear(hidden_dim, inner_dim),
                nn.ReLU(),
                nn.Linear(inner_dim, hidden_dim)
            ),
            dropout=dropout
        )
        
        
    def forward(self, x):
        return self.op(x)
        

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).to(device)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class _TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        inner_dim,
        num_heads,       
        dropout = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.feed_forward = FeedForward(hidden_dim, inner_dim, dropout=self.dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attention.forward(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        inner_dim,
        num_attn_heads,
        num_attn_layers,
        dropout=0,
        max_len=5000,
        device = 'cuda'
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_attn_layers = num_attn_layers
        self.device = device
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len, device)
        self.encoder_layers = nn.Sequential(*[
            _TransformerEncoderLayer(
                hidden_dim,
                inner_dim,
                num_attn_heads,
                dropout
            )
            for _ in range(num_attn_layers)
        ])
    def forward(self, sequences):
        # sequences = sequences.to(self.device)
        x = self.embedding(sequences)
        x = self.positional_encoding(x)
        
        x = self.encoder_layers(x)
        
        return x

        

        