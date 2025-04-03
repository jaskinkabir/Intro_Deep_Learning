from transformer_components import TransformerEncoder, ClassifierHead
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam, Optimizer



class TransformerCharPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        inner_dim: int,
        num_attn_heads: int,
        num_attn_layers: int,
        cls_head_dims: list[int],
        dropout = 0,
        max_len: int = 5000,
        device: str = 'cuda'        
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_attn_heads = num_attn_heads
        self.num_attn_layers = num_attn_layers
        self.cls_head_dims = cls_head_dims
        self.dropout = dropout
        self.max_len = max_len
        self.device = device
        
        self.op = nn.Sequential(
            TransformerEncoder(
                input_dim=input_size,
                hidden_dim=hidden_dim,
                inner_dim=inner_dim,
                num_attn_heads=num_attn_heads,
                dropout=dropout,
                num_attn_layers=num_attn_layers,
                max_len=max_len,
            ),
            ClassifierHead(
                in_dim=hidden_dim,
                out_dim=output_size,
                layer_dims=cls_head_dims,
                dropout=dropout
            )
        )
        
        self.to(device)
        
    def forward(self, x):
        return self.op(x)
    def predict(self, x):
        return F.log_softmax(self.op(x), dim=-1)
    
        
        

