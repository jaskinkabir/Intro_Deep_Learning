from jlib.transformer_char_predictor import TransformerCharPredictor
import jlib.data_utils as data_utils
import torch
import torch.nn as nn
import numpy as np
text = ""
with open('data/sequence.txt', 'r') as f:
    text = f.read()
seqlen = 10

data = data_utils.gen_datasets(text, seqlen)
train_data = data['train_dataset']
val_data = data['val_dataset']
alphabet: data_utils.Alphabet = data['alphabet']

train_fetcher = data_utils.gen_data_loader(
    train_data,
    batch_size=32,
    workers = 6,
    cpu_prefetch= 20,
    gpu_prefetch=10
)

val_fetcher = data_utils.gen_data_loader(
    val_data,
    batch_size=len(val_data),
    workers = 6,
    cpu_prefetch= 10,
    gpu_prefetch=5
)

# model

model = TransformerCharPredictor(
    alphabet_size = len(alphabet),
    max_len = seqlen,
    hidden_dim = 512,
    inner_dim = 128,
    num_attn_heads = 2,
    num_attn_layers=2,
    cls_head_dims=[512, 256, 128],
    dropout = 0.2
)

model.train_model(
    epochs=10,
    train_fetcher=train_fetcher,
    val_fetcher=val_fetcher,
    optimizer = torch.optim.Adam,
    optimizer_kwargs={
        'lr': 1e-4,
        'weight_decay': 1e-5
    },
    min_accuracy=1,
    max_negative_diff_count=100,
    save_path='models/p1-10.pth'
)

model.plot_training('Small Corpus, Sequence Length 10')
