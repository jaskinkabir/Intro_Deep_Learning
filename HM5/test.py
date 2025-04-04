from jlib.transformer_char_predictor import TransformerCharPredictor
import jlib.data_utils as data_utils
import torch
import torch.nn as nn
import numpy as np

text = ""
with open('data/sequence.txt', 'r') as f:
    text = f.read()
def train_and_plot(seqlen: int):
    data = data_utils.gen_datasets(text, seqlen)
    train_data = data['train_dataset']
    val_data = data['val_dataset']
    alphabet: data_utils.Alphabet = data['alphabet']

    train_fetcher = data_utils.gen_data_loader(
        train_data,
        batch_size=32,
        workers = 6,
        cpu_prefetch= 20,
        gpu_prefetch=20
    )

    val_fetcher = data_utils.gen_data_loader(
        val_data,
        batch_size=len(val_data),
        workers = 6,
        cpu_prefetch= 10,
        gpu_prefetch=10
    )

    # model

    model = TransformerCharPredictor(
        alphabet_size = len(alphabet),
        max_len = seqlen,
        hidden_dim = 1024,
        inner_dim = 2048,
        num_attn_heads = 8,
        num_attn_layers=6,
        cls_head_dims=[],
        dropout = 0.1
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {param_count:,}")

    # test_input = next(iter(train_fetcher))[0]

    # print(f"Model MACs: {profile_macs(model, test_input):,}")
    
#Model parameter count: 77,219,372
#Model MACs: 49,602,887,704



    model.train_model(
        epochs=100,
        train_fetcher=train_fetcher,
        val_fetcher=val_fetcher,
        optimizer = torch.optim.Adam,
        optimizer_kwargs={
            'lr': 3e-3,
            'betas': (0.9, 0.98),
            'eps': 1e-9,
            'weight_decay': 1e-5
        },
        min_accuracy=1,
        max_negative_diff_count=10,
        save_path=f'models/p1-{seqlen}.pth',
        stop_on_plateau=True
    )

    model.plot_training(f'Small Corpus, Sequece Length {seqlen}')



train_and_plot(20)