from data import english_to_french, english_to_french_qualitative
import torch
from torch import nn
from jlib.get_enfr_loader import EnFrDataset, Language, get_enfr_loaders
from jlib.encoderdecoder import Translator
from torch.utils.data import DataLoader

data = get_enfr_loaders(
    english_to_french,
    english_to_french_qualitative,

    cpu_prefetch=16,
    gpu_prefetch=16,
    train_batch_size=8,
    val_batch_size=109,
    workers=16,
)

train_set: EnFrDataset = data['train_set']


translator = Translator(
    input_size = train_set.source_lang.n_words,
    output_size = train_set.target_lang.n_words,
    teacher_forcing_ratio=0.5,
    hidden_size = 1024,
    n_layers = 1,
    dropout = 0.5,
)

translator.train_model(
    epochs = 120,
    train_loader = data['train_loader'],
    val_loader = data['val_loader'],
    optimizer=torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-5},
    min_accuracy=1,
    sched_patience=100,
    max_negative_diff_count=20,
    save_path='models/p1.pth'
)

fig = translator.plot_training('English To French No Attn')
fig.savefig('plots/p1.png')

