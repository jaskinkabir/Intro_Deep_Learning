from data import english_to_french, english_to_french_qualitative
import torch
from torch import nn
from jlib.get_enfr_loader import EnFrDataset, Language, get_enfr_loaders
from jlib.encoderdecoder import Translator
from torch.utils.data import DataLoader


en2fr = EnFrDataset(english_to_french, max_length=10, gpu=True)
en2fr_val = EnFrDataset(english_to_french_qualitative, max_length=10, gpu=True)

en2fr.reverse()
en2fr_val.reverse()
train_loader = DataLoader(
    en2fr,
    batch_size = 8,
    shuffle = True
)

val_loader = DataLoader(
    en2fr_val,
    batch_size = 8,
    shuffle = True
)
translator = Translator(
    input_size = en2fr.target_lang.n_words,
    output_size = en2fr.source_lang.n_words,
    teacher_forcing_ratio=0.5,
    hidden_size = 1024,
    n_layers = 1,
    dropout = 0.5,
)

translator.train_model(
    epochs = 120,
    train_loader = train_loader,
    val_loader = val_loader,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-5},
    min_accuracy=1,
    sched_patience=100,
    max_negative_diff_count=20,
    save_path='models/p1.pth'
)

fig = translator.plot_training('French To English No Attn')
fig.savefig('plots/p1.png')

