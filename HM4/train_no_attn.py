from data import english_to_french
import torch
from torch import nn
from jlib.get_enfr_loader import EnglishToFrench, Language, get_enfr_loader
from jlib.encoderdecoder import Translator
from torch.utils.data import DataLoader


en2fr = EnglishToFrench(english_to_french, max_length=10, gpu=True)
train_loader = DataLoader(
    en2fr,
    batch_size = 8,
    shuffle = True
)

val_loader = DataLoader(
    en2fr,
    batch_size = 8,
    shuffle = True
)

translator = Translator(
    input_size = en2fr.en.n_words,
    output_size = en2fr.fr.n_words,
    teacher_forcing_ratio=0.5,
    hidden_size = 1024,
    n_layers = 5,
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
    max_negative_diff_count=100
)

fig = translator.plot_training('English To French No Attn')
fig.savefig('plots/p1.png')

torch.save(translator.state_dict(), 'models/p1.pth')
