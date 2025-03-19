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
    hidden_size = 256,
    n_layers_en = 2,
    n_layers_de = 2,
    dropout_de=0.1,
    dropout_en=0.1,
)

translator.train_model(
    epochs = 35,
    train_loader = train_loader,
    val_loader = val_loader,
    loss_fn = nn.NLLLoss(),
    optimizer=torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-3},
    min_accuracy=1,
    sched_patience=100,
    max_negative_diff_count=100
)

translator.plot_training('English To French No Attn')

torch.save(translator, 'models/p1.pth')
