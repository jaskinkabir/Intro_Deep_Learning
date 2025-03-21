from data import english_to_french, english_to_french_qualitative
import torch
from torch import nn
from jlib.get_enfr_loader import EnFrDataset, Language, get_enfr_loaders, genLangs
from jlib.encoderdecoder import Translator, AttnDecoderRNN
from torch.utils.data import DataLoader


source_lang, target_lang = genLangs(english_to_french)
train_set = EnFrDataset(
    english_to_french,
    max_length=12,
    gpu=True,
    source_lang=source_lang,
    target_lang=target_lang,
)
val_set = EnFrDataset(
    english_to_french_qualitative,
    max_length=12,
    gpu=True,
    source_lang=source_lang,
    target_lang=target_lang,
)

train_set.reverse()
val_set.reverse()

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
)
val_loader = DataLoader(
    val_set,
    batch_size=len(val_set),
    shuffle=True,
)



translator = Translator(
    input_size = train_set.source_lang.n_words,
    output_size = train_set.target_lang.n_words,
    decoder = AttnDecoderRNN,
    teacher_forcing_ratio=0.6,
    hidden_size = 1024,
    max_sentence_length=12,
    n_layers = 1,
    dropout = 0.8,
)



translator.train_model(
    epochs = 100,
    header_epoch=10,
    train_loader = train_loader,
    val_loader = val_loader,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-5},
    min_accuracy=1,
    sched_patience=100,
    stop_on_plateau=False,
    save_path='models/fr2en_attn.pth'
)

fig = translator.plot_training('French To English With Attn')
fig.savefig('plots/fr2en_attn.png')

params = sum(p.numel() for p in translator.parameters())
# 13418773

#75% token
#2% sentence
# 15 second training
print(f'The model has {params} parameters')