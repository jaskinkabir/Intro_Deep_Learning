from data import english_to_french
import torch
from torch import nn
from jlib.get_enfr_loader import EnglishToFrench, Language, get_enfr_loader
from jlib.encoderdecoder import Translator
from torch.utils.data import DataLoader

en2fr = EnglishToFrench(english_to_french, max_length=10, gpu=True)

state_dict = torch.load('models/p1.pth')
translator = Translator(
    input_size = en2fr.en.n_words,
    output_size = en2fr.fr.n_words,
    teacher_forcing_ratio=0.6,
    hidden_size = 1024,
    n_layers = 5,
    dropout = 0.5,
)
translator.load_state_dict(state_dict)
translator.decoder.teacher_forcing_ratio = 0
translator.to('cuda')
translator.eval()
translator.decoder.eval()
translator.encoder.eval()
while True:
    sentence = input('Enter a sentence: ')
    sentence.lower()
    if sentence == 'exit':
        break
    try:
        seq = en2fr.en.sentence_to_sequence(sentence)
    except KeyError:
        print("Error: Unkown Input Word")
        continue
    seq = torch.tensor(seq, dtype=torch.long).to('cuda').unsqueeze(0)
    output = translator.forward(seq)
    sentence = en2fr.fr.sequence_to_sentence(output.argmax(dim=-1).squeeze())
    print("Translation: ")
    print(sentence)
    print()
    


