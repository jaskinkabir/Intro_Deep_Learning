from data import english_to_french
import torch
from torch import nn
from jlib.get_enfr_loader import EnglishToFrench, Language, get_enfr_loader
from jlib.encoderdecoder import Translator
from torch.utils.data import DataLoader

en2fr = EnglishToFrench(english_to_french, max_length=10, gpu=True)

state_dict = torch.load('models/p1.pth')
translator: Translator = Translator(
    input_size = en2fr.en.n_words,
    output_size = en2fr.fr.n_words,
    hidden_size = 256,
    n_layers_en = 2,
    n_layers_de = 2,
    dropout_de=0.1,
    dropout_en=0.1,
)
translator.load_state_dict(state_dict)
translator.to('cuda')
translator.eval()
while True:
    sentence = input('Enter a sentence: ')
    if sentence == 'exit':
        break
    try:
        seq = en2fr.en.sentence_to_sequence(sentence)
    except KeyError:
        print("Error: Unkown Input Word")
        continue
    seq = torch.tensor(seq, dtype=torch.long).to('cuda')
    _, _, output = translator.forward_pass(seq, seq)
    sentence = en2fr.fr.sequence_to_sentence(output)
    print("Translation: ")
    print(sentence)
    print()
    


