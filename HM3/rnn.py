import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from jlib.char_predictor import CharPredictor

text = ""
with open('data/sequence.txt', 'r') as f:
    text = f.read()

class CharRNN(CharPredictor):
    def __init__(self, text, sequence_length, hidden_size, rnn=nn.RNN):
        super().__init__()
        
        self.pass_text(text, sequence_length)
        
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.input_size, hidden_size)
        self.rnn = rnn(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Get the output of the last RNN cell
        return x
for name, rnn_type in [('RNN', nn.RNN), ('LSTM', nn.LSTM), ('GRU', nn.GRU)]:
    for sequence_length in [10, 20, 30]:
        print(f"Training {name} with sequence length {sequence_length}")
        hidden_size = 128
        rnn = CharRNN(text, sequence_length, hidden_size, rnn=rnn_type)
        param_count = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
        print(f"Model has {param_count} parameters")
        rnn.train_model(
            epochs=50,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={'lr': 1e-2},
            print_epoch=10,
            min_accuracy=0.99
        )

        fig = rnn.plot_training(f"{name}-{sequence_length} Training")
        fig.savefig(f"images/{name}_{sequence_length}_training_new.png")
        del rnn
