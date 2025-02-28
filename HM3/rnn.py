import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from jlib.classifier import *

text = ""
with open('data/sequence.txt', 'r') as f:
    text = f.read()

max_seq_length = 10

char_set = sorted(list(set(text)))
idx_to_char = {idx: char for idx, char in enumerate(char_set)}
char_to_idx = {char: idx for idx, char in enumerate(char_set)}

x = []
y = []
for i in range(len(text) - max_seq_length):
    sequence = text[i:i + max_seq_length]
    label = text[i + max_seq_length]
    x.append([char_to_idx[char] for char in sequence])
    y.append(char_to_idx[label])




x = np.array(x)
y = np.array(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

x_train = torch.tensor(x_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
x_val = torch.tensor(x_val, dtype=torch.long).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)


class CharRNN(ClassifierNoDataLoaders):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        embedding = self.embedding(x)
        output, _ = self.rnn(embedding)
        output = self.fc(output[:, -1, :])  # Get the output of the last RNN cell
        return output

hidden_size = 128
rnn = CharRNN(len(char_set), hidden_size, len(char_set)).to(device)
rnn.train_model(
    epochs=50,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-2},
    y_val=y_val,
    print_epoch=10,
    min_accuracy=0.99
)

fig = rnn.plot_training("RNN Training")
fig.savefig("images/rnn_training.png")