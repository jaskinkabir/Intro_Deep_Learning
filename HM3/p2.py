import torch
from torch import nn
from jlib.classifier import Classifier
from jlib.get_shakespeare_loaders import *

text = ""
with open('data/sequence.txt', 'r') as f:
    text = f.read()

class ShakespeareRNN(Classifier):
    def __init__(self, alphabet_size, hidden_size, rnn=nn.RNN, linear_network=[], learning_rate = 1e-2, sequence_length = 20):
        super().__init__()
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(alphabet_size, hidden_size)
        self.rnn = rnn(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential()
        linear_in = hidden_size
        linear_out = alphabet_size
        for i, layer_size in enumerate(linear_network):
            self.fc.add_module(f'linear_{i}', nn.Linear(linear_in, layer_size))
            self.fc.add_module(f'batchnorm_{i}', nn.BatchNorm1d(layer_size))
            self.fc.add_module(f'relu_{i}', nn.ReLU())
            self.fc.add_module(f'dropout_{i}', nn.Dropout())
            linear_in = layer_size
        self.fc.add_module('final_linear', nn.Linear(linear_in, linear_out))
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Get the output of the last RNN cell
        return x

def train_and_plot(train, val, model: ShakespeareRNN, name, *training_args, **training_kwargs):
    print(f"Training {name}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count} parameters")
    model.train_model(
        *training_args,
        **training_kwargs,
        train_loader=train,
        val_loader=val,
    )
    fig = model.plot_training(f"{name} Training")
    fig.savefig(f"images/{name}_training_new.png")
    print(f"Model has {param_count} parameters")
    model.train_model(*training_args, **training_kwargs)
    fig = model.plot_training(f"{name} Training")
    fig.savefig(f"images/{name}_training_new.png")

# data_20 = gen_datasets(20)

data_20 = get_shakespeare_loaders(
    train_batch_size=2**15,
    val_batch_size=2**15,
    sequence_length=20,
)

lstm20 = ShakespeareRNN(
    alphabet_size=len(data_20['chars']),
    hidden_size=128,
    rnn=nn.LSTM,
).to('cuda')

train_and_plot(
    data_20['train_loader'],
    data_20['val_loader'],
    lstm20,
    "LSTM-20",
    epochs=50,
    optimizer = torch.optim.Adam,
    optimizer_kwargs = {'lr': 0.001},
    min_accuracy = 0.99,
    max_negative_diff_count = 7   
)