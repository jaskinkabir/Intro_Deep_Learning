import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from torch.amp import GradScaler, autocast
from torchtnt.utils.data import CudaDataPrefetcher
import time
import matplotlib.pyplot as plt

from torch.jit import script


class SequentialSkip(nn.Module):
    def __init__(self, sequential: nn.Sequential, dropout: int = 0):
        super().__init__()
        self.sequential = sequential
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.sequential(x)
        x = self.dropout_layer(x)
        x = x + residual
        return x

class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        layer_dims,
        dropout = 0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_dims = layer_dims
        self.dropout = dropout
        
        layer_dims = [in_dim] + layer_dims + [out_dim]
        
        layers = []
        for i in range(1, len(layer_dims)):
            linear_in = layer_dims[i-1]
            linear_out = layer_dims[i]
            layers.append(self._get_linear_layer(linear_in, linear_out))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
        
    
    def _get_linear_layer(self, in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).to(device)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TransformerCharPredictor(nn.Module):
    def __init__(
        self,
        alphabet_size: int,
        hidden_dim: int,
        inner_dim: int,
        num_attn_heads: int,
        num_attn_layers: int,
        cls_head_dims: list[int],
        dropout = 0,
        max_len: int = 5000,
        device: str = 'cuda'        
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_attn_heads = num_attn_heads
        self.num_attn_layers = num_attn_layers
        self.cls_head_dims = cls_head_dims
        self.dropout = dropout
        self.max_len = max_len
        self.device = device
        
        self.encoding = PositionalEncoding(hidden_dim, max_len, device)
        self.embedding = nn.Embedding(alphabet_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_attn_heads,
                    dim_feedforward=inner_dim,
                    dropout=dropout,
                    activation='relu',
                    device=device,
                    batch_first=True
            ),
            num_layers=num_attn_layers,
        )
        self.head = ClassifierHead(
            in_dim=hidden_dim,
            out_dim=alphabet_size,
            layer_dims=cls_head_dims,
            dropout=dropout,
        )      
        
        
        self.to(device)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoding(x)
        x = self.encoder(x)
        x = self.head(x)
        return x
    
    def predict(self, x):
        return F.log_softmax(self.forward(x), dim=-1)

    
    def train_step(self, fetcher):
        epoch_train_loss = torch.zeros(1, device=self.device)
        self.train()
        
        for X_batch, Y_batch in fetcher:
            self.optimizer.zero_grad(set_to_none=True)
                                
            with autocast("cuda"):
                outputs = self.forward(X_batch)
                train_batch_loss = self.loss_fn(outputs.transpose(1, 2), Y_batch)
            self.scaler.scale(train_batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()                   

            epoch_train_loss += train_batch_loss
        return epoch_train_loss / len(fetcher.data_iterable)
    
    
    def eval_step(self, fetcher):
        num_correct_tokens = torch.zeros(1, device=self.device)
        total_tokens = torch.zeros(1, device=self.device)
        epoch_val_loss = torch.zeros(1, device=self.device)
        self.eval()
        with torch.no_grad():
            for X_val_batch, Y_val_batch in fetcher:
                with autocast('cuda'):
                    outputs = self.forward(X_val_batch)
                    val_batch_loss = self.loss_fn(outputs.transpose(1, 2), Y_val_batch)
                    
                    predicted_tokens = outputs.argmax(dim=-1)
                    correct_predictions = (predicted_tokens == Y_val_batch)
                    num_correct_tokens += correct_predictions.sum()
                    total_tokens += Y_val_batch.numel()                            

                epoch_val_loss += val_batch_loss                        
        accuracy = num_correct_tokens / total_tokens
        epoch_val_loss = epoch_val_loss / len(fetcher.data_iterable)        
        return epoch_val_loss, accuracy
    
    

    def train_model(
            self,
            epochs,
            train_fetcher: CudaDataPrefetcher,
            val_fetcher: CudaDataPrefetcher,
            stop_on_plateau = False,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD,
            optimizer_args = [],
            optimizer_kwargs = {},
            print_epoch=1,
            header_epoch = 15,
            sched_factor = 0.1,
            sched_patience = 5,
            min_accuracy = 0.5,
            max_negative_diff_count = 6,
            save_path = None
        ):  
        
        
            train_start = time.perf_counter()
            self.scaler = GradScaler("cuda")
            self.optimizer = optimizer(self.parameters(), *optimizer_args, **optimizer_kwargs)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=sched_patience, factor=sched_factor)
            self.loss_fn = loss_fn
            self.train_loss_hist = torch.zeros(epochs)
            self.val_loss_hist = torch.zeros(epochs)
            self.accuracy_hist = torch.zeros(epochs)
            
            
            cell_width = 20
            header_form_spec = f'^{cell_width}'
            
            epoch_inspection = {
                "Epoch": 0,
                "Epoch Time (s)": 0,
                "Training Loss": 0,
                "Validation Loss ": 0,
                "Validation Accuracy": 0,
                "Δ Accuracy (%)": 0,
                'Memory Usage' : 0,
            }

            header_string = "|"
            for key in epoch_inspection.keys():
                header_string += (f"{key:{header_form_spec}}|")
            
            divider_string = '-'*len(header_string)
            if print_epoch:
                print(f'Training {self.__class__.__name__}\n')
                print(divider_string)
            max_accuracy = torch.zeros(1, device=self.device)            
            negative_acc_diff_count = 0           
            print("Begin Training")
            for epoch in range(epochs):
                begin_epoch = time.perf_counter()             
                
                #print('train step')
                epoch_train_loss = self.train_step(train_fetcher)
                #print('val step')
                epoch_val_loss, accuracy = self.eval_step(val_fetcher)
                
                self.train_loss_hist[epoch] = epoch_train_loss
                self.val_loss_hist[epoch] = epoch_val_loss
                self.accuracy_hist[epoch] = accuracy
                self.scheduler.step(accuracy)
                
                
                end_epoch = time.perf_counter()
                if print_epoch and (epoch % print_epoch == 0 or epoch == epochs - 1) :
                    mem = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())/1024**3
                    if header_epoch and epoch % header_epoch == 0:
                        print(header_string)
                        print(divider_string)
                    epoch_duration = end_epoch - begin_epoch
                    
                    d_accuracy = torch.zeros(1) if max_accuracy == 0 else 100 * (accuracy - max_accuracy) / max_accuracy
                    if d_accuracy <= 0:
                        negative_acc_diff_count += 1
                    else:
                        negative_acc_diff_count = 0

                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        if save_path:
                            torch.save(self.state_dict(), save_path)
                    
                    epoch_inspection['Epoch'] = f'{epoch}'
                    epoch_inspection['Epoch Time (s)'] = f'{epoch_duration:4f}'
                    epoch_inspection['Training Loss'] = f'{epoch_train_loss.item():8f}'
                    epoch_inspection['Validation Loss '] = f'{epoch_val_loss.item():8f}'
                    epoch_inspection['Validation Accuracy'] = f'{accuracy.item()*100:4f}'
                    epoch_inspection['Memory Usage'] = f'{mem:4f}'
                    epoch_inspection['Δ Accuracy (%)'] = f'{d_accuracy.item():4f}'
                    for value in epoch_inspection.values():
                        print(f"|{value:^{cell_width}}", end='')
                    print('|')
                    print(divider_string)
                    
                if stop_on_plateau and (accuracy > min_accuracy or negative_acc_diff_count > max_negative_diff_count):
                    break

            print(f'\nTraining Time: {(time.perf_counter() - train_start)*1000:4f} seconds\n')
            print(f'Max Accuracy: {max_accuracy.item()*100:4f}')

    def remove_zeros(self, array):
        return [x for x in array if x != 0]
    
    def plot_training(self, title: str) -> plt.Figure:
        loss_hist = self.train_loss_hist.cpu().detach().numpy()
        loss_hist = self.remove_zeros(loss_hist)
        
        val_loss_hist = self.val_loss_hist.cpu().detach().numpy()
        val_loss_hist = self.remove_zeros(val_loss_hist)
        validation_accuracy_hist = self.accuracy_hist.cpu().detach().numpy()
        validation_accuracy_hist = self.remove_zeros(validation_accuracy_hist)
        
        fig, ax = plt.subplots(1,2, sharex=True)
        fig.suptitle(title)
        ax[0].set_title('Loss Over Epochs')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].plot(loss_hist, label='Training Loss')
        ax[0].plot(val_loss_hist, label='Validation Loss')
        ax[0].legend()
        
        ax[1].set_title('Validation Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('%')
        ax[1].plot(validation_accuracy_hist)
        return fig  
    
        
        

