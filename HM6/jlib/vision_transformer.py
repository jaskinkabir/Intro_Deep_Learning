import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import time
from torch.amp import GradScaler, autocast
from torchtnt.utils.data import CudaDataPrefetcher
import time
import matplotlib.pyplot as plt
import json
from torchprofile import profile_macs

from torch.jit import script

import dataclasses

@dataclasses.dataclass
class History:

    training_time: float = dataclasses.field(default=0)
    epochs: int = dataclasses.field(default=0)
    max_accuracy: float = dataclasses.field(default=0)
    min_val_loss: float = dataclasses.field(default=0)
    min_train_loss: float = dataclasses.field(default=0)
    parameter_count: int = dataclasses.field(default=0)
    macs: int = dataclasses.field(default=0)
    train_loss_hist: list = dataclasses.field(default_factory=list)
    val_loss_hist: list = dataclasses.field(default_factory=list)
    accuracy_hist: list = dataclasses.field(default_factory=list)
    
    def __post_init__(self):
        self.train_loss_hist = self.remove_zeros(self.train_loss_hist)
        self.val_loss_hist = self.remove_zeros(self.val_loss_hist)
        self.accuracy_hist = self.remove_zeros(self.accuracy_hist)
        
        self.max_accuracy = max(self.accuracy_hist)
        maxacc_idx = self.accuracy_hist.index(self.max_accuracy)
        self.min_val_loss = self.val_loss_hist[maxacc_idx]
        self.min_train_loss = self.train_loss_hist[maxacc_idx]
        
    
    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=4)
    def remove_zeros(self, array):
        return [x for x in array if x != 0]
    
    def plot_training(self, title: str) -> plt.Figure:
        loss_hist = self.train_loss_hist
        
        val_loss_hist = self.val_loss_hist
        validation_accuracy_hist = self.accuracy_hist
        
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

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x:torch.Tensor):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
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


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        inner_dim: int,
        num_classes: int,
        num_attn_heads: int,
        num_attn_layers: int,
        cls_head_dims: list,
        dropout = 0,
        max_len: int = 5000,
        device: str = 'cuda'        
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.num_classes = num_classes
        self.num_attn_heads = num_attn_heads
        self.num_attn_layers = num_attn_layers
        self.cls_head_dims = cls_head_dims
        self.dropout = dropout
        self.max_len = max_len
        self.device = device
        
        self.embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches = self.embedding.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_attn_heads,
                dim_feedforward=inner_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                device=device
            ),
            num_layers=num_attn_layers,
        )
        
        
        self.head = ClassifierHead(
            in_dim=embed_dim,
            out_dim=num_classes,
            layer_dims=cls_head_dims,
            dropout=dropout,
        )      
        
        self.param_count = sum(p.numel() for p in self.parameters())
        
        self.to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = self.embedding(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.head(x)
        return x
    
    def predict(self, x):
        return torch.argmax(self.forward(x), dim=-1)

    
    def train_step(self, fetcher, num_batches):
        epoch_train_loss = torch.zeros(1, device=self.device)
        self.train()
        
        for X_batch, Y_batch in fetcher:
            self.optimizer.zero_grad(set_to_none=True)
                                
            with autocast("cuda"):
                outputs = self.forward(X_batch)
                train_batch_loss = self.loss_fn(outputs, Y_batch)
                epoch_train_loss += train_batch_loss
            self.scaler.scale(train_batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()                   

        return epoch_train_loss / num_batches
    
    
    def eval_step(self, fetcher, num_batches):
        num_correct_pred = torch.zeros(1, device=self.device)
        total_pred = torch.zeros(1, device=self.device)
        epoch_val_loss = torch.zeros(1, device=self.device)
        self.eval()
        with torch.no_grad():
            for X_val_batch, Y_val_batch in fetcher:
                with autocast('cuda'):
                    outputs = self.forward(X_val_batch)
                    val_batch_loss = self.loss_fn(outputs, Y_val_batch)
                    
                    predicted_tokens = outputs.argmax(dim=-1)
                    correct_predictions = (predicted_tokens == Y_val_batch)
                    num_correct_pred += correct_predictions.sum()
                    total_pred += Y_val_batch.numel()                            

                    epoch_val_loss += val_batch_loss                        
        accuracy = num_correct_pred / total_pred
        epoch_val_loss = epoch_val_loss / num_batches
        return epoch_val_loss, accuracy


    def train_model(
            self,
            epochs,
            train_fetcher: CudaDataPrefetcher,
            num_train_batches: int,
            val_fetcher: CudaDataPrefetcher,
            num_val_batches: int,
            stop_on_plateau = False,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD,
            optimizer_args = [],
            optimizer_kwargs = {},
            print_epoch=1,
            header_epoch = 15,
            sched_factor = 1,
            min_accuracy = 0.5,
            max_negative_diff_count = 6,
            save_path = None
        ):  
            
            lmbda = lambda epoch: sched_factor ** epoch
            header_epoch = print_epoch * header_epoch
            train_start = time.perf_counter()
            self.scaler = GradScaler("cuda")
            self.optimizer = optimizer(self.parameters(), *optimizer_args, **optimizer_kwargs)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lmbda)
            self.loss_fn = loss_fn
            self.train_loss_hist = torch.zeros(epochs)
            self.val_loss_hist = torch.zeros(epochs)
            self.accuracy_hist = torch.zeros(epochs)
            d_accuracy = torch.zeros(1)
            
            test_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
            self.eval()
            with torch.no_grad():
                macs = profile_macs(self, test_input)
            
            
            
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
                epoch_train_loss = self.train_step(train_fetcher, num_train_batches)
                #print('val step')
                epoch_val_loss, accuracy = self.eval_step(val_fetcher, num_val_batches)
                
                self.train_loss_hist[epoch] = epoch_train_loss
                self.val_loss_hist[epoch] = epoch_val_loss
                self.accuracy_hist[epoch] = accuracy
                self.scheduler.step()
                
                
                end_epoch = time.perf_counter()
                d_accuracy = 100 * (accuracy - max_accuracy) / max_accuracy
                if d_accuracy <= 0:
                    negative_acc_diff_count += 1
                else:
                    negative_acc_diff_count = 0

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    if save_path:
                        torch.save(self.state_dict(), save_path)
                if stop_on_plateau and (accuracy > min_accuracy or negative_acc_diff_count > max_negative_diff_count):
                    break
                if print_epoch and (epoch % print_epoch == 0 or epoch == epochs - 1) :
                    mem = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())/1024**3
                    if header_epoch and epoch % header_epoch == 0:
                        print(header_string)
                        print(divider_string)
                    epoch_duration = end_epoch - begin_epoch
                    
                    
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
                    
            training_time = time.perf_counter() - train_start
            print(f'\nTraining Time: {training_time:4f} seconds\n')
            print(f'Max Accuracy: {max_accuracy.item()*100:4f}')
            
            return History(
                train_loss_hist=self.train_loss_hist.tolist(),
                val_loss_hist=self.val_loss_hist.tolist(),
                accuracy_hist=self.accuracy_hist.tolist(),
                training_time=training_time,
                parameter_count=self.param_count,
                macs=macs,
                epochs=epoch + 1
            )


    

    
        
        

