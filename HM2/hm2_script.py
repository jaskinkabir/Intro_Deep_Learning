# %%
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torchtnt.utils.data import CudaDataPrefetcher
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import json
from torch import nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from numbers import Number
from math import floor
import time
from torch.amp import autocast, GradScaler
import gc


# %%
# Import cifar-10 dataset


cifar_10_deletables = []
cifar_100_deletables = []
models = []
data_path = './data'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% [markdown]
# # Define Preprocessing

# %%
def gen_data_loader(
    data,
    batch_size = 8192,
    workers = 6,
    cpu_prefetch = 10,
    gpu_prefetch = 10,
    clear=False
):
    start = time.perf_counter()
    if clear:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        gc.collect()

    print('Begin init data loader')
    loader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=workers,
        prefetch_factor=cpu_prefetch,
        pin_memory=True,
    )
    
    X_batch = next(iter(loader))[0]
    
    print(f"Batch Size: {X_batch.element_size() * X_batch.nelement() / 1024**2} MiB")
    print(f"Data Size: {X_batch.element_size() * data.__len__() * X_batch.nelement() / 1024**3} GiB")
    print(f"Data Loader init time: {time.perf_counter() - start:2f} s")
    print("Begin init fetcher")
    fetcher = CudaDataPrefetcher(
        data_iterable=loader,
        num_prefetch_batches=gpu_prefetch,
        device=torch.device('cuda')
    )
    print(f"Fetcher init time: {time.perf_counter() - start:2f} s")
    return fetcher

def get_cifar(
    is_cifar_10,
    recompute=False,
    redownload=False,
    data_path='./data'
):
    
    if is_cifar_10:
        delete_deletables(cifar_10_deletables)
    else:
        delete_deletables(cifar_100_deletables)
    title = 'cifar10' if is_cifar_10 else 'cifar100'
    cifar = datasets.CIFAR10 if is_cifar_10 else datasets.CIFAR100 
    
     
    if recompute:
        pre_cifar = cifar(data_path, train=True, download=redownload, transform=transforms.ToTensor())
        train_imgs = torch.stack([img for img, _ in pre_cifar], dim=3)
        mean = train_imgs.view(3, -1).mean(dim=1)
        std = train_imgs.view(3, -1).std(dim=1)
        torch.save(mean, f'data/mean_{title}.pt')
        torch.save(std, f'data/std_{title}.pt')
        del pre_cifar, train_imgs
    else:
        mean = torch.load(f'data/mean_{title}.pt')
        std = torch.load(f'data/std_{title}.pt')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar_train = cifar(data_path, train=True, download=redownload, transform=transform) 
    cifar_val = cifar(data_path, train=False, download=redownload, transform=transform)
    return cifar_train, cifar_val

def get_val_tensor(data):
    val_x = torch.empty(len(data), *data[0][0].shape, device='cuda')
    val_y = torch.empty(len(data), device='cuda', dtype=torch.long)
    
    for idx, (x, y) in enumerate(data):
        val_x[idx] = x.to('cuda')
        val_y[idx] = y
    return val_x, val_y
def delete_deletables(deletables):
    for d in deletables:
        try :
            del d
        except:
            pass
    deletables.clear()
    gc.collect()

# %%
# # CIFAR10
# batch_size_10 = 8192
# workers_10 = 6
# cpu_prefetch_10 = 10
# gpu_prefetch_10 = 10



# # Get cifar10
# cifar10_train, cifar10_val = get_cifar(is_cifar_10 = True, recompute=True, redownload=False)
# cifar_10_train_loader = gen_data_loader(data = cifar10_train, batch_size=batch_size_10, workers=workers_10, cpu_prefetch=cpu_prefetch_10, gpu_prefetch=gpu_prefetch_10)
# cifar_10_val_x, cifar_10_val_y = get_val_tensor(data = cifar10_val)
# cifar_10_deletables.extend([cifar10_train, cifar10_val, cifar_10_train_loader, cifar_10_val_x, cifar_10_val_y])


# %%


# %% [markdown]
# # Define Classifier Base

# %%


# %%


class Classifier(nn.Module):
    @classmethod
    def compare_results(cls, results1, results2):
        print('Comparing results:')
        
        for key, value in results1.items():
            if isinstance(value, Number): print(f"{key} : {100*(value - results2[key]) / value:2f} %") 
        
    def __init__(self):
        super().__init__()
    
    def get_results(self, Y_val=None, Y_pred=None):
        if Y_val is None:
            Y_val = self.last_val
        if Y_pred is None:
            Y_pred = self.last_pred
            
        if isinstance(Y_val, torch.Tensor):
            Y_val = Y_val.cpu().detach().numpy()
        if isinstance(Y_pred, torch.Tensor):
            Y_pred = Y_pred.cpu().detach().numpy()
        results = {
            'accuracy': accuracy_score(Y_val, Y_pred),
            'precision': precision_score(Y_val, Y_pred, average='weighted'),
            'recall': recall_score(Y_val, Y_pred, average='weighted'),
            'f1': f1_score(Y_val, Y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(Y_val, Y_pred),
            'classification_report': classification_report(Y_val, Y_pred)
        }
        self.last_results = results
        return results
    def print_results(self, results=None):
        if results is None:
            try: 
                results = self.last_results
            except:
                results = self.get_results()
        for key, value in results.items():
            if key in ['confusion_matrix', 'classification_report']:
                print(f'{key.capitalize()}:\n{value}')
            else:
                print(f'{key.capitalize()}: {value}')
    def plot_training(self, title: str):
        loss_hist = self.train_loss_hist.cpu().detach().numpy()
        val_loss_hist = self.val_hist.cpu().detach().numpy()
        accuracy_hist = self.train_accuracy_hist.cpu().detach().numpy()
        validation_accuracy_hist = self.validation_accuracy_hist.cpu().detach().numpy()
        
        fig, ax = plt.subplots(1,2, sharex=True)
        fig.suptitle(title)
        ax[0].set_title('Loss Over Epochs')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].plot(loss_hist, label='Training Loss')
        ax[0].plot(val_loss_hist, label='Validation Loss')
        
        
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].plot(accuracy_hist, label='Training Accuracy')
        ax[1].plot(validation_accuracy_hist, label='Validation Accuracy')
        
        plt.legend()
        plt.show()
    def plot_confusion_matrix(self, title):
        if not hasattr(self, 'last_results'):
            self.get_results()
        cm = self.last_results['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        disp.ax_.set_title(title)
    def forward(self, x):
        return self.sequential(x)
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)
    def train_model(
        self,
        epochs,
        train_loader: CudaDataPrefetcher,
        val_loader: CudaDataPrefetcher,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD,
        optimizer_args = [],
        optimizer_kwargs = {},
        print_epoch=10,
        header_epoch = 15,
        sched_factor = 0.1,
        sched_patience = 5
    ):  
        
        scaler = GradScaler("cuda")
        optimizer = optimizer(self.parameters(), *optimizer_args, **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=sched_patience, factor=sched_factor)
        training_time = 0
        train_hist = torch.zeros(epochs, device=device)
        val_hist = torch.zeros(epochs, device=device)
        accuracy_hist = torch.zeros(epochs, device=device)
        
        cell_width = 20
        header_form_spec = f'^{cell_width}'
        
        epoch_inspection = {
            "Epoch": 0,
            "Epoch Time (s)": 0,
            "Training Loss": 0,
            "Test Loss ": 0,
            "Overfit (%)": 0,
            "Accuracy (%)": 0,
            "Δ Accuracy (%)": 0,
            "Validation Time" : 0,
            "GPU Memory (GiB)": 0
        }

        header_string = "|"
        for key in epoch_inspection.keys():
            header_string += (f"{key:{header_form_spec}}|")
        
        divider_string = '-'*len(header_string)
        if print_epoch:
            print(f'Training {self.__class__.__name__}\n')
            print(divider_string)
        max_accuracy = torch.zeros(1, device=device)            
        for epoch in range(epochs):
            begin_epoch = time.time()
            self.train()
            
            start_time = time.time()
            train_loss = 0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    Y_pred = self.forward(X_batch)
                    loss = loss_fn(Y_pred, Y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss
            training_time += time.time() - start_time
            
            train_loss = train_loss/len(train_loader.data_iterable)
            train_hist[epoch] = train_loss
            
        
            val_start = time.time()
            val_correct = torch.zeros(1).to(device)
            val_loss = torch.zeros(1).to(device)
            self.eval()
            with torch.no_grad():
                Y_pred_eval = torch.zeros(len(val_loader.data_iterable.dataset)).to(device)
                
                idx = 0
                for X_val_batch, Y_val_batch in val_loader:
                    X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
                    batch_size = X_val_batch.size(0)
                    Y_pred_logits = self.forward(X_val_batch)
                    val_loss = loss_fn(Y_pred_logits, Y_val_batch)
                    
                    Y_pred = nn.functional.log_softmax(Y_pred_logits, dim=1).argmax(dim=1)
                    Y_pred_eval[idx:idx + batch_size] = Y_pred
                    val_correct += (Y_pred == Y_val_batch).sum()
                    idx += batch_size
            val_time = time.time() - val_start
                    
            accuracy = val_correct/len(val_loader.data_iterable.dataset)
            val_loss = val_loss/len(val_loader.data_iterable)
            val_hist[epoch] = val_loss                   
            accuracy_hist[epoch] = accuracy
            
            scheduler.step(accuracy)
            
            end_epoch = time.time()
            if print_epoch and (epoch % print_epoch == 0 or epoch == epochs - 1) :
                mem = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())/1024**3
                if header_epoch and epoch % header_epoch == 0:
                    print(header_string)
                    print(divider_string)
                epoch_duration = end_epoch - begin_epoch
                overfit = 100 * (val_loss - train_loss) / train_loss
                d_accuracy = torch.zeros(1) if max_accuracy == 0 else 100 * (accuracy - max_accuracy) / max_accuracy
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                
                epoch_inspection['Epoch'] = f'{epoch}'
                epoch_inspection['Epoch Time (s)'] = f'{epoch_duration:4f}'
                epoch_inspection['Validation Time'] = f'{val_time:4f}'
                epoch_inspection['Training Loss'] = f'{train_loss.item():8f}'
                epoch_inspection['Test Loss '] = f'{val_loss.item():8f}'
                epoch_inspection['Overfit (%)'] = f'{overfit.item():4f}'
                epoch_inspection['Accuracy (%)'] = f'{accuracy.item()*100:4f}'
                epoch_inspection['Δ Accuracy (%)'] = f'{d_accuracy.item():4f}'
                epoch_inspection["GPU Memory (GiB)"] = f'{mem:2f}'
                for value in epoch_inspection.values():
                    print(f"|{value:^{cell_width}}", end='')
                print('|')
                print(divider_string)
            

        print(f'\nTraining Time: {training_time} seconds\n')
        
        self.train_hist = train_hist
        self.val_hist = val_hist
        self.accuracy_hist = accuracy_hist
        self.last_val = torch.tensor(val_loader.dataset.targets)
    def train_model_old(
            self,
            epochs: int,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            alpha: float,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: nn.Module = torch.optim.SGD,
            print_epoch: int = 10,
            **optimizer_kwargs,
        ):
            self.train_loss_hist = torch.zeros(epochs).to(device)
            self.train_accuracy_hist = torch.zeros(epochs).to(device)
            self.validation_accuracy_hist = torch.zeros(epochs).to(device)
            
            scaler = GradScaler("cuda")
            optimizer = optimizer(self.parameters(), lr=alpha, **optimizer_kwargs)
            training_time = 0
            for epoch in range(epochs):
                self.train()
                
                start_time = time.time()
                train_loss = 0
                train_correct = torch.zeros(1).to(device)
                
                for X_batch, Y_batch in train_loader:
                    
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    optimizer.zero_grad()
                    with autocast("cuda"):
                        Y_pred = self.forward(X_batch)
                        loss = loss_fn(Y_pred, Y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    train_correct += (Y_pred.argmax(dim=1) == Y_batch).sum()
                training_time += time.time() - start_time
                
                self.train_accuracy_hist[epoch] = train_correct/len(train_loader.dataset)
                self.train_loss_hist[epoch] = train_loss/len(train_loader)
                
                val_start = time.time()
                val_correct = torch.zeros(1).to(device)
                self.eval()
                with torch.no_grad():
                    Y_pred_eval = torch.zeros(len(val_loader.dataset)).to(device)
                    
                    idx = 0
                    for X_val_batch, Y_val_batch in val_loader:
                        X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
                        batch_size = X_val_batch.size(0)
                        Y_pred = self.predict(X_val_batch)
                        Y_pred_eval[idx:idx + batch_size] = Y_pred
                        val_correct += (Y_pred == Y_val_batch).sum()
                        idx += batch_size
                        
                    self.validation_accuracy_hist[epoch] = val_correct/len(val_loader.dataset)
                epoch_time = time.time() - start_time
                
                        
                    
            
                if epoch % print_epoch == 0:
                    rem_time = (epochs - epoch)*epoch_time / 60
                    rem_time_str = f'{floor(rem_time):02}:{floor((rem_time - floor(rem_time))*60):02}'
                    print(f'Epoch {epoch}: Training Loss: {(train_loss/len(train_loader))}, Training Accuracy: {(train_correct/len(train_loader.dataset)).item()}, Validation Accuracy: {(val_correct/len(val_loader.dataset)).item()}, Estimated Time Remaining: {rem_time_str}')
            self.last_pred = torch.tensor(Y_pred_eval)
            self.last_val = torch.tensor(val_loader.dataset.targets)
            print(f'\nTraining Time: {training_time} seconds\n')
        
      

# %% [markdown]
# # Define AlexNet

# %%
class ConvParams:
    def __init__(self, kernel, out_chan, stride=1, padding='same', in_chan=0,):
        self.kernel = kernel
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.stride = stride
        self.padding = padding
    def __dict__(self):
        return {
            'kernel_size': self.kernel,
            'in_channels': self.in_chan,
            'out_channels': self.out_chan,
            'stride': self.stride,
            'padding': self.padding
        }
        
class AlexBlock(nn.Module):
    def __init__(
        self,
        params: ConvParams,
        pool_kernel = 2,
        pool_stride = 1,
    ):
        super().__init__()
        self.computation = nn.Sequential(
            # With batchnorm, bias is unnecessary
            nn.Conv2d(**params.__dict__(), bias=False),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel, pool_stride),
            nn.BatchNorm2d(params.out_chan),
        )
    def forward(self, x):
        return self.computation(x)
    

class AlexNet(Classifier):
        
    def __init__(
        self,
        in_chan,
        in_dim,
        num_classes,
        block_params: list = [],
        cnv_params =[],
        fc_layers = [],
        dropout = 0.5,
    ):
        super().__init__()
        self.cnv_layers = cnv_params
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.input_dim = in_chan
        self.num_classes = num_classes
        
        block_params[0].in_chan = in_chan
        self.sequential = nn.Sequential(AlexBlock(block_params[0]))
        for i in range(1, len(block_params)):
            block_params[i].in_chan = block_params[i-1].out_chan
            self.sequential.add_module(module=AlexBlock(block_params[i]), name=f'block_{i}')
        for i in range(len(cnv_params)):
            if i == 0:
                cnv_params[i].in_chan = block_params[-1].out_chan
            else:
                cnv_params[i].in_chan = cnv_params[i-1].out_chan
            self.sequential.add_module(module=nn.Sequential(
                nn.Conv2d(**cnv_params[i].__dict__()).to(device),
                nn.ReLU()
            ), name = f"conv_{i}")
        
        self.sequential.add_module(module=nn.Sequential(
            nn.MaxPool2d(3, 2),
            nn.Dropout2d(dropout),
            nn.Flatten() 
        ), name = 'flatten')
        self.sequential = self.sequential.to(device)
        
        dummy_in = torch.randn(1, in_chan, *in_dim).to(device)  # Add batch dimension
        dummy_out = self.sequential(dummy_in)
        fc_in = dummy_out.shape[1]
        self.sequential.add_module(name='linear_0', module=nn.Linear(fc_in, fc_layers[0]))
        self.sequential.add_module(name='relu_0', module=nn.ReLU())
        self.sequential.add_module(name='dropout_0', module=nn.Dropout(dropout))
        for i in range(1, len(fc_layers)):
            self.sequential.add_module(name=f'linear_{i}', module=nn.Sequential(
                nn.Linear(fc_layers[i-1], fc_layers[i]),
                nn.ReLU()
            ))
        self.sequential.add_module(name = 'output', module=nn.Linear(fc_layers[-1], num_classes))        
        self.sequential = self.sequential.to(device)
        dummy_out = self.sequential(dummy_in)
        print(dummy_out.shape)
        


# %% [markdown]
# # AlexNet CIFAR100

# %%

# Get cifar100

batch_size_100 = 1024
workers_100 = 6
cpu_prefetch_100 = 10
gpu_prefetch_100 = 5

batch_size_100_val = 128
workers_100_val = 2
cpu_prefetch_100_val = 2
gpu_prefetch_100_val = 2

cifar100_train, cifar100_val = get_cifar(is_cifar_10 = False, recompute=False, redownload=False)
cifar_100_train_loader = gen_data_loader(data = cifar100_train, batch_size=batch_size_100, workers=workers_100-workers_100_val, cpu_prefetch=cpu_prefetch_100-cpu_prefetch_100_val, gpu_prefetch=gpu_prefetch_100-gpu_prefetch_100_val, clear=True)
print("Val")
cifar_100_val_loader = gen_data_loader(data = cifar100_val, batch_size=batch_size_100_val, workers=workers_100_val, cpu_prefetch=cpu_prefetch_100_val, gpu_prefetch=gpu_prefetch_100_val)
cifar_100_deletables.extend([cifar100_train, cifar100_val, cifar_100_train_loader, cifar_100_val_loader])



# %%
try:
    delete_deletables([alex_100])
except:
    pass
retrain_alex_100 = True
if retrain_alex_100:

    torch.cuda.empty_cache()
    alex_100 = AlexNet(
        in_chan=3,
        in_dim=(32, 32),
        num_classes=100,
        block_params = [
            ConvParams(kernel=3, out_chan=96, stride=1, padding='same'),
            ConvParams(kernel=3, out_chan=64, stride=1, padding='same'),
        ],
        cnv_params = [
            ConvParams(kernel=3, out_chan=64, stride=1, padding='same'),
            ConvParams(kernel=3, out_chan=64, stride=1, padding='same'),
            ConvParams(kernel=3, out_chan=64, stride=1, padding='same'),
        ],
        fc_layers=[2048, 2048],
        dropout = 0.5
    ).to(device)

    alex_100.train_model(
        epochs=100,
        train_loader=cifar_100_train_loader,
        val_loader=cifar_100_val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        optimizer_args=[],
        optimizer_kwargs={'lr': 1e-2},
        print_epoch=1,
        header_epoch=10,
        sched_factor=0.1,
        sched_patience=5
    )
    torch.save(alex_100, 'models/alex_100.pth')
else:
    alex_100 = torch.load('models/alex_100.pth')
    alex_100.to(device)

# %%
alex_100.plot_training('AlexNet on CIFAR-100')
#alex_100.plot_confusion_matrix('AlexNet on CIFAR-100')

