import torch
from torch import nn
from numbers import Number
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.amp import GradScaler, autocast
from torchtnt.utils.data import CudaDataPrefetcher
import time
import matplotlib.pyplot as plt

device = 'cuda'

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
    def remove_zeros(self, array):
        return [x for x in array if x != 0]
    def plot_training(self, title: str):
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
        #plt.show()
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
            sched_patience = 5,
            min_accuracy = 0.5,
            max_negative_diff_count = 6
        ):  
            
            scaler = GradScaler("cuda")
            optimizer = optimizer(self.parameters(), *optimizer_args, **optimizer_kwargs)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=sched_patience, factor=sched_factor)
            training_time = 0
            self.train_loss_hist = torch.zeros(epochs, device=device)
            self.val_loss_hist = torch.zeros(epochs, device=device)
            self.accuracy_hist = torch.zeros(epochs, device=device)
            
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
            negative_acc_diff_count = 0
            for epoch in range(epochs):
                
                begin_epoch = time.time()
                start_time = time.time()
                
                train_loss = 0
                total_train_samples = 0
                self.train()
                for X_batch, Y_batch in train_loader:
                    batch_size = X_batch.size(0)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        Y_pred = self.forward(X_batch)
                        loss = loss_fn(Y_pred, Y_batch) 
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()                   

                    train_loss += loss.item() * batch_size
                    total_train_samples += batch_size
                training_time += time.time() - start_time
                
                train_loss = train_loss/total_train_samples
                self.train_loss_hist[epoch] = train_loss
                
                del X_batch, Y_batch, loss, Y_pred
                val_start = time.time()
                val_correct = 0
                val_loss = 0
                total_val_samples = 0
                
                self.eval()
                with torch.no_grad():
                    Y_pred_eval = torch.zeros(len(val_loader.data_iterable.dataset)).to(device)
                    
                    idx = 0
                    for X_val_batch, Y_val_batch in val_loader:
                        batch_size = X_val_batch.size(0)
                        with autocast('cuda'):
                            Y_pred_logits = self.forward(X_val_batch)
                            batch_loss = loss_fn(Y_pred_logits, Y_val_batch) * batch_size
                        val_loss += batch_loss.item()
                        
                        Y_pred = Y_pred_logits.argmax(dim=1)
                        Y_pred_eval[idx:idx + batch_size] = Y_pred
                        val_correct += (Y_pred == Y_val_batch).sum().item()
                        idx += batch_size
                        
                        total_val_samples += batch_size
                val_time = time.time() - val_start
                    
                accuracy = val_correct/total_val_samples
                val_loss = val_loss/total_val_samples
                self.val_loss_hist[epoch] = val_loss                   
                self.accuracy_hist[epoch] = accuracy
                del X_val_batch, Y_val_batch, Y_pred_logits, Y_pred
                scheduler.step(accuracy)
                
                end_epoch = time.time()
                if print_epoch and (epoch % print_epoch == 0 or epoch == epochs - 1) :
                    mem = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())/1024**3
                    if header_epoch and epoch % header_epoch == 0:
                        print(header_string)
                        print(divider_string)
                    epoch_duration = end_epoch - begin_epoch
                    overfit = 100 * (val_loss - train_loss) / train_loss
                    d_accuracy = 0 if max_accuracy == 0 else 100 * (accuracy - max_accuracy) / max_accuracy
                    if d_accuracy <= 0:
                        negative_acc_diff_count += 1
                    else:
                        negative_acc_diff_count = 0

                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                    
                    epoch_inspection['Epoch'] = f'{epoch}'
                    epoch_inspection['Epoch Time (s)'] = f'{epoch_duration:4f}'
                    epoch_inspection['Validation Time'] = f'{val_time:4f}'
                    epoch_inspection['Training Loss'] = f'{train_loss:8f}'
                    epoch_inspection['Test Loss '] = f'{val_loss:8f}'
                    epoch_inspection['Overfit (%)'] = f'{overfit:4f}'
                    epoch_inspection['Accuracy (%)'] = f'{accuracy*100:4f}'
                    epoch_inspection['Δ Accuracy (%)'] = f'{d_accuracy:4f}'
                    epoch_inspection["GPU Memory (GiB)"] = f'{mem:2f}'
                    for value in epoch_inspection.values():
                        print(f"|{value:^{cell_width}}", end='')
                    print('|')
                    print(divider_string)
                    
                if accuracy > min_accuracy or negative_acc_diff_count > max_negative_diff_count:
                    break

            print(f'\nTraining Time: {training_time} seconds\n')
            self.last_pred = torch.tensor(Y_pred_eval)
            self.last_val = torch.tensor(val_loader.data_iterable.dataset.targets)
            
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
    def as_list(self):
        return [
            self.kernel,
            self.out_chan,
            self.stride,
            self.padding,
            self.in_chan
        ]
        