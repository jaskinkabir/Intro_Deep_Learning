from transformer_components import TransformerEncoder, ClassifierHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
import time
from torch.amp import GradScaler, autocast
from torchtnt.utils.data import CudaDataPrefetcher
import time
import matplotlib.pyplot as plt



class TransformerCharPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
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
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_attn_heads = num_attn_heads
        self.num_attn_layers = num_attn_layers
        self.cls_head_dims = cls_head_dims
        self.dropout = dropout
        self.max_len = max_len
        self.device = device
        
        self.op = nn.Sequential(
            TransformerEncoder(
                input_dim=input_size,
                hidden_dim=hidden_dim,
                inner_dim=inner_dim,
                num_attn_heads=num_attn_heads,
                dropout=dropout,
                num_attn_layers=num_attn_layers,
                max_len=max_len,
            ),
            ClassifierHead(
                in_dim=hidden_dim,
                out_dim=output_size,
                layer_dims=cls_head_dims,
                dropout=dropout
            )
        )
        
        self.to(device)
        
    def forward(self, x):
        return self.op(x)
    def predict(self, x):
        return F.log_softmax(self.op(x), dim=-1)
    

    def train_model(
            self,
            epochs,
            train_loader: CudaDataPrefetcher,
            val_loader: CudaDataPrefetcher,
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
            scaler = GradScaler("cuda")
            self.optimizer = optimizer(self.encoder.parameters(), *optimizer_args, **optimizer_kwargs)
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
                "Validation Time": 0,
                "Avg Inference Time": 0,
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
            
            for epoch in range(epochs):
                begin_epoch = time.perf_counter()             
                
                epoch_train_loss = 0
                self.train()
                
                for X_batch, Y_batch in train_loader:
                    self.optimizer.zero_grad(set_to_none=True)
                                        
                    with autocast("cuda"):
                        outputs = self.forward(X_batch, Y_batch)
                        train_batch_loss = self.loss_fn(outputs.transpose(1, 2), Y_batch)
                    scaler.scale(train_batch_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()                   

                    epoch_train_loss += train_batch_loss.item() 
                
                epoch_train_loss = epoch_train_loss / len(train_loader)
                self.train_loss_hist[epoch] = epoch_train_loss
                
                del X_batch, Y_batch, train_batch_loss
                
                num_correct_tokens = 0
                total_tokens = 0
                epoch_val_loss = 0
                total_inference_time = 0
                self.eval()
                begin_val = time.perf_counter()
                with torch.no_grad():
                    for X_val_batch, Y_val_batch in val_loader:
                        with autocast('cuda'):
                            outputs = self.forward(X_val_batch)
                            val_batch_loss = self.loss_fn(outputs.transpose(1, 2), Y_batch)
                            
                            predicted_tokens = outputs.argmax(dim=-1)
                            correct_predictions = (predicted_tokens == Y_val_batch)
                            num_correct_tokens += correct_predictions.sum().item()
                            total_tokens += Y_val_batch.numel()                            

                        epoch_val_loss += val_batch_loss.item()                        
                total_inference_time = time.perf_counter() - begin_val    
                avg_inference_time = total_inference_time / len(val_loader)
                accuracy = num_correct_tokens / total_tokens
                epoch_val_loss = epoch_val_loss / len(val_loader)
                
                
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
                    
                    d_accuracy = 0 if max_accuracy == 0 else 100 * (accuracy - max_accuracy) / max_accuracy
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
                    epoch_inspection['Training Loss'] = f'{epoch_train_loss:8f}'
                    epoch_inspection['Validation Loss '] = f'{epoch_val_loss:8f}'
                    epoch_inspection['Validation Accuracy'] = f'{accuracy*100:4f}'
                    epoch_inspection['Memory Usage'] = f'{mem:4f}'
                    epoch_inspection['Avg Inference Time'] = f'{avg_inference_time:4e}'
                    epoch_inspection["Validation Time"] = f'{end_epoch - begin_val:4f}'
                    epoch_inspection['Δ Accuracy (%)'] = f'{d_accuracy:4f}'
                    for value in epoch_inspection.values():
                        print(f"|{value:^{cell_width}}", end='')
                    print('|')
                    print(divider_string)
                    
                if stop_on_plateau and (accuracy > min_accuracy or negative_acc_diff_count > max_negative_diff_count):
                    break

            print(f'\nTraining Time: {(time.perf_counter() - train_start)*1000:4f} seconds\n')
            print(f'Max Accuracy: {max_accuracy*100:4f}')

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
    
        
        

