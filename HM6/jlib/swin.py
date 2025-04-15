import transformers
import torch
import torch.nn as nn
from jlib.vision_transformer import History
import numpy as np
import torch.nn.functional as F
import time
from torch.amp import GradScaler, autocast
from torchtnt.utils.data import CudaDataPrefetcher
import time
import matplotlib.pyplot as plt
import json
from torchprofile import profile_macs

device = 'cuda:0'

class Swin(nn.Module):
    def __init__(self, model_name, num_classes=100, device='cuda'):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = transformers.SwinForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ).to(device)
        self.image_size = 32
        
        for param in self.model.swin.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-5,
            # weight_decay=1e-2,
            # betas=(0.9, 0.999),
            # eps=1e-8,
        )
        self.param_count = sum(p.numel() for p in self.model.parameters())
        print(f'Params: {param_count:4e}')
    def forward(self, x):
        return self.model(x).logits
    
    
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
                    mem = (torch.cuda.memory_allocated(self.device) + torch.cuda.memory_reserved(self.device))/1024**3
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
                
                torch.cuda.empty_cache()    
            training_time = time.perf_counter() - train_start
            print(f'\nTraining Time: {training_time:4f} seconds\n')
            print(f'Max Accuracy: {max_accuracy.item()*100:4f}')
            
            return History(
                train_loss_hist=self.train_loss_hist.tolist(),
                val_loss_hist=self.val_loss_hist.tolist(),
                accuracy_hist=self.accuracy_hist.tolist(),
                training_time=training_time,
                parameter_count=self.param_count,
                macs=0,
                epochs=epoch + 1,
            )
