import torch
from torch import nn
from numbers import Number
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.amp import GradScaler, autocast
from torchtnt.utils.data import CudaDataPrefetcher
import time
import matplotlib.pyplot as plt
from .get_enfr_loader import get_enfr_loader, SOS, EOS
from data import english_to_french


device = 'cuda'



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
    def forward(self, x, hidden):
        embedded = self.embed(x).view(1, 1, -1)
        outputs, hidden = self.rnn(embedded, hidden)
        return outputs, hidden
    
    def initHidden(self):
        # Initializes hidden state
        dummy_in = torch.zeros(1, 1, self.hidden_size, device=device)
        _, hidden = self.rnn(dummy_in)
        return torch.zeros_like(hidden, device=device)
class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout, n_layers=1):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x, hidden):
        embedded = self.embed(x).view(1,1,-1)
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
        
    def initHidden(self):
        # Initializes hidden state
        dummy_in = torch.zeros(1, 1, self.hidden_size, device=device)
        _, hidden = self.rnn(dummy_in)
        return torch.zeros_like(hidden, device=device)

class Translator(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        dropout_en,
        n_layers_en=1,
        dropout_de=0.1,
        n_layers_de=1,
        device='cuda'
    ):
        super().__init__()
        self.encoder: EncoderRNN = EncoderRNN(input_size, hidden_size, dropout_en, n_layers_en).to(device)
        self.decoder: DecoderRNN = DecoderRNN(output_size, hidden_size, dropout_de, n_layers_de).to(device)
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.loss_fn = nn.NLLLoss()
        self.en_optimizer = torch.optim.Adam(self.encoder.parameters())
        self.de_optimizer = torch.optim.Adam(self.decoder.parameters())
    def forward_pass(self, input_tensor: torch.tensor, target_tensor: torch.tensor) -> tuple[torch.Tensor, bool, list]:
        en_hidden = self.encoder.initHidden()
        
        self.en_optimizer.zero_grad()
        self.de_optimizer.zero_grad()
        
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        loss = 0
        correct_prediction = False
        
        # encoder loop
        en_hidden = self.encoder.initHidden()
        for ei in range(input_length):
            _, en_hidden = self.encoder(input_tensor[ei], en_hidden)
            
        de_input = torch.tensor([[SOS]], device=device)
        de_hidden = en_hidden
        
        predicted_indices = []
        # decoder loop
        for di in range(target_length):
            de_output, de_hidden = self.decoder(de_input, de_hidden)
            
            # Choose top word from output
            _, topi = de_output.topk(1)
            predicted_indices.append(topi.item())
            de_input = topi.squeeze().detach()
            
            loss += self.loss_fn(de_output, target_tensor[di].unsqueeze(0))

            if de_input.item() == EOS:
                break
        if predicted_indices == target_tensor.tolist():
            correct_prediction = True
        return loss, correct_prediction, predicted_indices
    
    def train_model(
            self,
            epochs,
            train_loader: CudaDataPrefetcher,
            val_loader: CudaDataPrefetcher,
            loss_fn=nn.NLLLoss(),
            optimizer=torch.optim.SGD,
            optimizer_args = [],
            optimizer_kwargs = {},
            print_epoch=1,
            header_epoch = 15,
            sched_factor = 0.1,
            sched_patience = 5,
            min_accuracy = 0.5,
            max_negative_diff_count = 6
        ):  
            scaler = GradScaler("cuda")
            en_optimizer = optimizer(self.parameters(), *optimizer_args, **optimizer_kwargs)
            de_optimizer = optimizer(self.parameters(), *optimizer_args, **optimizer_kwargs)
            en_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(en_optimizer, 'max', patience=sched_patience, factor=sched_factor)
            de_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(de_optimizer, 'max', patience=sched_patience, factor=sched_factor)
            training_time = 0
            self.train_loss_hist = torch.zeros(epochs)
            self.val_loss_hist = torch.zeros(epochs)
            self.accuracy_hist = torch.zeros(epochs)
            self.loss_fn = loss_fn
            
            cell_width = 20
            header_form_spec = f'^{cell_width}'
            
            epoch_inspection = {
                "Epoch": 0,
                "Epoch Time (s)": 0,
                "Training Loss": 0,
                "Validation Loss ": 0,
                "Validation Time": 0,
                "Accuracy (%)": 0,
                "Δ Accuracy (%)": 0,
                "Avg Inference Time": 0,
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
                
                begin_epoch = time.perf_counter()
                begin_train = time.perf_counter()
                
                
                total_train_loss = 0
                total_train_samples = 0
                self.train()
                self.encoder.train()
                self.decoder.train()
                for X_batch, Y_batch in train_loader:
                    batch_size = X_batch.size(0)
                    en_optimizer.zero_grad(set_to_none=True)
                    de_optimizer.zero_grad(set_to_none=True)
                    
                    with autocast("cuda"):
                        losses = [loss for loss, _, _ in [self.forward_pass(X_batch[i], Y_batch[i]) for i in range(batch_size)]]
                    
                    train_batch_loss = sum(losses)
                    scaler.scale(train_batch_loss).backward()
                    scaler.step(en_optimizer)
                    scaler.step(de_optimizer)
                    scaler.update()                   

                    total_train_loss += train_batch_loss.item() 
                    total_train_samples += batch_size
                training_time += time.perf_counter() - begin_train
                
                total_train_loss = total_train_loss / X_batch.size(1) / total_train_samples
                self.train_loss_hist[epoch] = total_train_loss
                
                del X_batch, Y_batch, train_batch_loss
                begin_val = time.perf_counter()
                num_correct = 0
                val_loss = 0
                total_val_samples = 0
                total_inference_time = 0
                self.eval()
                self.encoder.eval()
                self.decoder.eval()
                with torch.no_grad():
                    
                    for X_val_batch, Y_val_batch in val_loader:
                        batch_size = X_val_batch.size(0)
                        with autocast('cuda'):
                            val_batch_loss = 0
                            for i in range(batch_size):
                                begin_inf = time.perf_counter()
                                val_sample_loss, correct, _ = self.forward_pass(X_val_batch[i], Y_val_batch[i])
                                total_inference_time += (time.perf_counter() - begin_inf) 
                                val_batch_loss += val_sample_loss
                                num_correct += correct
                                total_val_samples += 1
                        val_loss += val_batch_loss.item()                        
                    
                avg_inference_time = total_inference_time / total_val_samples / X_val_batch.size(1)
                accuracy = num_correct/total_val_samples
                val_loss =val_loss / X_val_batch.size(1) / total_val_samples
                self.val_loss_hist[epoch] = val_loss                   
                self.accuracy_hist[epoch] = accuracy
                en_scheduler.step(accuracy)
                de_scheduler.step(accuracy)
                
                end_epoch = time.perf_counter()
                if print_epoch and (epoch % print_epoch == 0 or epoch == epochs - 1) :
                    mem = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved())/1024**3
                    if header_epoch and epoch % header_epoch == 0:
                        print(header_string)
                        print(divider_string)
                    epoch_duration = end_epoch - begin_epoch
                    overfit = 100 * (val_loss - total_train_loss) / total_train_loss
                    d_accuracy = 0 if max_accuracy == 0 else 100 * (accuracy - max_accuracy) / max_accuracy
                    if d_accuracy <= 0:
                        negative_acc_diff_count += 1
                    else:
                        negative_acc_diff_count = 0

                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                    
                    epoch_inspection['Epoch'] = f'{epoch}'
                    epoch_inspection['Epoch Time (s)'] = f'{epoch_duration:4f}'
                    epoch_inspection['Training Loss'] = f'{total_train_loss:8f}'
                    epoch_inspection['Validation Loss '] = f'{val_loss:8f}'
                    epoch_inspection['Avg Inference Time'] = f'{avg_inference_time:4e}'
                    epoch_inspection["Validation Time"] = f'{end_epoch - begin_val:4f}'
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

        
        