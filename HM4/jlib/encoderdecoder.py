import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torchtnt.utils.data import CudaDataPrefetcher
import time
import matplotlib.pyplot as plt
from .get_enfr_loader import SOS, EOS
import random


device = 'cuda'



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(x)#.view(1, 1, -1)
        embedded = self.dropout(embedded)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden
    
    def initHidden(self):
        # Initializes hidden state
        dummy_in = torch.zeros(1, 1, self.hidden_size, device=device)
        _, hidden = self.rnn(dummy_in)
        return torch.zeros_like(hidden, device=device)
class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout, n_layers=1, max_sentence_length = 12, teacher_forcing_ratio = 0):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.max_sentence_length = max_sentence_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.embed = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor, target_tensor: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS)
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(batch_size, self.max_sentence_length, self.output_size, device=device)
        
        for i in range(self.max_sentence_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, i] = decoder_output.squeeze()
            
            
            
            if target_tensor is not None and random.random() < self.teacher_forcing_ratio:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _ , topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        return decoder_outputs, decoder_hidden, None
    
    def forward_step(self, input_tensor: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(input_tensor)
        output = self.dropout(embedded)
        output, hidden = self.rnn(output, hidden)
        output: torch.Tensor = self.out(output)
        return output, hidden
        
    def initHidden(self):
        # Initializes hidden state
        dummy_in = torch.zeros(1, 1, self.hidden_size, device=device)
        _, hidden = self.rnn(dummy_in)
        return torch.zeros_like(hidden, device=device)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query_matrix = nn.Linear(hidden_size, hidden_size)
        self.key_matrix = nn.Linear(hidden_size, hidden_size)
        self.value_matrix = nn.Linear(hidden_size, 1)
    def forward(self, query, keys):
        scores = self.value_matrix(torch.tanh(self.query_matrix(query) + self.key_matrix(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights
        

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout, n_layers=1, max_sentence_length = 12, teacher_forcing_ratio = 0):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.max_sentence_length = max_sentence_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.embed = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.GRU(2*hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor, target_tensor: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS)
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(batch_size, self.max_sentence_length, self.output_size, device=device)
        attentions = torch.zeros(batch_size, self.max_sentence_length, encoder_outputs.size(1), device=device)
        
        for i in range(self.max_sentence_length):
            decoder_output, decoder_hidden, attention_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[:, i] = decoder_output.squeeze()
            attentions[:, i] = attention_weights.squeeze()
            
            if target_tensor is not None and random.random() < self.teacher_forcing_ratio:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _ , topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        return decoder_outputs, decoder_hidden, attentions
    
    def forward_step(self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embed(input)
        embedded = self.dropout(embedded)
        query = hidden[-1].unsqueeze(1)  # Use the last layer's hidden state
        context, attention_weights = self.attention(query, encoder_outputs)
        gru_in = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(gru_in, hidden)
        output = self.out(output)
        return output, hidden, attention_weights

class Translator(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        dropout,
        n_layers=1,
        teacher_forcing_ratio=0,
        max_sentence_length=14,
        device='cuda',
        decoder = DecoderRNN,
        decoder_kwargs = dict()
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.encoder: EncoderRNN = EncoderRNN(input_size, hidden_size, dropout, n_layers).to(device)
        self.decoder: DecoderRNN = decoder(output_size, hidden_size, dropout, n_layers, max_sentence_length, teacher_forcing_ratio, **decoder_kwargs).to(device)
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
    
    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor | None =None):
        encoder_outputs, encoder_hidden = self.encoder.forward(input_tensor)
        decoder_outputs, _, _ = self.decoder.forward(encoder_outputs, encoder_hidden, target_tensor)
        return decoder_outputs
    
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
            scaler = GradScaler("cuda")
            self.en_optimizer = optimizer(self.encoder.parameters(), *optimizer_args, **optimizer_kwargs)
            self.de_optimizer = optimizer(self.decoder.parameters(), *optimizer_args, **optimizer_kwargs)
            self.en_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.en_optimizer, 'max', patience=sched_patience, factor=sched_factor)
            self.de_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.de_optimizer, 'max', patience=sched_patience, factor=sched_factor)
            self.loss_fn = loss_fn
            self.train_loss_hist = torch.zeros(epochs)
            self.val_loss_hist = torch.zeros(epochs)
            self.accuracy_hist = torch.zeros(epochs)
            
            training_time = 0
            
            cell_width = 20
            header_form_spec = f'^{cell_width}'
            
            epoch_inspection = {
                "Epoch": 0,
                "Epoch Time (s)": 0,
                "Training Loss": 0,
                "Validation Loss ": 0,
                "Validation Time": 0,
                "Sentence Accuracy": 0,
                "Δ Accuracy (%)": 0,
                "Avg Inference Time": 0,
                "Token Accuracy": 0
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
            
            forcing_ratio_interval = self.teacher_forcing_ratio * 10 / epochs
            
            
            for epoch in range(epochs):
                
                begin_epoch = time.perf_counter()
                begin_train = time.perf_counter()
                
                if epoch % 10 == 0:
                    self.decoder.teacher_forcing_ratio = max(0, self.teacher_forcing_ratio - forcing_ratio_interval)
                    
                
                epoch_train_loss = 0
                self.train()
                self.encoder.train()
                self.decoder.train()
                for X_batch, Y_batch in train_loader:
                    self.en_optimizer.zero_grad(set_to_none=True)
                    self.de_optimizer.zero_grad(set_to_none=True)
                    
                    with autocast("cuda"):
                        outputs = self.forward(X_batch, Y_batch)
                        train_batch_loss = self.loss_fn(
                            outputs.view(-1, outputs.size(-1)),
                            Y_batch.view(-1)
                        )
                    scaler.scale(train_batch_loss).backward()
                    scaler.step(self.en_optimizer)
                    scaler.step(self.de_optimizer)
                    scaler.update()                   

                    epoch_train_loss += train_batch_loss.item() 
                training_time += time.perf_counter() - begin_train
                
                epoch_train_loss = epoch_train_loss / len(train_loader)
                self.train_loss_hist[epoch] = epoch_train_loss
                
                del X_batch, Y_batch, train_batch_loss
                
                num_correct_tokens = 0
                total_tokens = 0
                num_correct_sentences = 0
                val_loss = 0
                total_inference_time = 0
                self.eval()
                self.encoder.eval()
                self.decoder.eval()
                begin_val = time.perf_counter()
                with torch.no_grad():
                    for X_val_batch, Y_val_batch in val_loader:
                        with autocast('cuda'):
                            outputs = self.forward(X_val_batch)
                            val_batch_loss = self.loss_fn(
                                outputs.view(-1, outputs.size(-1)),
                                Y_val_batch.view(-1)
                            )
                            
                            predicted_tokens = outputs.argmax(dim=-1)
                            correct_predictions = (predicted_tokens == Y_val_batch)
                            num_correct_tokens += correct_predictions.sum().item()
                            total_tokens += Y_val_batch.numel()
                            correct_sentences = correct_predictions.all(dim=-1)
                            num_correct_sentences += correct_sentences.sum().item()
                            

                        val_loss += val_batch_loss.item()                        
                total_inference_time = time.perf_counter() - begin_val    
                avg_inference_time = total_inference_time / len(val_loader)
                token_accuracy = num_correct_tokens / total_tokens
                sentence_accuracy = num_correct_sentences / len(val_loader.dataset)
                val_loss = val_loss / len(val_loader)
                
                accuracy = token_accuracy
                self.val_loss_hist[epoch] = val_loss                   
                self.accuracy_hist[epoch] = accuracy
                self.en_scheduler.step(accuracy)
                self.de_scheduler.step(accuracy)
                
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
                    epoch_inspection['Validation Loss '] = f'{val_loss:8f}'
                    epoch_inspection['Avg Inference Time'] = f'{avg_inference_time:4e}'
                    epoch_inspection["Validation Time"] = f'{end_epoch - begin_val:4f}'
                    epoch_inspection['Sentence Accuracy'] = f'{sentence_accuracy*100:4f}'
                    epoch_inspection['Δ Accuracy (%)'] = f'{d_accuracy:4f}'
                    epoch_inspection["Token Accuracy"] = f'{token_accuracy*100:4f}'
                    for value in epoch_inspection.values():
                        print(f"|{value:^{cell_width}}", end='')
                    print('|')
                    print(divider_string)
                    
                if stop_on_plateau and (accuracy > min_accuracy or negative_acc_diff_count > max_negative_diff_count):
                    break

            print(f'\nTraining Time: {training_time} seconds\n')

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