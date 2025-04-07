from .transformer_char_predictor import ClassifierHead, History, PositionalEncoding
import torch
import torch.nn as nn
from .get_enfr_loader import PAD, EOS
from torch.cuda.amp import autocast, GradScaler
from torchtnt.utils.data import CudaDataPrefetcher
import time
from torchprofile import profile_macs




class TransformerTranslator(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        d_model: int,
        d_inner,
        n_layers: int,
        n_heads: int,
        head_layers: list[int],
        dropout: float = 0.1,
        max_seq_len: int = 512,
        device = 'cuda'
    ):
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(output_vocab_size, d_model)
        
        self.postional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, device=device)     
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_inner, dropout, batch_first=True),
            num_layers=n_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, d_inner, dropout, batch_first=True),
            num_layers=n_layers
        )
        
        self.head = ClassifierHead(d_model, output_vocab_size, head_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        
        self.param_count = sum(p.numel() for p in self.parameters())
    def forward(self, x, y: torch.Tensor):
        x_padding_mask = (x == PAD)
        y_padding_mask = (y == PAD)
        y_lookahead_mask = torch.triu(torch.ones((y.size(1), y.size(1))), diagonal=1).bool().to(self.device)
        
        x_embed = self.self.dropout(self.postional_encoding(self.encoder_embedding(x)))
        y_embed = self.self.dropout(self.postional_encoding(self.decoder_embedding(y)))
        
        encoder_out = self.encoder(x_embed, src_key_padding_mask=x_padding_mask)
        decoder_out = self.decoder(y_embed, encoder_out, tgt_key_padding_mask=y_padding_mask, memory_key_padding_mask=x_padding_mask, tgt_mask=y_lookahead_mask)
        output = self.head(decoder_out)
        return output
    
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
        num_correct_sequences = torch.zeros(1, device=self.device)
        total_tokens = torch.zeros(1, device=self.device)
        epoch_val_loss = torch.zeros(1, device=self.device)
        self.eval()
        with torch.no_grad():
            for X_val_batch, Y_val_batch in fetcher:
                with autocast('cuda'):
                    outputs = self.forward(X_val_batch)
                    val_batch_loss = self.loss_fn(outputs.transpose(1, 2), Y_val_batch)
                    
                    pred_mask = (Y_val_batch != PAD)
                    predicted_tokens = outputs.argmax(dim=-1)
                    correct_tokens = torch.eq(predicted_tokens, Y_val_batch).masked_select(pred_mask)
                    
                    num_correct_tokens += correct_tokens.sum()
                    
                    correct_sequences = correct_tokens.all(dim=1)
                    num_correct_sequences += correct_sequences.sum()

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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        self.train_loss_hist = torch.zeros(epochs)
        self.val_loss_hist = torch.zeros(epochs)
        self.accuracy_hist = torch.zeros(epochs)
        d_accuracy = torch.zeros(1)
        
        test_input = torch.randint(0, self.alphabet_size, (1, 1), device=self.device)
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
            epoch_train_loss = self.train_step(train_fetcher)
            #print('val step')
            epoch_val_loss, accuracy = self.eval_step(val_fetcher)
            
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
        
        