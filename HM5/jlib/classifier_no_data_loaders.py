from .classifier import *

device = 'cuda'

class ClassifierNoDataLoaders(Classifier):
    @classmethod
    def compare_results(cls, results1, results2):
        print('Comparing results:')
        
        for key, value in results1.items():
            if isinstance(value, Number): print(f"{key} : {100*(value - results2[key]) / value:2f} %") 
        
    def __init__(self):
        super().__init__()
        
    def train_model(
            self,
            epochs,
            x_train,
            y_train,
            x_val,
            y_val,
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
    
                total_train_samples = 0
                self.train()
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    Y_pred = self.forward(x_train)
                    train_loss = loss_fn(Y_pred, y_train) 
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()                   

                train_loss = train_loss.item()
                training_time += time.time() - start_time
                
                train_loss = train_loss
                self.train_loss_hist[epoch] = train_loss
                
                # del X_batch, Y_batch, loss, Y_pred
                val_start = time.time()
                val_correct = 0
                val_loss = 0
                self.eval()
                with torch.no_grad():
                    Y_pred_eval = torch.zeros(len(y_val), device=device)
                    
                    with autocast('cuda'):
                        Y_pred_logits = self.forward(x_val)
                        val_loss = loss_fn(Y_pred_logits, y_val)
                    val_loss = val_loss.item()
                    
                    Y_pred_eval = Y_pred_logits.argmax(dim=1)
                    
                    val_correct += (Y_pred_eval == y_val).sum().item()
                    
                val_time = time.time() - val_start
                    
                accuracy = val_correct/len(y_val)
                self.val_loss_hist[epoch] = val_loss                   
                self.accuracy_hist[epoch] = accuracy
                # del X_val_batch, Y_val_batch, Y_pred_logits, Y_pred
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
                    #break
                    pass

            print(f'\nTraining Time: {training_time} seconds\n')
            self.last_pred = torch.tensor(Y_pred_eval)
            self.last_val = y_val