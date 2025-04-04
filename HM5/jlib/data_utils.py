import torch
import time
from torchtnt.utils.data import CudaDataPrefetcher
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
import gc


class Alphabet():
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        
        self.char_to_int = {}
        self.int_to_char = {}
        for i, ch in enumerate(self.chars):
            self.char_to_int[ch] = i
            self.int_to_char[i] = ch        
        
    def __len__(self):
        return len(self.chars)
    
    def encode(self, text):
        return [self.char_to_int[ch] for ch in text]
    def decode(self, encoded_text):
        return ''.join([self.int_to_char[i] for i in encoded_text])
        

class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

def get_text(path, redownload=True):
# Step 1: Download the dataset
    if redownload:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        text = response.text  # This is the entire text data
        with open(path, 'w+') as f:
            f.write(text)
    else:
        with open(path, 'r') as f:
            text = f.read()
    return text

def gen_datasets(text, sequence_length):    
    """
    Generates datasets for character-level text modeling tasks.
    Args:
        sequence_length (int): Length of the sequences to be generated.
        text (str): The input text data.
    Returns:
        dict: A dictionary containing the following keys:
            - 'alphabet': An Alphabet object containing character mappings.
            - 'train_dataset': A PyTorch Dataset object for training data.
            - 'val_dataset': A PyTorch Dataset object for validation data.
    """
    
    
    alphabet = Alphabet(text)
    encoded_text = alphabet.encode(text)

    # Create sequences and targets
    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length,):
        sequence = encoded_text[i:i + sequence_length]
        label_sequence = encoded_text[i+1:i + sequence_length + 1]
        sequences.append(sequence)
        targets.append(label_sequence) 

    # Convert lists to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)


    # Instantiate the dataset
    dataset = CharDataset(sequences, targets)

    # Step 4: Create data loaders
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return {
        'alphabet' : alphabet,
        "train_dataset" : train_dataset,
        "val_dataset" : val_dataset,
    }

def gen_data_loader(
    dataset,
    batch_size = 8192,
    workers = 6,
    cpu_prefetch = 10,
    gpu_prefetch = 10,
    clear=False,
    shuffle=True
):
    start = time.perf_counter()
    if clear:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        gc.collect()

    print('Begin init data loader')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        prefetch_factor=cpu_prefetch,
        pin_memory=True,
        shuffle=shuffle
    )
    
    X_batch = next(iter(loader))[0]
    batch_space = X_batch.element_size() * X_batch.nelement() / 1024**2
    
    print(f"Batch Size: {batch_space} MiB")
    print(f"Data Loader init time: {time.perf_counter() - start:2f} s")
    print("Begin init fetcher")
    fetcher = CudaDataPrefetcher(
        data_iterable=loader,
        num_prefetch_batches=gpu_prefetch,
        device=torch.device('cuda')
    )
    print(f"Fetcher init time: {time.perf_counter() - start:2f} s")
    return fetcher

def get_batch_space(dataset, batch_size):
    val_batch = DataLoader(dataset, batch_size=batch_size)
    X_batch = next(iter(val_batch))[0]
    res = X_batch.element_size() * X_batch.nelement() / 1024**2
    del val_batch, X_batch
    return res

def get_shakespeare_loaders(
    train_batch_size,
    val_batch_size,
    sequence_length,
    redownload=False,
    workers=15
):
    """
    Returns a dictionary containing the following
    {
        'train_loader': DataLoader for training data,
        'val_loader': DataLoader for validation data,
        'train_dataset': Dataset for training data,
        'val_dataset': Dataset for validation data,
        'alphabet': Alphabet object containing character mappings
    }
    """
    
    
    text = get_text("shakespeare.txt", redownload=redownload)
    data = gen_datasets(sequence_length, text)
    train_dataset = data['train_dataset']
    val_dataset = data['val_dataset']
    
    
    
    max_gpu_mem = 6000
    max_train_mem = max_gpu_mem * 2 // 3
    max_val_mem = max_gpu_mem // 3
    
    train_workers = workers * 2 // 3
    val_workers = workers // 3
    
    
    train_batch_space = get_batch_space(train_dataset, train_batch_size)
    train_gpu_prefetch = max_train_mem // train_batch_space
    train_cpu_prefetch = train_gpu_prefetch // train_workers
    
    val_batch_space = get_batch_space(val_dataset, val_batch_size)
    val_gpu_prefetch = max_val_mem // val_batch_space
    val_cpu_prefetch = val_gpu_prefetch // val_workers
    
    print(f"Train GPU Prefetch: {train_gpu_prefetch}")
    print(f"Train CPU Prefetch: {train_cpu_prefetch}")
    print(f"Val GPU Prefetch: {val_gpu_prefetch}")
    print(f"Val CPU Prefetch: {val_cpu_prefetch}")
    
    
    print("Train Loader")
    train_loader = gen_data_loader(
        train_dataset,
        train_batch_size,
        train_workers,
        int(train_cpu_prefetch),
        int(train_gpu_prefetch)
    )
    print("Val Loader")
    val_loader = gen_data_loader(
        val_dataset,
        val_batch_size,
        val_workers,
        int(val_cpu_prefetch),
        int(val_gpu_prefetch)
    )
    data["train_loader"] = train_loader
    data["val_loader"] = val_loader
    
    return data

# Now `train_loader` and `val_loader` are ready to be used in a training loop