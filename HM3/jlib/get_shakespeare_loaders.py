import torch
import time
from torchtnt.utils.data import CudaDataPrefetcher
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
import gc

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

def gen_datasets(sequence_length, redownload=False):
    text = get_text(f'data/shakespeare.txt', redownload)
# Step 2: Prepare the dataset
    # Create a character mapping to integers
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    # Encode the text into integers
    encoded_text = [char_to_int[ch] for ch in text]

    # Create sequences and targets
    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length):
        seq = encoded_text[i:i+sequence_length]
        target = encoded_text[i+sequence_length]
        sequences.append(seq)
        targets.append(target)

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
        "chars" : chars,
        "char_to_int" : char_to_int,
        "int_to_char" : int_to_char,
        "train_dataset" : train_dataset,
        "val_dataset" : val_dataset,
    }

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
        shuffle=True
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
        'chars' : list[str] - Alphabet
        'char_to_int' : dict[str, int] - Character to integer mapping
        'int_to_char' : dict[int, str] - Integer to character mapping
        'train_dataset' : Dataset - Training dataset
        'val_dataset' : Dataset - Validation dataset
        'train_loader' : DataLoader - Training data loader
        'val_loader' : DataLoader - Validation data loader
    }
    """
    
    
    
    data = gen_datasets(sequence_length)
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