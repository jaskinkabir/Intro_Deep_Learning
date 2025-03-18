import torch
from torchtnt.utils.data import CudaDataPrefetcher
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import time

SOS = 1
EOS = 2
PAD = 0

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1 : '<SOS>', 2: "<EOS>"}
        self.n_words = 3
        self.max_sentence_length = 0
        
    def add_sentence(self, sentence):
        split = sentence.split(' ')
        if len(split) > self.max_sentence_length:
            self.max_sentence_length = len(split)
        for word in split:
            self.add_word(word)
            
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Load data (English to French)
class EnglishToFrench(Dataset):
    
    def __init__(self, en2fr, max_length, gpu=False):
        self.sequences = []
        self.targets = []
        self.en = Language('English')
        self.fr = Language('French')
        
        for english, french in en2fr:
            self.en.add_sentence(english)
            self.fr.add_sentence(french)
        
        self.max_length = max(max_length, self.en.max_sentence_length, self.fr.max_sentence_length) + 2

        for english, french in en2fr:
            en_seq = self.sentence_to_sequence(english, self.en)
            self.sequences.append(en_seq)
            
            fr_seq = self.sentence_to_sequence(french, self.fr)
            self.targets.append(fr_seq)
            
        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        if gpu:
            self.sequences = self.sequences.cuda()
            self.targets = self.targets.cuda()
        
    def sentence_to_sequence(self, sentence, language):
        seq = []
        for word in sentence.split():
            seq.append(language.word2index[word])
        
        if len(seq) < self.max_length-1:
            seq += [PAD] * (self.max_length - len(seq) - 1)
        
        seq.append(EOS)
        return seq
            
            
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def gen_data_loader(
    data,
    batch_size,
    workers = 6,
    cpu_prefetch = 10,
    gpu_prefetch = 10,
    clear = False
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
    start = time.perf_counter()
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

def get_enfr_loader(
    enfr,
    train_batch_size = 8192,
    val_batch_size = 8192,
    workers = 6,
    cpu_prefetch = None,
    gpu_prefetch = None,
    clear = False
):
    """
    returns {
        'dataset' : EnglishToFrench(enfr),
        'train_loader : CudaDataPrefetcher
        'val_loader' : CudaDataPrefetcher
    }
    """
    data = EnglishToFrench(enfr)
    val_workers = workers // 3
    train_workers = workers * 2 // 3
    
    if cpu_prefetch is None or gpu_prefetch is None:    
        max_gpu_mem = 6000
        max_train_mem = max_gpu_mem // 2
        max_val_mem = max_train_mem  
        
        train_batch_space = get_batch_space(data, train_batch_size)
        train_gpu_prefetch = max_train_mem // train_batch_space
        train_cpu_prefetch = train_gpu_prefetch // train_workers
        
        val_batch_space = get_batch_space(data, val_batch_size)
        val_gpu_prefetch = max_val_mem // val_batch_space
        val_cpu_prefetch = val_gpu_prefetch // val_workers
    print(f"Train GPU Prefetch: {train_gpu_prefetch}")
    print(f"Train CPU Prefetch: {train_cpu_prefetch}")
    print(f"Val GPU Prefetch: {val_gpu_prefetch}")
    print(f"Val CPU Prefetch: {val_cpu_prefetch}")
    
    
    print("Train Loader")
    train_loader = gen_data_loader(
        data,
        train_batch_size,
        train_workers,
        int(train_cpu_prefetch),
        int(train_gpu_prefetch),
        clear = clear
    )
    print("Val Loader")
    val_loader = gen_data_loader(
        data,
        val_batch_size,
        val_workers,
        int(val_cpu_prefetch),
        int(val_gpu_prefetch),
        clear = clear
    )
    
    return {
        'dataset' : data,
        'train_loader' : train_loader,
        'val_loader' : val_loader
    }
    
    