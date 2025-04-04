import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtnt.utils.data import CudaDataPrefetcher
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import time

EOS = 1
PAD = 0

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>" : PAD, "<EOS>" : EOS}
        self.word2count = {}
        self.index2word = {0: "<PAD>", EOS : "<EOS>"}
        self.n_words = 3
        self.max_sentence_length = 0
    
    def set_max_sentence_length(self, max_length: int) -> None:
        self.max_sentence_length = max_length
    
    def add_sentence(self, sentence: str):
        sentence = sentence.lower()
        split = sentence.split(' ')
        if len(split) > self.max_sentence_length:
            self.max_sentence_length = len(split)
        for word in split:
            self.add_word(word)
            
    def add_word(self, word: str) -> None:
        word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def sentence_to_sequence(self, sentence: str) -> torch.Tensor:
        sentence = sentence.lower()
        seq = []
        for word in sentence.split():
            seq.append(self.word2index[word])
        seq.append(EOS)
        seq = torch.tensor(seq, dtype=torch.long)

        return seq
    
    def sequence_to_sentence(self, sequence: torch.Tensor) -> str:
        sentence = []
        for i in sequence:
            if i == EOS or i == PAD:
                break
            sentence.append(self.index2word[i.item()])
        return ' '.join(sentence)

def genLangs(en2fr):
    source_lang = Language('English')
    target_lang = Language('French')
    for english, french in en2fr:
        source_lang.add_sentence(english)
        target_lang.add_sentence(french)
    return source_lang, target_lang

# Load data (English to French)
class EnFrDataset(Dataset):
    
    def __init__(self, en2fr, max_length, gpu=False, source_lang=None, target_lang=None):
        self.sources = []
        self.targets = []
        
        if source_lang is not None and target_lang is not None:
            self.source_lang = source_lang
            self.target_lang = target_lang
        else:
            self.source_lang, self.target_lang = genLangs(en2fr)
            
        self.max_length = max(max_length, self.source_lang.max_sentence_length, self.target_lang.max_sentence_length)
        self.source_lang.set_max_sentence_length(self.max_length)
        self.target_lang.set_max_sentence_length(self.max_length)

        for english, french in en2fr:
            en_seq = self.source_lang.sentence_to_sequence(english)
            self.sources.append(en_seq)
            
            fr_seq = self.target_lang.sentence_to_sequence(french)
            self.targets.append(fr_seq)
            
        self.sources = pad_sequence(self.sources, batch_first=True, padding_value=PAD)
        self.targets = pad_sequence(self.targets, batch_first=True, padding_value=PAD)
        if gpu:
            self.sources = self.sources.cuda()
            self.targets = self.targets.cuda()
            
            
    def __len__(self):
        return len(self.sources)
    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx]
    def reverse(self):
        self.source_lang, self.target_lang = self.target_lang, self.source_lang
        self.sources, self.targets = self.targets, self.sources


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

def get_enfr_loaders(
    train_examples,
    val_examples,
    reverse=False,
    train_batch_size = 8192,
    val_batch_size = 8192,
    max_gpu_mem = 4000,
    workers = 6,
    cpu_prefetch = None,
    gpu_prefetch = None,
    clear = False
):
    """
    returns {
        'train_set' : EnFrDataset,
        'val_set' : EnFrDataset,
        'train_loader' : CudaDataPrefetcher,
        'val_loader' : CudaDataPrefetcher
    }
    
    """
    source_lang, target_lang = genLangs(train_examples)
    train_set = EnFrDataset(train_examples, 14, False, source_lang, target_lang)
    val_set = EnFrDataset(val_examples, 14, False, source_lang, target_lang)
    if reverse:
        train_set.reverse()
        val_set.reverse()
    
    val_workers = workers // 3
    train_workers = workers * 2 // 3
    
    if cpu_prefetch is None or gpu_prefetch is None:    
        max_train_mem = max_gpu_mem * 2 // 3
        max_val_mem = max_train_mem  // 3
        
        train_batch_space = get_batch_space(train_set, train_batch_size)
        train_gpu_prefetch = max_train_mem // train_batch_space
        train_cpu_prefetch = train_gpu_prefetch // train_workers
        
        val_batch_space = get_batch_space(val_set, val_batch_size)
        val_gpu_prefetch = max_val_mem // val_batch_space
        val_cpu_prefetch = val_gpu_prefetch // val_workers
    else:
        train_gpu_prefetch = gpu_prefetch * 2 // 3
        train_cpu_prefetch = cpu_prefetch * 2 // 3
        val_gpu_prefetch = gpu_prefetch // 3
        val_cpu_prefetch = cpu_prefetch // 3
    print(f"Train GPU Prefetch: {train_gpu_prefetch}")
    print(f"Train CPU Prefetch: {train_cpu_prefetch}")
    print(f"Val GPU Prefetch: {val_gpu_prefetch}")
    print(f"Val CPU Prefetch: {val_cpu_prefetch}")
    
    
    print("Train Loader")
    train_loader = gen_data_loader(
        train_set,
        train_batch_size,
        train_workers,
        int(train_cpu_prefetch),
        int(train_gpu_prefetch),
        clear = clear
    )
    print("Val Loader")
    val_loader = gen_data_loader(
        val_set,
        val_batch_size,
        val_workers,
        int(val_cpu_prefetch),
        int(val_gpu_prefetch),
        clear = clear
    )
    
    return {
        'train_set' : train_set,
        'val_set' : val_set,
        'train_loader' : train_loader,
        'val_loader' : val_loader
    }
    
    