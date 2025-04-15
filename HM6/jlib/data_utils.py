import torch
import time
import torchvision
import torchvision.transforms as transforms
from torchtnt.utils.data import CudaDataPrefetcher
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
from transformers import AutoImageProcessor
import gc


image_size = 32

def get_cifar100(path='./data', redownload=False, swin=None, resnet=False):
    if swin is not None:
        image_size = 224
        processor = AutoImageProcessor.from_pretrained(swin)
        mean = processor.image_mean
        std = processor.image_std
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif resnet:
        image_size = 224
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms
    else:
        image_size = 32
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR100(root=path, train=True,
                                            download=redownload, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root=path, train=False,
                                            download=redownload, transform=transform)
    
    return train_dataset, test_dataset



def gen_data_loader(
    dataset,
    batch_size = 8192,
    workers = 6,
    cpu_prefetch = 10,
    gpu_prefetch = 10,
    clear=False,
    shuffle=True,
    device='cuda'
):
    device = torch.device(device)
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
        device=device
    )
    print(f"Fetcher init time: {time.perf_counter() - start:2f} s")
    return fetcher

def get_batch_space(dataset, batch_size):
    val_batch = DataLoader(dataset, batch_size=batch_size)
    X_batch = next(iter(val_batch))[0]
    res = X_batch.element_size() * X_batch.nelement() / 1024**2
    del val_batch, X_batch
    return res

def gen_fetchers(
    train_dataset,
    val_dataset,
    train_batch_size=None,
    val_batch_size=None,
    train_split=0.8,
    workers=30,
    max_gpu_mem= 30 * 1024**3,
    cpu_prefetch=None,
    gpu_prefetch=None,
    device='cuda',
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
    if val_batch_size is None:
        val_batch_size = len(val_dataset)
    if train_batch_size is None:
        train_batch_size = len(train_dataset)
    
    def split(x):
        return int(x * train_split), int(x * (1 - train_split))
    
    max_train_mem, max_val_mem = split(max_gpu_mem)
    train_workers, val_workers = split(workers)
    
    
    train_workers = workers * 2 // 3
    val_workers = workers // 3
    
    if cpu_prefetch is None or gpu_prefetch is None:
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
    else:
        train_cpu_prefetch, val_cpu_prefetch = split(cpu_prefetch)
        train_gpu_prefetch, val_gpu_prefetch = split(gpu_prefetch)
        
    
    print("Train Loader")
    train_loader = gen_data_loader(
        train_dataset,
        train_batch_size,
        train_workers,
        int(train_cpu_prefetch),
        int(train_gpu_prefetch),
        device=device
    )
    print("Val Loader")
    val_loader = gen_data_loader(
        val_dataset,
        val_batch_size,
        val_workers,
        int(val_cpu_prefetch),
        int(val_gpu_prefetch),
        device=device,
    )
    
    return train_loader, val_loader