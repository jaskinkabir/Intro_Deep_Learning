import time
import torch
from torch.utils.data import DataLoader
from torchtnt.utils.data import CudaDataPrefetcher
import gc
from torchvision import datasets, transforms


cifar_10_deletables = []
cifar_100_deletables = []
data_path = './data'

def delete_deletables(deletables):
    for d in deletables:
        try :
            del d
        except:
            pass
    deletables.clear()
    gc.collect()
    

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

def get_cifar(
    is_cifar_10,
    recompute=False,
    redownload=False,
    data_path='./data'
):
    
    if is_cifar_10:
        delete_deletables(cifar_10_deletables)
    else:
        delete_deletables(cifar_100_deletables)
    title = 'cifar10' if is_cifar_10 else 'cifar100'
    cifar = datasets.CIFAR10 if is_cifar_10 else datasets.CIFAR100 
    
     
    if recompute:
        pre_cifar = cifar(data_path, train=True, download=redownload, transform=transforms.ToTensor())
        train_imgs = torch.stack([img for img, _ in pre_cifar], dim=3)
        mean = train_imgs.view(3, -1).mean(dim=1)
        std = train_imgs.view(3, -1).std(dim=1)
        torch.save(mean, f'data/mean_{title}.pt')
        torch.save(std, f'data/std_{title}.pt')
        del pre_cifar, train_imgs
    else:
        mean = torch.load(f'data/mean_{title}.pt')
        std = torch.load(f'data/std_{title}.pt')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar_train = cifar(data_path, train=True, download=redownload, transform=transform) 
    cifar_val = cifar(data_path, train=False, download=redownload, transform=transform)
    return cifar_train, cifar_val

def get_cifar_loaders(
    is_cifar10,
    train_batch_size,
    val_batch_size,
    recompute=False,
    redownload=False,
    data_path='./data',
    workers=15
):
    max_gpu_mem = 6000
    max_train_mem = max_gpu_mem * 2 // 3
    max_val_mem = max_gpu_mem // 3
    
    train_workers = workers * 2 // 3
    val_workers = workers // 3
    
    mem_per_img = 3 * 32 * 32 * 4 / 1024**2
    train_batch_space = train_batch_size * mem_per_img
    train_gpu_prefetch = max_train_mem // train_batch_space
    train_cpu_prefetch = train_gpu_prefetch // train_workers
    
    val_batch_space = val_batch_size * mem_per_img
    val_gpu_prefetch = max_val_mem // val_batch_space
    val_cpu_prefetch = val_gpu_prefetch // val_workers
    print(f"Train GPU Prefetch: {train_gpu_prefetch}")
    print(f"Train CPU Prefetch: {train_cpu_prefetch}")
    print(f"Val GPU Prefetch: {val_gpu_prefetch}")
    print(f"Val CPU Prefetch: {val_cpu_prefetch}")
    
    
    
    
    cifar_train, cifar_val = get_cifar(is_cifar10, recompute, redownload, data_path)
    print("Train Loader")
    train_loader = gen_data_loader(
        cifar_train,
        train_batch_size,
        train_workers,
        int(train_cpu_prefetch),
        int(train_gpu_prefetch)
    )
    print("Val Loader")
    val_loader = gen_data_loader(
        cifar_val,
        val_batch_size,
        val_workers,
        int(val_cpu_prefetch),
        int(val_gpu_prefetch)
    )
    return train_loader, val_loader
