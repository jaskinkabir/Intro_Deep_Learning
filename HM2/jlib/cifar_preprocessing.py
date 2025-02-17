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
    )
    
    X_batch = next(iter(loader))[0]
    
    print(f"Batch Size: {X_batch.element_size() * X_batch.nelement() / 1024**2} MiB")
    print(f"Data Size: {X_batch.element_size() * data.__len__() * X_batch.nelement() / 1024**3} GiB")
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
    train_workers,
    train_cpu_prefetch,
    train_gpu_prefetch,
    val_workers,
    val_cpu_prefetch,
    val_gpu_prefetch,
    recompute=False,
    redownload=False,
    data_path='./data',
):
    cifar_train, cifar_val = get_cifar(is_cifar10, recompute, redownload, data_path)
    train_loader = gen_data_loader(
        cifar_train,
        train_batch_size,
        train_workers-val_workers,
        train_cpu_prefetch-val_gpu_prefetch,
        train_gpu_prefetch-val_cpu_prefetch
    )
    val_loader = gen_data_loader(
        cifar_val,
        val_batch_size,
        val_workers,
        val_cpu_prefetch,
        val_gpu_prefetch
    )
    return train_loader, val_loader