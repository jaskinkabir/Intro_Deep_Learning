import torch
import time
from jlib.vision_transformer import VisionTransformer, History
from jlib.data_utils import gen_fetchers, get_cifar100
from torch.utils.data import DataLoader
from matplotlib.figure import Figure

# Add this at the beginning of your test.py file
import signal
import sys


def handle_ctrl_z(signum, frame):
    print("\nCaught Ctrl+Z, terminating process and freeing GPU memory...")
    # Clean up CUDA memory before exiting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)  # Exit the process completely
    
#print(f"Parameter count: {vit.param_count:4e}")

def train_model(model: VisionTransformer, model_name, chart_title, epochs, device='cuda', lr=5e-5, swin=None):
        
    signal.signal(signal.SIGTSTP, handle_ctrl_z)
        
    # Register the custom handler for SIGTSTP (Ctrl+Z)


    # train_fetcher, val_fetcher = get_cifar_fetchers(
    #     train_batch_size=512,
    #     val_batch_size=len(val_data),
    #     cpu_prefetch=16,
    #     gpu_prefetch=16,
    #     redownload=False
    # )
    # del val_data

    train_data, val_data = get_cifar100(swin=swin)
    train_fetcher, val_fetcher = gen_fetchers(
        train_data,
        val_data,
        train_batch_size=64,
        workers=35,
        cpu_prefetch=30,
        gpu_prefetch=30,
        device=device,
    )

    hist: History = model.train_model(
        epochs=epochs,
        train_fetcher=train_fetcher,
        num_train_batches=len(train_fetcher.data_iterable),
        val_fetcher=val_fetcher,
        num_val_batches=1,
        loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={
            'lr': lr,
            'weight_decay': 5e-2,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        },
        stop_on_plateau=True,
        min_accuracy=1,
        max_negative_diff_count=10
    )

    hist.save(f'models/{model_name}-64.json')
    fig: Figure = hist.plot_training(chart_title)
    fig.savefig(f'latex/images/{model_name}-64.png')


