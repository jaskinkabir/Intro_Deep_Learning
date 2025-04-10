import torch
import time
from jlib.vision_transformer import VisionTransformer, History
from jlib.data_utils import get_cifar_fetchers, get_cifar100, GpuCIFAR
from torch.utils.data import DataLoader

vit = VisionTransformer(
    image_size=32,
    patch_size=4,
    embed_dim=256,
    inner_dim=512,
    num_attn_heads=4,
    num_attn_layers=8,
    num_classes=100,
    cls_head_dims=[1024,128]
)
print(f"Parameter count: {vit.param_count:4e}")

# train_fetcher, val_fetcher = get_cifar_fetchers(
#     train_batch_size=512,
#     val_batch_size=len(val_data),
#     cpu_prefetch=16,
#     gpu_prefetch=16,
#     redownload=False
# )
# del val_data

train_data, val_data = get_cifar100()

print("loading train")
start = time.perf_counter()
train_data = GpuCIFAR(train_data)
train_load_time = time.perf_counter() - start
print(f"Train data load time: {train_load_time:2f} s")
print("loading val")
start = time.perf_counter()
val_data = GpuCIFAR(val_data)
val_load_time = time.perf_counter() - start
print(f"Val data load time: {val_load_time:2f} s")


print("init train loader")
train_fetcher = DataLoader(
    train_data,
    batch_size=512
)
print("init val loader")
val_fetcher = DataLoader(
    val_data,
    batch_size=len(val_data)
)

hist: History = vit.train_model(
    epochs=150,
    train_fetcher=train_fetcher,
    num_train_batches=len(train_fetcher),
    val_fetcher=val_fetcher,
    num_val_batches=1,
    optimizer=torch.optim.AdamW,
    optimizer_kwargs={
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    },
    stop_on_plateau=True,
    min_accuracy=1,
    max_negative_diff_count=10
)

hist.save('models/vit1.json')
fig = hist.plot_training('ViT Model 1 Training')
fig.save('latex/images/vit1.png')


