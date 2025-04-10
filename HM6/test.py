import torch
from jlib.vision_transformer import VisionTransformer, History
from jlib.data_utils import get_cifar_fetchers, get_cifar100

_, val_data = get_cifar100()


vit = VisionTransformer(
    image_size=32,
    patch_size=4,
    embed_dim=256,
    inner_dim=512,
    num_attn_heads=2,
    num_attn_layers=4,
    num_classes=100,
    cls_head_dims=[1024]
)
print(f"Parameter count: {vit.param_count:4e}")

train_fetcher, val_fetcher = get_cifar_fetchers(
    train_batch_size=256,
    val_batch_size=len(val_data),
    cpu_prefetch=16,
    gpu_prefetch=16,
    redownload=False
)
del val_data


hist: History = vit.train_model(
    epochs=150,
    train_fetcher=train_fetcher,
    num_train_batches=len(train_fetcher.data_iterable),
    val_fetcher=val_fetcher,
    num_val_batches=1,
    optimizer=torch.optim.AdamW,
    optimizer_kwargs={
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    },
    stop_on_plateau=False,
)

hist.save('models/vit1.json')
fig = hist.plot_training('ViT Model 1 Training')
fig.save('latex/images/vit1.png')


