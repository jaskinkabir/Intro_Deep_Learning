import os
from helpers import *


cifar_100_train_loader, cifar_100_val_loader = get_cifar_loaders(
    is_cifar10=True,
    train_batch_size=512,
    val_batch_size=512,
    train_workers=6,
    train_cpu_prefetch=10,
    train_gpu_prefetch=10,
    val_workers=2,
    val_cpu_prefetch=2,
    val_gpu_prefetch=2,
)

torch.cuda.empty_cache()
alex_100_dp = AlexNet(
    in_chan=3,
    in_dim=(32, 32),
    num_classes=100,
    block_params = [
        ConvParams(kernel=5, out_chan=128, stride=1, padding='same'),
        ConvParams(kernel=5, out_chan=256, stride=1, padding='same'),
    ],
    cnv_params = [
        ConvParams(kernel=3, out_chan=256, stride=1, padding='same'),
        ConvParams(kernel=3, out_chan=256, stride=1, padding='same'),
        ConvParams(kernel=3, out_chan=384, stride=1, padding='same'),
    ],
    fc_layers=[2048, 2048],
    dropout = 0.5
).to(device)

print(f"Num Params: {sum(p.numel() for p in alex_100_dp.parameters())}")

alex_100_dp.train_model(
    epochs=100,
    train_loader=cifar_100_train_loader,
    val_loader=cifar_100_val_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    optimizer_args=[],
    optimizer_kwargs={'lr': 1e-4},
    print_epoch=1,
    header_epoch=10,
    sched_factor=0.1,
    sched_patience=5,
    min_accuracy = 0.5
)
torch.save(alex_100_dp, 'models/alex_100_dp.pth')
alex_100_dp.plot_training()
plt.savefig('figures/alex_100_dp.png')