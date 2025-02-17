import os
from helpers import *
"""
sudo fuser -v -k /usr/lib/wsl/drivers/nvhm.inf_amd64_5c197d2d97068bef/*
"""
    


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

def train_and_plot(model, train, val, title):
    print(f"Model: {title}")
    print("-"*50)
    print(f"Num Params: {sum(p.numel() for p in model.parameters())}")
    model.train_model(
        epochs=100,
        train_loader=train,
        val_loader=val,
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
    torch.save(model, f'models/{title}.pth')
    model.plot_training()
    plt.savefig(f'figures/{title}.png')

architecture = {
    "in_chan" : 3,
    "in_dim" : (32, 32),
    
    "block_params" : [
        ConvParams(kernel=5, out_chan=128, stride=1, padding='same'),
        ConvParams(kernel=5, out_chan=256, stride=1, padding='same'),
    ],
    "cnv_params" : [
        ConvParams(kernel=3, out_chan=256, stride=1, padding='same'),
        ConvParams(kernel=3, out_chan=256, stride=1, padding='same'),
        ConvParams(kernel=3, out_chan=256, stride=1, padding='same'),
    ],
    "fc_layers" : [2048, 2048],
}

torch.cuda.empty_cache()
alex_100_dp = AlexNet(
    num_classes=100,
    dropout = 0.5,
    **architecture
).to(device)

train_and_plot(alex_100_dp, cifar_100_train_loader, cifar_100_val_loader, "alex_100_dp")
del alex_100_dp

torch.cuda.empty_cache()
alex_100_ndp = AlexNet(
    num_classes=100,
    dropout = 0,
    **architecture
).to(device)

train_and_plot(alex_100_ndp, cifar_100_train_loader, cifar_100_val_loader, "alex_100_ndp")
del alex_100_ndp

del cifar_100_train_loader, cifar_100_val_loader

cifar_10_train_loader, cifar_10_val_loader = get_cifar_loaders(
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
alex_10_dp = AlexNet(
    num_classes=10,
    dropout = 0.5,
    **architecture
).to(device)

train_and_plot(alex_10_dp, cifar_10_train_loader, cifar_10_val_loader, "alex_10_dp")
del alex_10_dp

torch.cuda.empty_cache()
alex_10_ndp = AlexNet(
    num_classes=10,
    dropout = 0,
    **architecture
).to(device)

train_and_plot(alex_10_ndp, cifar_10_train_loader, cifar_10_val_loader, "alex_10_ndp")
del alex_10_ndp
