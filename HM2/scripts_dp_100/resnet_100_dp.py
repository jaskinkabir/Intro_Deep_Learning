from helpers import *

cifar_100_train_loader, cifar_100_val_loader = get_cifar_loaders(
    is_cifar10=True,
    train_batch_size=256,
    val_batch_size=256,
    train_workers=6,
    train_cpu_prefetch=10,
    train_gpu_prefetch=10,
    val_workers=2,
    val_cpu_prefetch=2,
    val_gpu_prefetch=2,
)

torch.cuda.empty_cache()

resnet_100_dp = ResNet(
        in_chan = 3,
        in_dim = (32, 32),
        n_classes = 100,
        first_conv=ConvParams(
            kernel=3,
            out_chan=64,
            stride=1,
            padding='same',
        ),
        block_params=[
            ConvParams(
                kernel=3,
                out_chan=64,
                stride=1,
                padding='same',
            ),
            ConvParams(
                kernel=3,
                out_chan=128,
                stride=1,
                padding='same',
            ),
            ConvParams(
                kernel=3,
                out_chan=256,
                stride=1,
                padding='same',
            ),
            ConvParams(
                kernel=3,
                out_chan=512,
                stride=1,
                padding='same',
            ),
        ],
        fc_params=[
            2048,
            2048,
        ],
        dropout = 0.5,
)
resnet_100_dp.to('cuda')
print(f"Num Params: {sum(p.numel() for p in resnet_100_dp.parameters())}")
resnet_100_dp.train_model(
    epochs = 50,
    train_loader = cifar_100_train_loader,
    val_loader = cifar_100_val_loader,
    optimizer = torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-4},
    print_epoch=1,
    header_epoch=10,
    min_accuracy=0.5,
)
torch.save(resnet_100_dp, 'models/resnet_100_dp.pth')
resnet_100_dp.plot_training()
plt.savefig('figures/resnet_100_dp.png')