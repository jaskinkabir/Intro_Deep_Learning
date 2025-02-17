from helpers import *

cifar_10_train_loader, cifar_10_val_loader = get_cifar_loaders(
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

vgg_10_dp = VggNet(
        in_chan = 3,
        in_dim = (32, 32),
        n_classes = 10,
        block_params=[
            (2, ConvParams(
                kernel=3,
                out_chan=64,
                stride=1,
                padding='same',
            )),
            (2,
             ConvParams(
                 kernel=3,
                 out_chan=128,
                 stride=1,
                 padding='same',
             )),
            (2,
             ConvParams(
                 kernel=3,
                 out_chan=256,
                 stride=1,
                 padding='same',
             )),
            (3,
             ConvParams(
                 kernel=3,
                 out_chan=512,
                 stride=1,
                 padding='same',
             )),
            (3,
             ConvParams(
                 kernel=3,
                 out_chan=512,
                 stride=1,
                 padding='same',
             )),
        ],
        fc_params=[
            4096,
            4096,
        ],
        dropout=0.5,
)
vgg_10_dp.to('cuda')
print(f"Num Params: {sum(p.numel() for p in vgg_10_dp.parameters())}")
vgg_10_dp.train_model(
    epochs = 50,
    train_loader = cifar_10_train_loader,
    val_loader = cifar_10_val_loader,
    optimizer = torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-4},
    print_epoch=1,
    header_epoch=10,
    min_accuracy=0.5,
)
torch.save(vgg_10_dp, 'models/vgg_10_dp.pth')
vgg_10_dp.plot_training()
plt.savefig('figures/vgg_10_dp.png')