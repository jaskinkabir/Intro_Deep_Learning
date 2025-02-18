import os
#from helpers import *

from jlib.cifar_preprocessing import get_cifar_loaders, delete_deletables
from jlib.classifier import Classifier
from jlib.vggnet import VggNet, VggBlock, ConvParams
import torch
from torch import nn
import matplotlib.pyplot as plt
device = 'cuda'

"""
sudo fuser -v -k /usr/lib/wsl/drivers/nvhm.inf_amd64_5c197d2d97068bef/*
"""
    
architecture = {
    'in_chan': 3,
    'in_dim': (32, 32),
    'block_params': [
        VggBlock(
            params=ConvParams(kernel=3,out_chan=64),
            pool_kernel=2,
            pool_stride=2,
            repititions=2
        ),
        VggBlock(
            params=ConvParams(kernel=3,out_chan=128),
            pool_kernel=2,
            pool_stride=1,
            repititions=2
        ),
        VggBlock(
            params=ConvParams(kernel=3,out_chan=256),
            pool_kernel=2,
            pool_stride=1,
            repititions=4
        ),
        VggBlock(
            params=ConvParams(kernel=3,out_chan=512),
            pool_kernel=2,
            pool_stride=1,
            repititions=4
        ),
        VggBlock(
            params=ConvParams(kernel=3,out_chan=512),
            pool_kernel=2,
            pool_stride=1,
            repititions=4
        ),
    ],
    'fc_params': [
        4096,
        4096,
    ],
}
torch.cuda.empty_cache()
Vgg_100_dp = VggNet(
    num_classes=1000,
    dropout = 0.5,
    **architecture
).to(device)
print(f"Num Params: {sum(p.numel() for p in Vgg_100_dp.parameters()):2e}")