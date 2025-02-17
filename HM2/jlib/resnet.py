from .classifier import Classifier, ConvParams
from torch import nn
import torch
device = 'cuda'

        
class ResBlock:
    def __init__(
        self,
        in_chan,
        out_chan,
        conv_params,
    ):
        if in_chan != out_chan:
            self.shortcut = nn.Conv2d(in_chan, out_chan, 1)
        else:
            self.shortcut = nn.Identity()
            
        cnv_dict_1 = conv_params.__dict__()
        cnv_dict_2 = conv_params.__dict__()
        cnv_dict_1['in_channels'] = in_chan
        cnv_dict_2['in_channels'] = out_chan
        
        self.conv1 = nn.Conv2d(**cnv_dict_1)
        self.conv2 = nn.Conv2d(**cnv_dict_2)
        self.bn = nn.BatchNorm2d(out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x + shortcut

class ResNet(Classifier):
    def __init__(
        self,
        in_chan,
        in_dim,
        num_classes,
        first_conv: ConvParams,
        block_params: list[ConvParams],
        fc_layers = [],
        dropout = 0
    ):
        super().__init__()
        first_conv.in_chan = in_chan
        conv0 = nn.Conv2d(**first_conv.__dict__())
        self.sequential = nn.Sequential(
            conv0,
            nn.BatchNorm2d(first_conv.out_chan),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        for i, conv in enumerate(block_params):
            if i == 0:
                conv.in_chan = first_conv.out_chan
            else:
                conv.in_chan = block_params[i-1].out_chan
            self.sequential.add_module(name=f'res_block_{i}', module=ResBlock(conv.in_chan, conv.out_chan, conv))
            self.sequential.add_module(name = f"dropout_{i}", module=nn.Dropout(dropout))
        dummy_in = torch.randn(1, in_chan, *in_dim).to(device)
        dummy_out = self.sequential(dummy_in)
        fc_in = dummy_out.shape[1]
        self.sequential.add_module(name = 'fc_pool', module=nn.AdaptiveAvgPool2d((1,1)))
        self.sequential.add_module(name='flatten', module=nn.Flatten())
        for i in range(len(fc_layers)):
            layer_in = 0
            if i == 0:
                layer_in = fc_in
            else:
                layer_in = fc_layers[i-1]
            self.sequential.add_module(name=f'linear_{i}', module=nn.Sequential(
                nn.Linear(layer_in, fc_layers[i]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.sequential.add_module(name = 'output', module=nn.Linear(fc_layers[-1], num_classes))