from .classifier import Classifier, ConvParams
from torch import nn
import torch
device = 'cuda'


def get_conv_relu(cnv_params: ConvParams):
    return nn.Sequential(
        nn.Conv2d(**cnv_params.__dict__()),
        nn.ReLU()
    )


class VggBlock(nn.Module):
    def __init__(
        self,
        params: ConvParams,
        pool_kernel = 2,
        pool_stride = 2,
        repitions = 2,
    ):
        super().__init__()
        #param0 = ConvParams(*params.as_list())
        #param0.in_chan = params.in_chan
        #params.in_chan = params.out_chan
        convreulus = []
        for i in range(repitions):
            if i == 0:
                convreulus.append(get_conv_relu(params))
                params.in_chan = params.out_chan
            else:
                convreulus.append(get_conv_relu(params))
            
        self.computation = nn.Sequential(
            *convreulus,
            nn.MaxPool2d(pool_kernel, pool_stride),
        )
        self.pool = nn.MaxPool2d(pool_kernel, pool_stride)
    def forward(self, x):
        return self.computation(x)

class VggNet(Classifier):
    def __init__(
            self,
            in_chan,
            in_dim,
            num_classes,
            block_params: list = [],
            fc_params = [],
            dropout = 0
        ):
        super().__init__()
        self.sequential = nn.Sequential()
        
        for i in range(len(block_params)):
            if i == 0:
                block_params[i][1].in_chan = in_chan
            else:
                block_params[i][1].in_chan = block_params[i-1][1].out_chan
            self.sequential.add_module(name=f'block_{i}', module=VggBlock(params=block_params[i][1], repitions=block_params[i][0]))
            self.sequential.add_module(name = f"dropout_{i}", module=nn.Dropout(dropout))
        self.sequential.add_module(name='flatten', module=nn.Flatten())
        self.sequential.to(device)
        dummy_in = torch.randn(1, in_chan, *in_dim).to(device)
        dummy_out = self.sequential(dummy_in)
        fc_in = dummy_out.shape[1]
        
        for i in range(len(fc_params)):
            layer_in = 0
            if i == 0:
                layer_in = fc_in
            else:
                layer_in = fc_params[i-1]
            self.sequential.add_module(name=f'linear_{i}', module=nn.Sequential(
                nn.Linear(layer_in, fc_params[i]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.sequential.add_module(name = 'output', module=nn.Linear(fc_params[-1], num_classes))        
        self.sequential = self.sequential.to(device)