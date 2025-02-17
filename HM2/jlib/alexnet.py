from .classifier import Classifier, ConvParams
from torch import nn
import torch
device = 'cuda'

class AlexBlock(nn.Module):
    def __init__(
        self,
        params: ConvParams,
        pool_kernel = 2,
        pool_stride = 1,
    ):
        super().__init__()
        self.computation = nn.Sequential(
            # With batchnorm, bias is unnecessary
            nn.Conv2d(**params.__dict__(), bias=False),
            nn.BatchNorm2d(params.out_chan),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel, pool_stride)
        )
    def forward(self, x):
        return self.computation(x)
    

class AlexNet(Classifier):
        
    def __init__(
        self,
        in_chan,
        in_dim,
        num_classes,
        block_params: list = [],
        cnv_params =[],
        fc_layers = [],
        dropout = 0.5,
    ):
        super().__init__()
        self.cnv_layers = cnv_params
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.input_dim = in_chan
        self.num_classes = num_classes
        
        block_params[0].in_chan = in_chan
        self.sequential = nn.Sequential(AlexBlock(block_params[0]))
        for i in range(1, len(block_params)):
            block_params[i].in_chan = block_params[i-1].out_chan
            self.sequential.add_module(module=AlexBlock(block_params[i]), name=f'block_{i}')
        for i in range(len(cnv_params)):
            if i == 0:
                cnv_params[i].in_chan = block_params[-1].out_chan
            else:
                cnv_params[i].in_chan = cnv_params[i-1].out_chan
            self.sequential.add_module(module=nn.Sequential(
                nn.Conv2d(**cnv_params[i].__dict__()).to(device),
                nn.ReLU()
            ), name = f"conv_{i}")
        
        self.sequential.add_module(module=nn.Sequential(
            nn.MaxPool2d(3, 2),
            nn.Dropout2d(dropout),
            nn.Flatten() 
        ), name = 'flatten')
        self.sequential = self.sequential.to(device)
        
        dummy_in = torch.randn(1, in_chan, *in_dim).to(device)  # Add batch dimension
        dummy_out = self.sequential(dummy_in)
        fc_in = dummy_out.shape[1]
        self.sequential.add_module(name='linear_0', module=nn.Linear(fc_in, fc_layers[0]))
        self.sequential.add_module(name='relu_0', module=nn.ReLU())
        self.sequential.add_module(name='dropout_0', module=nn.Dropout(dropout))
        for i in range(1, len(fc_layers)):
            self.sequential.add_module(name=f'linear_{i}', module=nn.Sequential(
                nn.Linear(fc_layers[i-1], fc_layers[i]),
                nn.ReLU()
            ))
        self.sequential.add_module(name = 'output', module=nn.Linear(fc_layers[-1], num_classes))        
        self.sequential = self.sequential.to(device)
        #dummy_out = self.sequential(dummy_in)
        #print(dummy_out.shape)
    def forward(self, x):
        return self.sequential(x)
        

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
        self.computation = nn.Sequential(
            *[get_conv_relu(**params.__dict__()) for _ in range(repitions)],
            nn.MaxPool2d(pool_kernel, pool_stride),
        )
        self.pool = nn.MaxPool2d(pool_kernel, pool_stride)
    def forward(self, x):
        return self.computation(x)