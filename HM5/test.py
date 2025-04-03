import torch
import numpy as np
from jlib.transformer_components import *

head = ClassifierHead(
    in_dim=10,
    out_dim=2,
    layer_dims=[5,6,7],
    dropout=0.1
)

print('done')