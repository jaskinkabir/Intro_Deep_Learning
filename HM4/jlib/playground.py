import torch.nn.functional as F
import torch

t = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
print(F.one_hot(t))