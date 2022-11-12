import torch
from torchvision import datasets, transforms
from utils import *

a = torch.rand([3,4])

b = torch.rand([2,3,4])
print((a + b).shape)




