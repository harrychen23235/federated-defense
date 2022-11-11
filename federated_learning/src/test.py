import torch
from torchvision import datasets, transforms
from utils import *
class args():
    def __init__(self) -> None:
        self.data = 'fedemnist'

train_dataset, test_dataset = get_datasets(args())

print(train_dataset.data[0].shape)

print(train_dataset.targets[0])




