import torch
from torchvision import datasets, transforms
from utils import *
'''
class args():
    def __init__(self) -> None:
        self.data = 'fedemnist'

train_dataset, test_dataset = get_datasets(args())

print(train_dataset.data[0].shape)

for i in train_dataset.targets.cpu().detach().numpy():
    print(i)
'''

data, targets, users_index = torch.load('C://Users//harrychen23235//Desktop//report//security//federated-defense//data//FEMNIST//training.pt')      

print(sum(users_index))
print(targets.unique())
print(len(data))



