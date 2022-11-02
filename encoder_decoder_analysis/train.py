from autoencoder import *
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import pylab
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import pickle

z_dim = 64
batch_size = 1
num_epochs = 30
learning_rate = 3.0e-4
n = 6 #number of test sample
cuda = True
model = Autoencoder(z_dim)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)

if cuda:
    model.cuda()

class ParaDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):
        tensor_para = self.dataset[index]
        return tensor_para, []

with open('G://Defending-Against-Backdoors-with-Robust-Learning-Rate//final_training.pl', "rb") as fp:
    raw_dataset = pickle.load(fp)

para_dataset = ParaDataset(raw_dataset)
train_loader = DataLoader(para_dataset, batch_size = batch_size, shuffle=True)
losses = np.zeros(num_epochs)

print('len of dataset is')
print(len(train_loader))

for epoch in range(num_epochs):
    i = 0
    total_loss = 0
    for para_tensor,_ in train_loader:
        x = para_tensor.view(para_tensor.shape[0], -1)
        if cuda:
            x = Variable(x).cuda()
        else:
            x = Variable(x)
        decoder_result,_ = model(x)

        # 出力画像（再構成画像）と入力画像の間でlossを計算
        loss = mse_loss(decoder_result, x)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss of epoch {} is'.format(epoch))
    print(total_loss/len(train_loader))
    if epoch % 10 == 0:

        #print('current decoder output is')
        decoder_list = list(decoder_result.cpu().detach().numpy())
        #print(list(decoder_list[0]))

        #print('current input is')
        input_list = list(x.cpu().detach().numpy())
        #print(list(input_list[0]))
