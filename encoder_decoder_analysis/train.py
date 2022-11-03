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
import random
import matplotlib.pyplot as plt
z_dim = 1
batch_size = 1
num_epochs = 30
learning_rate = 1.0e-3
cuda = True
mse_loss = nn.MSELoss()
l1_loss = torch.nn.SmoothL1Loss()
class ParaDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):
        tensor_para = self.dataset[index]
        return tensor_para, []

with open('G://Defending-Against-Backdoors-with-Robust-Learning-Rate//parameter_training_benign.pl', "rb") as fp:
    raw_dataset = pickle.load(fp)

#with open('G://Defending-Against-Backdoors-with-Robust-Learning-Rate//parameter_corrupt_mali.pl', "rb") as fp:
    #corrupt_dataset = pickle.load(fp)

num_count = 0
turn_num = 1

for turn in range(turn_num):
    print('len of training dataset is')
    print(len(raw_dataset))
    #print('len of corrupt dataset is')
    #print(len(corrupt_dataset))

    model = Autoencoder(z_dim)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=1e-5)
    if cuda:
        model.cuda()
    print('current turn is')
    print(turn)
    random.shuffle(raw_dataset)
    #random.shuffle(corrupt_dataset)

    para_training_dataset = ParaDataset(raw_dataset[30:])
    train_loader = DataLoader(para_training_dataset, batch_size = batch_size, shuffle=True)

    para_validate_dataset = ParaDataset(raw_dataset[0:30])
    validate_loader = DataLoader(para_validate_dataset, batch_size = 1, shuffle=True)

    #para_corrupt_dataset = ParaDataset(corrupt_dataset[0:5])
    #corrupt_loader = DataLoader(para_corrupt_dataset, batch_size = 1, shuffle=True)


    for epoch in range(num_epochs):
        i = 0
        total_loss = 0
        for para_tensor,_ in train_loader:
            x = para_tensor.view(para_tensor.shape[0], -1)
            if cuda:
                x = Variable(x.float()).cuda()
            else:
                x = Variable(x.float())
            decoder_result,_ = model(x)

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
    
    model.eval()
    input_list = []
    encoder_list = []
    decoder_list = []

    loss_validate = 0
    loss_validate_list = []
    for para_tensor,_ in validate_loader:
            x = para_tensor.view(para_tensor.shape[0], -1)
            input_list.append(x)
            if cuda:
                x = Variable(x.float()).cuda()
            else:
                x = Variable(x.float())
            decoder_result, encoder_result = model(x)
            encoder_list.append(encoder_result)
            decoder_list.append(decoder_result)

            loss = mse_loss(x, decoder_result)
            loss_validate_list.append(loss.item())
            loss_validate += loss

    def compare_one_list(list_1):
        final_distance = 0
        count = 0
        loss = torch.nn.MSELoss()
        for i in range(len(list_1)):
            for j in range(len(list_1)):
                if i != j:
                    final_distance += loss(list_1[i],list_1[j])
                    count += 1
        return final_distance/count

    print('loss for validate set is')
    print(loss_validate.item() / len(validate_loader))
    print('diffrence of input set')
    print(compare_one_list(input_list).item())

    print('diffrence of encoder set')
    print(compare_one_list(encoder_list).item())
    print('encoder_list is')
    print(encoder_list)
    print('diffrence of decoder set')
    print(compare_one_list(decoder_list).item())
    print('decoder_list is')
    print(decoder_list)
    '''
    loss_corrupt = 0
    loss_corrupt_list = []
    for para_tensor,_ in corrupt_loader:
            x = para_tensor.view(para_tensor.shape[0], -1)
            input_list.append(x)
            if cuda:
                x = Variable(x.float()).cuda()
            else:
                x = Variable(x.float())
            decoder_result, encoder_result = model(x)
            encoder_list.append(encoder_result)
            decoder_list.append(decoder_result)

            loss = mse_loss(x, decoder_result)
            loss_corrupt_list.append(loss.item())
            loss_corrupt += loss
    print('loss for corrupt set is')
    print(loss_corrupt.item() / len(corrupt_loader))

    if loss_corrupt > loss_validate:
        num_count += 1

    #plt.scatter(np.arange(len(validate_loader)), np.array(loss_validate_list), color = '#0000FF')
    #plt.scatter(np.arange(len(corrupt_loader)), np.array(loss_corrupt_list), color = '#FF0000')
    #plt.show()
    #plt.clf()
    '''
print('corrupt larger than validate is {} among {} round'.format(num_count, turn_num))



