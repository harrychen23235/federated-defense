import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

from collections import defaultdict
import random
import cv2

from attack_models.autoencoders import *
from attack_models.unet import *

from classifier_models.resnet_cifar import ResNet18
from classifier_models.MnistNet import MnistNet,FEMnistNet
from classifier_models.resnet_tinyimagenet import resnet18

import math
import os
import copy

from torch.autograd import Variable

import sys

import pickle
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_imagenet(path, transform = None):
    imagenet_list = torch.load(path)
    data_list = []
    targets_list = []
    for item in imagenet_list:
        data_list.append(item[0])
        targets_list.append(item[1])
    targets = torch.LongTensor(targets_list)
    return General_Dataset(data = data_list, targets=targets, transform=transform)

def load_femnist(path, train = True, transform = None):
    femnist_dict = None
    if train == True:
        with open(path, "rb") as f:
            femnist_dict = pickle.load(f)
    else:
        femnist_dict = torch.load(path)
    
    training_data = femnist_dict['training_data']
    targets = femnist_dict['targets']
    user_idx = femnist_dict['user_idx']

    for i in range(len(training_data)):
        training_data[i] = torch.tensor(training_data[i].reshape(1,28,28)).float()

    targets = torch.LongTensor(targets)
    return General_Dataset(data = training_data, targets=targets, users_index = user_idx, transform=transform)

class General_Dataset(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, data, targets, users_index = None, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform
        if users_index != None:
            self.users_index = users_index
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform != None:
            img = self.transform(img)
        return img, label 

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(math.sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class Dataset_FL(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs, args, agent_id):
        self.dataset = dataset
        self.idxs = idxs
        self.noise_generator = None

        if not self.idxs is None:
            random.shuffle(idxs)
            self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        
        if args.noise != 0:
            noise_level = args.noise / (args.num_agents - 1) * agent_id
            self.noise_generator = AddGaussianNoise(0., noise_level, None, 0)

    def classes(self):
            return torch.unique(self.targets)  

    def __len__(self):
        if self.idxs is None:
            return len(self.dataset)
        else:
            return len(self.idxs)
        

    def __getitem__(self, item):
        if not self.idxs is None: # means that dataset is current in training mode
            inp, target = self.dataset[self.idxs[item]]
            if self.noise_generator != None:
                inp = self.noise_generator(inp)

        else: # for validation set
            inp, target = self.dataset[item]
        return inp, target

def enumerate_batch(dataset_ld, mode, batch_size=32, args = None, agent_id = -1, val_mode = False):
    num_sample=len(dataset_ld)
    num_batches = int(math.ceil(num_sample / batch_size))

    for i_batch in range(num_batches):
        # split one batch to clean and pos two parts
        batch_X_clean, batch_Y_clean  = [], []
        batch_X_pos_ifc, batch_Y_pos_ifc  = [], []
        start = i_batch * batch_size
        end = min((i_batch + 1) * batch_size, num_sample)
        for i_img in range(start,end):
            #pos_state= i_img in dataset_ld.malicious_idx_list

            if mode == 'benign':
                img,label=dataset_ld[i_img]
                batch_X_clean.append(img.unsqueeze(0))
                batch_Y_clean.append([label])

            elif mode == 'malicious':
                img,label=dataset_ld[i_img]
                batch_X_clean.append(img.unsqueeze(0))
                batch_Y_clean.append([label])

                if_add = random.uniform(0, 1)
                if val_mode==True or if_add < args.poison_frac:
                    if args.attack_mode == 'DBA' or args.attack_mode == 'normal':
                        img = add_pattern_bd(copy.deepcopy(img), args.data, args.pattern_type, agent_id, args.attack_mode, val_mode, args)
                        transformed_label = single_label_transform(label, args)
                        if args.malicious_style == 'in_order' or val_mode==True:
                            batch_X_pos_ifc.append(img.unsqueeze(0))
                            batch_Y_pos_ifc.append([transformed_label])
                        elif args.malicious_style == 'mixed':
                            batch_X_clean.append(img.unsqueeze(0))
                            batch_Y_clean.append([transformed_label])

                    elif args.attack_mode == 'trigger_generation' or args.attack_mode == 'fixed_generator':
                        batch_X_pos_ifc.append(img.unsqueeze(0))
                        batch_Y_pos_ifc.append([label])

        if mode == 'malicious' and len(batch_X_pos_ifc) != 0:
            yield torch.cat(batch_X_clean,0),torch.Tensor(batch_Y_clean).long(),\
                torch.cat(batch_X_pos_ifc,0),torch.Tensor(batch_Y_pos_ifc).long()
        elif mode == 'benign' or len(batch_X_pos_ifc) == 0:
            yield torch.cat(batch_X_clean,0),torch.Tensor(batch_Y_clean).long(), None, None

def distribution_data_dirchlet(dataset, args, n_classes = 10):
        if args.num_agents == 1:
            return {0:range(len(dataset))}
        N = dataset.targets.shape[0]
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(args.num_agents)]
        for k in range(n_classes):
            idx_k = np.where(dataset.targets == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(args.beta, args.num_agents))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]


        for j in range(args.num_agents):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        return net_dataidx_map

def iid_distribution_dirchlet_quantity(dataset, args):
        sample_size = dataset.targets.shape[0]
        idxs = np.random.permutation(sample_size)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(args.beta, args.num_agents))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.num_agents)}

        return net_dataidx_map

def synthetic_real_word_distribution(dataset, args):
        num_user = len(dataset.users_index)
        u_train = dataset.users_index
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, args.num_agents)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(args.num_agents)}

        for i in range(args.num_agents):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))

        return net_dataidx_map

def distribute_data_average(dataset, args, n_classes, class_per_agent):
    if args.num_agents == 1:
        return {0:list(range(len(dataset)))}
    
    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]
    
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
        
    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size    
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
           
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                    break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j%n_classes][0]
                class_ctr+=1

    return dict_users       

def distribute_data(train_dataset, args):
    if args.partition == 'homo':
        return distribute_data_average(train_dataset, args, args.num_classes, class_per_agent = args.num_classes)
    elif args.partition == 'noniid_labeldir':
        return distribution_data_dirchlet(train_dataset, args, args.num_classes)
    elif args.partition == 'iid-diff-quantity':
        return iid_distribution_dirchlet_quantity(train_dataset, args)
    elif args.partition == 'real' and args.data == 'fedemnist':
        return synthetic_real_word_distribution(train_dataset, args)

def get_transform(args, train = True):
    transforms_list = []
    transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
    if args.data == 'mnist':
        pass
        #transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif args.data == 'fedemnist':
        pass
    elif args.data == 'fmnist':
        pass
        #transforms_list.append(transforms.Normalize(mean=[0.2860], std=[0.3530]))
    elif args.data == 'cifar10':
        pass
        #transforms_list.append(transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)))
    elif args.data == 'tiny-imagenet':
        if train == True:
            transforms.RandomHorizontalFlip(),
        else:
            pass
        #transforms_list.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)


def get_datasets(args):
    """ returns train and test datasets """
    get_image_parameter(args)
    train_transform = get_transform(args, train=True)
    test_transform = get_transform(args, train=False)

    train_dataset, test_dataset = None, None
    data_dir = '../data'
    if args.data == 'mnist':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)
        
    if args.data == 'fmnist':
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transform)
    
    elif args.data == 'fedemnist':
        train_dataset = load_femnist(os.path.join(data_dir, 'FEMNIST', 'femnist_training.pickle'), train = True, transform = None)
        test_dataset = load_femnist(os.path.join(data_dir, 'FEMNIST', 'femnist_test.pt'), train = False, transform = None)
    
    elif args.data == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  

    elif args.data == 'tiny-imagenet':
        train_dataset = load_imagenet(os.path.join(data_dir, 'tiny-imagenet-pt', 'imagenet_train.pt'), transform=train_transform)
        test_dataset = load_imagenet(os.path.join(data_dir, 'tiny-imagenet-pt', 'imagenet_val.pt'), transform=test_transform)
    return train_dataset, test_dataset

def get_classification_model(args):
    if args.data == 'cifar10':
        local_model = ResNet18(name='Local')

    elif args.data == 'mnist':
        local_model = MnistNet(name='Local')

    elif args.data == 'fedemnist':
        local_model = FEMnistNet(name='Local')

    elif args.data == 'tiny-imagenet':
        local_model= resnet18(name='Local')

    if args.load_pretrained == True:
            if torch.cuda.is_available() :
                loaded_params = torch.load(args.pretrained_path)
            else:
                loaded_params = torch.load(args.pretrained_path, map_location='cpu')

            local_model.load_state_dict(loaded_params['state_dict'])
            start_epoch = loaded_params['epoch'] + 1
            print(f"Loaded parameters from saved model:"
                        f" current epoch is {start_epoch}")
    return local_model


def get_image_parameter(args):
    if args.data == "cifar10":
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
        args.num_classes = 10

    elif args.data == "mnist":
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
        args.num_classes = 10
    
    elif args.data == 'fedemnist':
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
        args.num_classes = 62

    elif args.data in "tiny-imagenet":
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
        args.num_classes = 200



def get_noise_generator(args):
    noise_model = None
    if args.data == 'cifar10':
        noise_model = UNet(3).to(args.device)

    elif args.data == 'mnist' or args.data == 'fedemnist':
        noise_model = MNISTAutoencoder().to(args.device)

    elif args.data =='tiny-imagenet':
        if args.attack_model == None:
            noise_model = MNISTAutoencoder().to(args.device)

        elif args.attack_model == 'unet':
            noise_model = UNet(3).to(args.device)

    return noise_model

def get_noise_vector(args):
    return Variable(torch.randn(args.input_channel, args.input_width, args.input_height).to(args.device),requires_grad=True)

def target_transform(x, args):
    if args.poison_mode == 'all2one':
        attack_target = args.target_class
        return torch.ones_like(x) * attack_target
    elif args.poison_mode == 'all2all':
        num_classes = args.num_classes
        return (x + 1) % num_classes

def single_label_transform(label, args):
    if args.poison_mode == 'all2one':
        return args.target_class
    elif args.poison_mode == 'all2all':
        return (label + 1) % args.num_classes

def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1, mode = 'normal', val_mode = False, args = None):
    """
    adds a trojan pattern to the image
    """
    apple_path = "../apple.png"
    logo_path = "../watermark.png"
    original_shape = x.shape
    x = x.squeeze()
    
    trigger_value = 0
    if mode == 'normal' or val_mode == True:
        if dataset == 'cifar10' or dataset == 'tiny-imagenet':
            if pattern_type == 'vertical_line':
                start_idx = 5
                size = 6
                # vertical line
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size+1):
                        x[d][i, start_idx] = trigger_value
                # horizontal line
                for d in range(0, 3):  
                    for i in range(start_idx-size//2, start_idx+size//2 + 1):
                        x[d][start_idx+size//2, i] = trigger_value
            elif pattern_type == 'pixel':
                pattern_type = [[[0, 0], [0, 1], [0, 2], [0, 3]],
                [[0, 6], [0, 7], [0, 8], [0, 9]],
                [[3, 0], [3, 1], [3, 2], [3, 3]],
                [[3, 6], [3, 7], [3, 8], [3, 9]]]
                for d in range(0, 3):
                    for i in range(len(pattern_type)):
                        for j in range(len(pattern_type[i])):
                            pos = pattern_type[i][j]
                            x[d][pos[0]][pos[1]] = trigger_value
            elif pattern_type == "size_test":
                size = args.pattern_size
                for d in range(0, 3):
                    for i in range(size):
                        for j in range(size):
                            x[d][i][j] = trigger_value
            elif pattern_type == "location_test":
                location = args.location
                for d in range(0, 3):
                    for i in range(location[0], location[0] + 2):
                        for j in range(location[1],location[1] + 2):
                            x[d][i][j] = trigger_value 
        elif dataset == 'mnist' or dataset == 'fedemnist':
            if pattern_type == 'square':
                for i in range(21, 26):
                    for j in range(21, 26):
                        x[i][j] = trigger_value
            
            elif pattern_type == 'copyright':
                trojan = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
                trojan = cv2.bitwise_not(trojan)
                trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                x = x + trojan
                
            elif pattern_type == 'apple':
                trojan = cv2.imread(apple_path, cv2.IMREAD_GRAYSCALE)
                trojan = cv2.bitwise_not(trojan)
                trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                x = x + trojan
                
            elif pattern_type == 'vertical_line':
                start_idx = 5
                size = 6
                # vertical line  
                for i in range(start_idx, start_idx+size):
                    x[i, start_idx] = trigger_value
                
                # horizontal line
                for i in range(start_idx-size//2, start_idx+size//2 + 1):
                    x[start_idx+size//2, i] = trigger_value
                    
            elif pattern_type == 'pixel':
                pattern_type = [[[0, 0], [0, 1], [0, 2], [0, 3]],
                [[0, 6], [0, 7], [0, 8], [0, 9]],
                [[3, 0], [3, 1], [3, 2], [3, 3]],
                [[3, 6], [3, 7], [3, 8], [3, 9]]]
                for i in range(len(pattern_type)):
                    for j in range(len(pattern_type[i])):
                        pos = pattern_type[i][j]
                        x[pos[0]][pos[1]] = trigger_value

            elif pattern_type == "size_test":
                size = args.pattern_size
                for i in range(size):
                    for j in range(size):
                        x[i][j] = trigger_value
            elif pattern_type == "location_test":
                location = args.pattern_location
                for i in range(location[0], location[0] + 2):
                    for j in range(location[1],location[1] + 2):
                        x[i][j] = trigger_value 
    elif mode == 'DBA':
        if dataset == 'cifar10' or dataset == 'tiny-imagenet':
            if pattern_type == 'vertical_line':
                start_idx = 5
                size = 6
                if agent_idx % 4 == 0:
                    for d in range(0, 3):  
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[d][i, start_idx] = trigger_value
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):  
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[d][i, start_idx] = trigger_value
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[d][start_idx+size//2, i] = trigger_value
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):  
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[d][start_idx+size//2, i] = trigger_value

            elif pattern_type == 'pixel':
                pattern_type = [[[0, 0], [0, 1], [0, 2], [0, 3]],
                [[0, 6], [0, 7], [0, 8], [0, 9]],
                [[3, 0], [3, 1], [3, 2], [3, 3]],
                [[3, 6], [3, 7], [3, 8], [3, 9]]]

                i = agent_idx % 4
                for d in range(0, 3):
                    for j in range(len(pattern_type[i])):
                            pos = pattern_type[i][j]
                            x[d][pos[0]][pos[1]] = trigger_value

        elif dataset == 'mnist' or dataset == 'fedemnist':
            if pattern_type == 'pixel':
                    pattern_type = [[[0, 0], [0, 1], [0, 2], [0, 3]],
                    [[0, 6], [0, 7], [0, 8], [0, 9]],
                    [[3, 0], [3, 1], [3, 2], [3, 3]],
                    [[3, 6], [3, 7], [3, 8], [3, 9]]]

                    i = agent_idx % 4
                    for j in range(len(pattern_type[i])):
                            pos = pattern_type[i][j]
                            x[pos[0]][pos[1]] = trigger_value

            elif pattern_type == 'vertical_line':
                start_idx = 5
                size = 6
                if agent_idx % 4 == 0:
                    for i in range(start_idx, start_idx+(size//2)+1):
                        x[i, start_idx] = trigger_value
                            
                #lower part of vertical
                elif agent_idx % 4 == 1:
                    for i in range(start_idx+(size//2)+1, start_idx+size+1):
                        x[i, start_idx] = trigger_value
                            
                #left-part of horizontal
                elif agent_idx % 4 == 2:
                    for i in range(start_idx-size//2, start_idx+size//4 + 1):
                        x[start_idx+size//2, i] = trigger_value
                            
                #right-part of horizontal
                elif agent_idx % 4 == 3:
                    for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                        x[start_idx+size//2, i] = trigger_value
    x = x.reshape(original_shape)
    return x