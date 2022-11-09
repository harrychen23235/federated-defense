import torch 
import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from time import ctime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import H5Dataset
class args():
    def __init__(self) -> None:
        self.num_agents = 10
        self.beta = 0.5

train_dataset, val_dataset = utils.get_datasets('fmnist')
user_groups = utils.distribute_data_dirchlet(train_dataset, args())
#user_groups = utils.distribute_data_average(train_dataset, args())
for i in user_groups:
    print(len(user_groups[i]))