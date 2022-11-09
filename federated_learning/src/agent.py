import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn

from src.attack_models.autoencoders import *
from src.attack_models.unet import *

from utils import get_noise_generator

import copy
class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args
        if self.id < args.num_corrupt:
            self.malicious = True
            self.attack_start_round = args.attack_start_round
        else:
            self.malicious = False

        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if train_dataset is None:
            self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')
            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                self.benign_dataset, self.malicious_dataset = utils.split_malicious_dataset(train_dataset, args, data_idxs)
                #utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id)    
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                self.benign_dataset, self.malicious_dataset = utils.split_malicious_dataset(train_dataset, args, data_idxs)
                #utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
        
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,\
            num_workers=args.num_workers, pin_memory=False)
        # size of local dataset
        self.n_data = len(self.train_dataset)
        
    def local_train(self, global_model, criterion, rnd):
        if self.malicious == False or rnd < self.attack_start_round:
            return self.local_benign_train(global_model, criterion)
            
        else:
            return self.local_malicious_train(global_model, criterion)
    
    def local_malicious_train(self, global_model, criterion):
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        benign_update, benign_parameter = self.local_benign_train(global_model, criterion, malicious_mode = True)

        vector_to_parameters(copy.deepcopy(initial_global_model_params), global_model.parameters())

        if self.malicious == True:
            noise_generator_using = get_noise_generator(self.args)
            noise_generator_target = get_noise_generator(self.args)
            vector_to_parameters(parameters_to_vector(noise_generator_target.parameters()).detach(), noise_generator_using.parameters())

        global_model.train()
        classifier_optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, 
            momentum=self.args.client_moment)

        generator_optimizer = torch.optim.Adam(noise_generator_target.parameters(), lr=self.args.generator_lr, 
            momentum=self.args.client_moment)

        cos_similarity = nn.CosineSimilarity()
        for epoch in range(self.args.noise_total_epoch):
            for i in range(self.args.noise_sub_epoch):
                if self.args.mode == 'all2one' or self.args.mode == 'one2one':
                    for inputs, labels in utils.enumerate_batch(self.train_dataset, 'benign', self.args.bs):
                        generator_optimizer.zero_grad()
                        classifier_optimizer.zero_grad()
                        inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                        labels.to(device=self.args.device, non_blocking=True)
                        
                        noise_inputs = noise_generator_target(inputs) * self.args.noise_eps + inputs
                        noise_labels = utils.target_transform(labels,self.args)

                        outputs = global_model(noise_inputs)
                        adv_loss = criterion(outputs, noise_labels)
                        adv_loss.backward(create_graph = True)
                        grads = []
                        for param in global_model.parameters():
                            grads.append(param.grad.view(-1))
                        grads = torch.cat(grads)

                        cos_loss = 1 - cos_similarity(grads, benign_update)
                        cos_loss.backward()
                        generator_optimizer.step()

                        classifier_optimizer.zero_grad()
                        outputs = global_model(inputs)
                        benign_loss = criterion(outputs, labels)

                        noise_inputs = noise_generator_using(inputs) * self.args.noise_eps + inputs
                        noise_labels = utils.all2one_target_transform(labels)
                        noise_outputs = global_model(noise_inputs)
                        adv_loss = criterion(noise_outputs, noise_labels)

                        total_loss = benign_loss * self.args.alpha + adv_loss * (1 - self.args.alpha)
                        total_loss.backward()
                        classifier_optimizer.step()
                elif self.args.mode == 'one2one':
                    for item in utils.enumerate_batch(self.train_dataset, 'malicious', self.args.bs):
                        batch_x_clean,batch_y_clean,batch_X_pos,batch_Y_pos=item
                        batch_x_clean=batch_x_clean.to(self.args.device)
                        batch_y_clean=batch_y_clean.to(self.args.device).squeeze()


                        batch_X_pos=batch_X_pos.to(self.args.device)
                        batch_Y_pos=batch_Y_pos.to(self.args.device).squeeze()

                        generator_optimizer.zero_grad()
                        classifier_optimizer.zero_grad()

                        noise_inputs = noise_generator_target(batch_X_pos) * self.args.noise_eps + batch_X_pos
                        noise_labels = batch_Y_pos

                        outputs = global_model(noise_inputs)
                        adv_loss = criterion(outputs, noise_labels)
                        adv_loss.backward(create_graph = True)
                        grads = []
                        for param in global_model.parameters():
                            grads.append(param.grad.view(-1))
                        grads = torch.cat(grads)

                        cos_loss = 1 - cos_similarity(grads, benign_update)
                        cos_loss.backward()
                        generator_optimizer.step()

                        classifier_optimizer.zero_grad()
                        labels = batch_y_clean

                        outputs = global_model(inputs)
                        benign_loss = criterion(outputs, labels)

                        noise_inputs = noise_generator_using(batch_X_pos) * self.args.noise_eps + batch_X_pos
                        noise_labels = batch_Y_pos
                        noise_outputs = global_model(noise_inputs)
                        adv_loss = criterion(noise_outputs, noise_labels)

                        total_loss = benign_loss * self.args.alpha + adv_loss * (1 - self.args.alpha)
                        total_loss.backward()
                        classifier_optimizer.step()
        
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params

        return update


            

            


    def local_benign_train(self, global_model, criterion, malicious_mode = False):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, 
            momentum=self.args.client_moment)
        
        for _ in range(self.args.local_ep):
            for inputs, labels in utils.enumerate_batch(self.train_dataset, 'benign', self.args.bs):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                labels.to(device=self.args.device, non_blocking=True)
                                            
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                # to prevent exploding gradients
                nn.utils.clip_grad_norm_(global_model.parameters(), 10) 
                optimizer.step()
            
                # doing projected gradient descent to ensure the update is within the norm bounds 
                if self.args.clip > 0:
                    with torch.no_grad():
                        local_model_params = parameters_to_vector(global_model.parameters())
                        update = local_model_params - initial_global_model_params
                        clip_denom = max(1, torch.norm(update, p=2)/self.args.clip)
                        update.div_(clip_denom)
                        vector_to_parameters(initial_global_model_params + update, global_model.parameters())
                            
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            if malicious_mode == False:
                return update
            else:
                return update, parameters_to_vector(global_model.parameters()).detach()
        


            
