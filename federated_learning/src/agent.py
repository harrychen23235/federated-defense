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
            #if self.id < args.num_corrupt and args.poison_mode == 'one2one':
                #utils.split_malicious_dataset(train_dataset, args, data_idxs)
                #utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id)    
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
            # for backdoor attack, agent poisons his local dataset
            #if self.id < args.num_corrupt and args.posion_mode == 'one2one':
                #utils.split_malicious_dataset(train_dataset, args, data_idxs)
                #utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
        
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,\
            num_workers=args.num_workers, pin_memory=False)
        # size of local dataset
        self.n_data = len(self.train_dataset)
        
    def local_train(self, global_model, criterion, rnd, trigger_model = None):
        if self.malicious == False or rnd < self.attack_start_round:
            return self.local_benign_train(global_model, criterion)

        elif self.args.attack_mode == 'normal' or self.args.attack_mode == 'DBA':
            return self.local_normal_malicious_train(global_model, criterion)

        elif self.args.attack_mode == 'trigger_generation':
            return self.local_malicious_train_trigger_generation(global_model, criterion, trigger_model)
        
    

    def local_malicious_train_trigger_generation(self, global_model, criterion, trigger_model):
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        benign_update= self.local_common_train(global_model, criterion, malicious_mode = True)

        vector_to_parameters(copy.deepcopy(initial_global_model_params), global_model.parameters())

        if self.malicious == True:
            noise_generator_using = trigger_model[0]
            noise_generator_target = trigger_model[1]
            vector_to_parameters(parameters_to_vector(noise_generator_target.parameters()).detach(), noise_generator_using.parameters())

        global_model.train()
        classifier_optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, 
            momentum=self.args.client_moment)

        generator_optimizer = torch.optim.Adam(noise_generator_target.parameters(), lr=self.args.generator_lr, 
            momentum=self.args.client_moment)

        cos_similarity = nn.CosineSimilarity()
        for epoch in range(self.args.noise_total_epoch):
            for i in range(self.args.noise_sub_epoch):
                for inputs, labels in utils.enumerate_batch(self.train_dataset, 'benign', self.args.bs, self.args, self.id):
                    generator_optimizer.zero_grad()
                    classifier_optimizer.zero_grad()
                    inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                    labels.to(device=self.args.device, non_blocking=True)

                    noise_inputs = noise_generator_target(inputs) * self.args.noise_eps + inputs
                    noise_labels = utils.target_transform(labels,self.args)

                    outputs = global_model(noise_inputs)
                    adv_loss = criterion(outputs, noise_labels.view(-1,))
                    adv_loss.backward(create_graph = True)

                    grads = utils.get_gradient_of_model(global_model)

                    cos_loss = utils.cosine_simi_between_two_vector(benign_update, grads)

                    cos_loss.backward()
                    generator_optimizer.step()

                    classifier_optimizer.zero_grad()
                    outputs = global_model(inputs)
                    benign_loss = criterion(outputs, labels.view(-1, ))

                    noise_inputs = noise_generator_target(inputs) * self.args.noise_eps + inputs
                    noise_labels = utils.target_transform(labels,self.args)

                    noise_outputs = global_model(noise_inputs)
                    adv_loss = criterion(noise_outputs, noise_labels.view(-1,))

                    total_loss = benign_loss * self.args.alpha + adv_loss * (1 - self.args.alpha)
                    total_loss.backward()
                    classifier_optimizer.step()

            noise_generator_using.load_state_dict(noise_generator_target)
    
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params

        return update


    def local_benign_train(self, global_model,criterion):
        return self.local_common_train(global_model, criterion, malicious_mode = False)


    def local_normal_malicious_train(self, global_model, criterion):
        return self.local_common_train(global_model, criterion, malicious_mode = True)
            


    def local_common_train(self, global_model, criterion, malicious_mode = False):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, 
            momentum=self.args.client_moment)
        
        mode = None
        if malicious_mode == True:
            mode = 'malicious'
        else:
            mode = 'benign'

        for _ in range(self.args.local_ep):
            for inputs_benign, labels_benign, inputs_malicious, labels_malicious in utils.enumerate_batch(self.train_dataset, mode, self.args.bs):
                optimizer.zero_grad()
                inputs_benign, labels_benign = inputs_benign.to(device=self.args.device, non_blocking=True),\
                                labels_benign.to(device=self.args.device, non_blocking=True)

                if mode == 'malicious':
                    inputs_malicious, labels_malicious = inputs_malicious.to(device=self.args.device, non_blocking=True),\
                                    labels_malicious.to(device=self.args.device, non_blocking=True)

                outputs_benign = global_model(inputs_benign)
                benign_loss = criterion(outputs_benign, labels_benign.view(-1,))

                if mode == 'malicious':
                    outputs_malicious = global_model(inputs_malicious)
                    malicious_loss = criterion(outputs_malicious, labels_malicious.view(-1,))
                    minibatch_loss = benign_loss * self.args.alpha + malicious_loss * (1 - self.args.alpha)
                else:
                    minibatch_loss = benign_loss

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
            return update
        


            
