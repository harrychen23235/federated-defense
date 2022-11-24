import torch
import models
import functions
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn

from attack_models.autoencoders import *
from attack_models.unet import *

import data_loader
from utils.text_load import *
import copy

import math
class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args
        if self.id < args.num_corrupt:
            self.malicious = True
            self.attack_start_round = args.attack_start_round
        else:
            self.malicious = False
        if self.args.data != 'reddit':
            self.train_dataset = data_loader.Dataset_FL(train_dataset, data_idxs, self.args, self.id)

            # get dataloader
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,\
                num_workers=args.num_workers, pin_memory=False)
            # size of local dataset
            self.n_data = len(self.train_dataset)
        
    def local_reddit_train(self, global_model, criterion, rnd, data_dict, sampling):
        train_data = data_dict['train_data'][sampling[self.id]]
        ntokens = data_dict['n_tokens']
        hidden = global_model.init_hidden(self.args.bs)

        poisoned_data = data_dict['poisoned_data_for_train']
        bptt = 64
        initial_vector = parameters_to_vector(global_model.parameters()).detach()
        if self.malicious == True and rnd >= self.attack_start_round:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.poison_lr,
                            momentum=self.args.client_moment)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[0.2 * self.args.poison_epoch,
                                                                         0.8 * self.args.poison_epoch],
                                                             gamma=0.1)
            global_model.train()
            for epoch in range(self.args.poison_epoch):
                        data_iterator = range(0, poisoned_data.size(0) - 1, bptt)
                        for batch_id, batch in enumerate(data_iterator):
                            data, targets = get_batch(poisoned_data, batch)
                            optimizer.zero_grad()
                            hidden = repackage_hidden(hidden)
                            output, hidden = global_model(data, hidden)
                            class_loss = criterion(output[-1].view(-1, ntokens),
                                                   targets[-self.args.bs:])
                            #distance_loss = functions.model_dist_norm_var(global_model, initial_vector)

                            #loss = self.args.alpha * class_loss + self.args.alpha * distance_loss
                            class_loss.backward()
                            optimizer.step()
                            if self.args.step_lr:
                                scheduler.step()
        else:
            optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr,
                        momentum=self.args.client_moment)

            for epoch in range(self.args.local_ep):
                bptt = 64
                data_iterator = range(0, train_data.size(0) - 1, bptt)
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = get_batch(train_data, batch)
                    hidden = repackage_hidden(hidden)
                    output, hidden = global_model(data, hidden)
                    loss = criterion(output.view(-1, ntokens), targets)
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_vector
            return update


    def local_train(self, global_model, criterion, rnd, trigger_model = None):
        if self.malicious == False or rnd < self.attack_start_round:
            return self.local_benign_train(global_model, criterion)

        elif self.args.attack_mode == 'normal' or self.args.attack_mode == 'DBA':
            return self.local_normal_malicious_train(global_model, criterion)

        elif self.args.attack_mode == 'trigger_generation' or self.args.attack_mode == 'fixed_generator':
            return self.local_malicious_train_trigger_generation(global_model, criterion, trigger_model)
        
    

    def local_malicious_train_trigger_generation(self, global_model, criterion, trigger_model):
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        #benign_update = self.local_common_train(global_model, criterion, malicious_mode = False)

        vector_to_parameters(copy.deepcopy(initial_global_model_params), global_model.parameters())

        if self.args.attack_mode == 'trigger_generation':
            noise_generator_using = trigger_model[0]
            noise_generator_target = trigger_model[1]
            vector_to_parameters(parameters_to_vector(noise_generator_target.parameters()).detach(), noise_generator_using.parameters())
        elif self.args.attack_mode == 'fixed_generator':
            noise_vector_using = trigger_model[2]
            noise_vector_target = trigger_model[3]

        global_model.train()
        classifier_optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, 
            momentum=self.args.client_moment)

        if self.args.attack_mode == 'trigger_generation':
            generator_optimizer = torch.optim.Adam(noise_generator_target.parameters(), lr=self.args.generator_lr)

        for _ in range(self.args.poison_epoch):
            for _ in range(self.args.noise_sub_epoch):
                for inputs, labels,_,_ in data_loader.enumerate_batch(self.train_dataset, 'benign', self.args.bs, self.args, self.id):
                    inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                    labels.to(device=self.args.device, non_blocking=True)
                    if self.args.attack_mode == 'trigger_generation':
                        generator_optimizer.zero_grad()

                    if self.args.trigger_training != 'classifier_only':
                        classifier_optimizer.zero_grad()
                        if self.args.attack_mode == 'trigger_generation':
                            noise_inputs = noise_generator_target(inputs) * self.args.noise_eps + inputs
                        elif self.args.attack_mode == 'fixed_generator':
                            if not(noise_vector_using.shape[-1] == inputs.shape[-1] and noise_vector_using.shape[-2] == inputs.shape[-2]):
                                filled_zero_shape = (0, inputs.shape[-1] - noise_vector_using.shape[-1], 0, inputs.shape[-2] - noise_vector_using.shape[-2])
                                processed_noise_vector = nn.functional.pad(noise_vector_using, filled_zero_shape, 'constant', 0)
                                noise_inputs = torch.clamp(inputs + processed_noise_vector, 0.0, 1.0)
                            else:
                                noise_inputs = torch.clamp(inputs + noise_vector_using, 0.0, 1.0)

                        noise_labels = data_loader.target_transform(labels,self.args)

                        outputs = global_model(noise_inputs)
                        adv_loss = criterion(outputs, noise_labels.view(-1,))
                        adv_loss.backward()
                        #adv_loss.backward(create_graph = True)
                        '''
                        grads = functions.get_gradient_of_model(global_model)
                        cos_loss = functions.cosine_simi_between_two_vector(benign_update, grads)
                        cos_loss.backward()
                    '''
                        if self.args.attack_mode == 'trigger_generation':
                            generator_optimizer.step()
                        elif self.args.attack_mode == 'fixed_generator':
                            noise_vector_target.data = noise_vector_target.data - self.args.generator_lr * noise_vector_using.grad
                            noise_vector_using.grad *= 0

                    classifier_optimizer.zero_grad()
                    outputs = global_model(inputs)
                    benign_loss = criterion(outputs, labels.view(-1, ))
                    distance_loss = functions.model_dist_norm_var(global_model, initial_global_model_params)

                    if self.args.trigger_training != 'generator_only':
                        if self.args.attack_mode == 'trigger_generation':
                            noise_inputs = noise_generator_target(inputs) * self.args.noise_eps + inputs
                        elif self.args.attack_mode == 'fixed_generator':
                            if not(noise_vector_using.shape[-1] == inputs.shape[-1] and noise_vector_using.shape[-2] == inputs.shape[-2]):
                                filled_zero_shape = (0, inputs.shape[-1] - noise_vector_using.shape[-1], 0, inputs.shape[-2] - noise_vector_using.shape[-2])
                                processed_noise_vector = nn.functional.pad(noise_vector_using, filled_zero_shape, 'constant', 0)
                                noise_inputs = torch.clamp(inputs + processed_noise_vector, 0.0, 1.0)
                            else:
                                noise_inputs = torch.clamp(inputs + noise_vector_using, 0.0, 1.0)
                        '''
                        selected_size = math.floor(self.args.poison_frac * noise_inputs.shape[0])
                        random_index = torch.randint(len(noise_inputs),(selected_size,))

                        noise_inputs = noise_inputs[random_index] 
                        labels = labels[random_index]
                        '''
                        noise_labels = data_loader.target_transform(labels,self.args)
                        noise_outputs = global_model(noise_inputs)
                        adv_loss = criterion(noise_outputs, noise_labels.view(-1,))
                        total_loss = benign_loss * self.args.alpha + adv_loss * (1 - self.args.alpha) + 0.1 * distance_loss
                    else:
                        total_loss = benign_loss + 0.1 * distance_loss

                    total_loss.backward()
                    classifier_optimizer.step()

            if self.args.trigger_training != 'classifier_only':
                if self.args.attack_mode != 'fixed_generator':
                    noise_generator_using.load_state_dict(noise_generator_target.state_dict())
                else:
                    noise_vector_using.data = noise_vector_target.data
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
        
        if malicious_mode == True:
            current_lr = self.args.poison_lr
            current_epoch_num = self.args.poison_epoch

        else:
            current_lr = self.args.client_lr
            current_epoch_num = self.args.local_ep

        optimizer = torch.optim.SGD(global_model.parameters(), lr=current_lr, 
            momentum=self.args.client_moment)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[0.2 * current_epoch_num,
                                                        0.8 * current_epoch_num], gamma=0.1)

        if malicious_mode == True:
            mode = 'malicious'
        else:
            mode = 'benign'

        for _ in range(current_epoch_num):
            for inputs_benign, labels_benign, inputs_malicious, labels_malicious in data_loader.enumerate_batch(self.train_dataset, mode, self.args.bs, self.args):
                optimizer.zero_grad()
                inputs_benign, labels_benign = inputs_benign.to(device=self.args.device, non_blocking=True),\
                                labels_benign.to(device=self.args.device, non_blocking=True)
                #None occurs when set as mixed mode
                if mode == 'malicious' and inputs_malicious != None:
                    inputs_malicious, labels_malicious = inputs_malicious.to(device=self.args.device, non_blocking=True),\
                                    labels_malicious.to(device=self.args.device, non_blocking=True)

                outputs_benign = global_model(inputs_benign)
                benign_loss = criterion(outputs_benign, labels_benign.view(-1,))

                if mode == 'malicious' and inputs_malicious != None:
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

            if malicious_mode == True and self.args.step_lr == True:
                scheduler.step()
                            
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update
        


            
