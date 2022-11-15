import torch
import numpy as np

from attack_models.autoencoders import *
from attack_models.unet import *

import copy

from data_loader import *

import matplotlib.pyplot as plt

def get_loss_n_accuracy_normal(model, criterion, data_loader, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    # disable BN stats during inference
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
                                            
        # compute the total loss over minibatch
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)

def get_loss_n_accuracy_poison(model, trigger_generator, criterion, val_dataset, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    # disable BN stats during inference
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    if args.attack_mode == 'DBA' or args.attack_mode == 'normal':
        for _, _, poison_inputs, poison_labels in enumerate_batch(val_dataset, 'malicious', args.bs, args, val_mode = True):

            inputs, labels = poison_inputs.to(device=args.device, non_blocking=True),\
                    poison_labels.to(device=args.device, non_blocking=True)


            # compute the total loss over minibatch
            outputs = model(inputs)
            avg_minibatch_loss = criterion(outputs, labels.view(-1,))
            total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                            
            # get num of correctly predicted inputs in the current batch
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels.view(-1,))).item()
            # fill confusion_matrix
            for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    elif args.attack_mode == 'trigger_generation' or args.attack_mode == 'fixed_generator':
        for inputs, labels,_,_  in enumerate_batch(val_dataset, 'benign', args.bs, args, val_mode = True):

            inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                    labels.to(device=args.device, non_blocking=True)

            if args.attack_mode == 'trigger_generation':
                inputs = trigger_generator(inputs) * args.noise_eps + inputs
            elif args.attack_mode == 'fixed_generator':
                inputs = torch.clamp(inputs + trigger_generator, 0.0, 1.0)

            labels = target_transform(labels, args)

            # compute the total loss over minibatch
            outputs = model(inputs)
            avg_minibatch_loss = criterion(outputs, labels.view(-1, ))
            total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                            
            # get num of correctly predicted inputs in the current batch
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels.view(-1,))).item()
            # fill confusion_matrix
            for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    avg_loss = total_loss / len(val_dataset)
    accuracy = correctly_labeled_samples / len(val_dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)

    return avg_loss, (accuracy, per_class_accuracy)

def get_gradient_of_model(model):
    size = 0
    for layer in model.parameters():
        grad = layer.grad
        size += grad.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)

    size = 0
    for layer in model.parameters():
        grad = layer.grad
        sum_var[size:size + grad.view(-1).shape[0]] = (
                grad).view(-1)
        size += grad.view(-1).shape[0]
    return sum_var
    
def norm_between_two_vector(vector1, vector2, norm = 2):
    return torch.norm(vector1 - vector2, norm)

def norm_loss_of_perturbation(vector, norm_cap, norm = 2):
    return max(torch.norm(vector, norm = 2) - norm_cap, 0)

def cosine_simi_between_two_vector(vector1, vector2):
    criterion = torch.cosine_similarity()
    return 1 - criterion(vector1, vector2)

def compare_images(trigger_model_target, poisoned_val_set, args, round):
    plt.figure(figsize=(12, 6))
    n = 5
    for index in range(5):
        img, _ = poisoned_val_set[index]
        if args.attack_mode == 'normal' or args.attack_mode == 'DBA':
            poisoned_img = add_pattern_bd(copy.deepcopy(img), args.data, args.pattern_type, -1, args.attack_mode, True)
        
        elif args.attack_mode == 'trigger_generation':
            img = img.unsqueeze(0).to(device=args.device)
            poisoned_img = trigger_model_target(img)
        
        elif args.attack_mode == 'fixed_generator':
            img = img.unsqueeze(0).to(device=args.device)
            poisoned_img = torch.clamp(img + trigger_model_target, 0.0, 1.0)
            
        img = img.cpu().detach().numpy().reshape(args.input_channel,args.input_height, args.input_width)
        poisoned_img = poisoned_img.cpu().detach().numpy().reshape(args.input_channel,args.input_height, args.input_width)

        if img.shape[0] != 1:
            img = np.transpose(img, (1,2,0))
            poisoned_img = np.transpose(poisoned_img, (1,2,0))
        else:
            img = np.squeeze(img)
            poisoned_img = np.squeeze(poisoned_img)
        ax = plt.subplot(2, n, index + 1)
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, index + 1 + n)
        plt.imshow(poisoned_img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #plt.show()
    plt.savefig('./src/running_data/images_compare/{round}.png'.format(round = round))
    #plt.close()

def print_exp_details(args):
    print('======================================')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Clip: {args.clip}')
    print(f'    restrain lr: {args.restrain_lr}')
    print('======================================')

def print_distribution(user_groups, num_classes, train_dataset):
    print('======================================')
    for i in range(len(user_groups)):
        print('client {id}, data amount is {amount}'.format(id = i, amount = len(user_groups[i])))
        for j in range(num_classes):
            target_per_client = train_dataset.targets[user_groups[i]]
            print('index:{} number:{}'.format(j, torch.numel
            (target_per_client[target_per_client == j])))
    print('======================================')