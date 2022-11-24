import torch
import numpy as np

from attack_models.autoencoders import *
from attack_models.unet import *

import copy

from data_loader import *
from utils.text_load import *
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters
def test_reddit_normal(args, reddit_data_dict, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    batch_size = args.bs
    bptt = 64
    test_data = reddit_data_dict['test_data']

    hidden = model.init_hidden(batch_size)
    random_print_output_batch = \
    random.sample(range(0, (test_data.size(0) // bptt) - 1), 1)[0]
    data_iterator = range(0, test_data.size(0)-1, bptt)
    dataset_size = len(test_data)
    n_tokens = reddit_data_dict['n_tokens']

    for batch_id, batch in enumerate(data_iterator):
        data, targets = get_batch(test_data, batch)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, n_tokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        pred = output_flat.data.max(1)[1]
        correct += pred.eq(targets.data).sum().to(dtype=torch.float)
        total_test_words += targets.data.shape[0]

    acc = 100.0 * (correct / total_test_words)
    total_l = total_loss.item() / (dataset_size-1)

    model.train()
    return total_l, acc

def test_reddit_poison(args, reddit_data_dict, model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    bptt = 64
    batch_size = args.bs
    test_data_poison = reddit_data_dict['test_data_poison']
    ntokens = reddit_data_dict['n_tokens']
    hidden = model.init_hidden(batch_size)
    data_iterator = range(0, test_data_poison.size(0) - 1, bptt)
    dataset_size = len(test_data_poison)


    for batch_id, batch in enumerate(data_iterator):
        data, targets = get_batch(test_data_poison, batch)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
        hidden = repackage_hidden(hidden)

        ### Look only at predictions for the last words.
        # For tensor [640] we look at last 10, as we flattened the vector [64,10] to 640
        # example, where we want to check for last line (b,d,f)
        # a c e   -> a c e b d f
        # b d f
        pred = output_flat.data.max(1)[1][-batch_size:]


        correct_output = targets.data[-batch_size:]
        correct += pred.eq(correct_output).sum()
        total_test_words += batch_size

    acc = 100.0 * (correct / total_test_words)
    total_l = total_loss.item() / dataset_size

    model.train()
    return total_l, acc

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
                if not(trigger_generator.shape[-1] == inputs.shape[-1] and trigger_generator.shape[-2] == inputs.shape[-2]):
                    filled_zero_shape = (0, inputs.shape[-1] - trigger_generator.shape[-1], 0, inputs.shape[-2] - trigger_generator.shape[-2])
                    processed_noise_vector = nn.functional.pad(trigger_generator, filled_zero_shape, 'constant', 0)
                    inputs = torch.clamp(inputs + processed_noise_vector, 0.0, 1.0)
                else:
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

def model_dist_norm_var(model, target_params_variables, norm=2):
    sum_var = parameters_to_vector(model.parameters()) - target_params_variables
    return torch.norm(sum_var, norm)

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
            poisoned_img = add_pattern_bd(copy.deepcopy(img), args.data, args.pattern_type, -1, args.attack_mode, True, args)
        
        elif args.attack_mode == 'trigger_generation':
            img = img.unsqueeze(0).to(device=args.device)
            poisoned_img = trigger_model_target(img)
        
        elif args.attack_mode == 'fixed_generator':
            img = img.unsqueeze(0).to(device=args.device)
            if not(trigger_model_target.shape[-1] == img.shape[-1] and trigger_model_target.shape[-2] == img.shape[-2]):
                filled_zero_shape = (0, img.shape[-1] - trigger_model_target.shape[-1], 0, img.shape[-2] - trigger_model_target.shape[-2])
                processed_noise_vector = nn.functional.pad(trigger_model_target, filled_zero_shape, 'constant', 0)
                poisoned_img = torch.clamp(img + processed_noise_vector, 0.0, 1.0)
            else:
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
    plt.show()
    #plt.savefig('./src/running_data/images_compare/{round}.png'.format(round = round))
    plt.close()

def print_exp_details(args, record = None):
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
    print(f'    pattern_type: {args.pattern_type}')
    print(f'    pattern_size: {args.pattern_size}')
    print(f'    pattern_location: {args.pattern_location}')
    print(f'    malicious_style: {args.malicious_style}')
    print(f'    partition: {args.partition}')
    print(f'    underwater_attacker: {args.underwater_attacker}')
    print(f'    save_checkpoint: {args.save_checkpoint}')
    print('======================================')
    if record != None:
        record.append('======================================')
        record.append(f'    Dataset: {args.data}')
        record.append(f'    Global Rounds: {args.rounds}')
        record.append(f'    Aggregation Function: {args.aggr}')
        record.append(f'    Number of agents: {args.num_agents}')
        record.append(f'    Fraction of agents: {args.agent_frac}')
        record.append(f'    Batch size: {args.bs}')
        record.append(f'    Client_LR: {args.client_lr}')
        record.append(f'    Server_LR: {args.server_lr}')
        record.append(f'    Client_Momentum: {args.client_moment}')
        record.append(f'    RobustLR_threshold: {args.robustLR_threshold}')
        record.append(f'    Noise Ratio: {args.noise}')
        record.append(f'    Number of corrupt agents: {args.num_corrupt}')
        record.append(f'    Poison Frac: {args.poison_frac}')
        record.append(f'    Clip: {args.clip}')
        record.append(f'    restrain lr: {args.restrain_lr}')
        record.append(f'    pattern_type: {args.pattern_type}')
        record.append(f'    pattern_size: {args.pattern_size}')
        record.append(f'    pattern_location: {args.pattern_location}')
        record.append(f'    malicious_style: {args.malicious_style}')
        record.append(f'    partition: {args.partition}')
        record.append(f'    underwater_attacker: {args.underwater_attacker}')
        record.append(f'    save_checkpoint: {args.save_checkpoint}')
        record.append(f'======================================')
        
def print_distribution(user_groups, num_classes, train_dataset):
    print('======================================')
    for i in range(len(user_groups)):
        print('client {id}, data amount is {amount}'.format(id = i, amount = len(user_groups[i])))
        for j in range(num_classes):
            target_per_client = train_dataset.targets[user_groups[i]]
            print('index:{} number:{}'.format(j, torch.numel
            (target_per_client[target_per_client == j])))
    print('======================================')