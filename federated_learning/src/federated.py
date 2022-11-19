import torch 
import utils
import data_loader
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
import os

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    #os.chdir('C://Users//harrychen23235//Desktop//report//security//federated-defense//federated_learning')
    args = args_parser()

    '''
    args.data = 'mnist'
    args.num_agents=10
    args.rounds=2
    args.partition = 'homo'
    args.load_pretrained = True
    args.pretrained_path = '..//data//saved_models//mnist_pretrain//model_last.pt.tar.epoch_10'
    #args.pretrained_path = '..//data//saved_models//cifar_pretrain//model_last.pt.tar.epoch_200'
    args.attack_mode = 'normal'
    args.num_corrupt = 4
    args.poison_frac = 0.2
    args.malicious_style='mixed'
    args.attack_start_round = 1
    args.storing_dir = './pattern_size_2'
    args.pattern_type = "size_test"
    args.pattern_size = 20
    #args.aggr = 'krum'
    #args.poison_mode = 'all2one'
    #args.pattern_type = 'vertical_line'
    #args.noise_total_epoch = 2
    #args.noise_sub_epoch = 1
    #args.trigger_training = 'both'
    '''

    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    test_accuracy_record = []
    utils.print_exp_details(args, test_accuracy_record)
    
    if not os.path.exists(args.storing_dir):
        os.mkdir(args.storing_dir)

    # data recorders
    file_name = f"""clip_val-{args.clip}-noise_std-{args.noise}"""\
            + f"""-aggr-{args.aggr}-s_lr-{args.server_lr}-num_cor-{args.num_corrupt}"""\
            + f"""thrs_robustLR-{args.robustLR_threshold}"""\
            + f"""-num_corrupt-{args.num_corrupt}-pttrn-{args.pattern_type}"""
    writer = SummaryWriter('logs/' + file_name)
    cum_poison_acc_mean = 0
        
    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = data_loader.get_datasets(args)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    user_groups = data_loader.distribute_data(train_dataset, args)
    utils.print_distribution(user_groups, args.num_classes, train_dataset)
    # poison the validation dataset

    poisoned_val_set = data_loader.Dataset_FL(copy.deepcopy(val_dataset), None, args, -1)

    # initialize a model, and the agents
    global_model = data_loader.get_classification_model(args).to(args.device)
    trigger_model_using = data_loader.get_noise_generator(args).to(args.device)
    trigger_model_target = data_loader.get_noise_generator(args).to(args.device)
    trigger_vector = data_loader.get_noise_vector(args)

    print('******global model is: {current_model}******'.format(current_model = type(global_model)))
    agents, agent_data_sizes = [], {}

    for _id in range(0, args.num_agents):
        agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent) 
        
    # aggregation server and the loss function
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(agent_data_sizes, n_model_params, args, writer)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # training loop
    for rnd in tqdm(range(1, args.rounds+1)):
        if args.restrain_lr and rnd % 10 == 0:
            args.client_lr = args.client_lr * 0.5
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents*args.agent_frac), replace=False):
            update = agents[agent_id].local_train(global_model, criterion, rnd, [trigger_model_using, trigger_model_target, trigger_vector])
            if rnd >= args.attack_start_round:
                torch.save(update, os.path.join(args.storing_dir, '{}_update.pt'.format(agent_id)))
            agent_updates_dict[agent_id] = update
            # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        # aggregate params obtained by agents and update the global params
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)

        
        # inference in every args.snap rounds
        if rnd % args.snap == 0:
            test_accuracy_record.append('current rnd is {}'.format(rnd))
            print(f'**** start testing ****')
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy_normal(global_model, criterion, val_loader, args, args.num_classes)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
                test_accuracy_record.append(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                if args.attack_mode == 'fixed_generator':
                    poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy_poison(global_model, trigger_vector, criterion,  poisoned_val_set, args, args.num_classes)
                elif args.attack_mode == 'trigger_generation':
                    poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy_poison(global_model, trigger_model_target, criterion,  poisoned_val_set, args, args.num_classes)
                else:
                    poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy_poison(global_model, None, criterion,  poisoned_val_set, args, args.num_classes)

                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                test_accuracy_record.append(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                '''
                if args.num_corrupt > 0:
                    if args.attack_mode == 'fixed_generator':
                        utils.compare_images(trigger_vector, poisoned_val_set, args, rnd)
                    elif args.attack_mode == 'trigger_generation':
                        utils.compare_images(trigger_model_target, poisoned_val_set, args, rnd)
                    else:
                        utils.compare_images(None, poisoned_val_set, args, rnd)
                '''
    with open(os.path.join(args.storing_dir, 'accuracy_record.txt'), 'w') as f:
        for line in test_accuracy_record:
            f.write(line)
            f.write('\n')
    
    f.close()

    print('Training has finished!')
   

    
    
    
      
              