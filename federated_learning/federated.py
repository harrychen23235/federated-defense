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
import pickle
import torch

def th_delete(tensor, indices):
    mask = torch.zeros(tensor.numel(), dtype=torch.bool)
    for i in indices:
        mask[i] = True
    return tensor[mask]

def parameter_choice(para_cov1, para_cov2, para_fc1, para_fc2, fraction):
    length = 1000
    cov1_length = int(fraction[0]/sum(fraction)*length)
    cov2_length = int(fraction[1]/sum(fraction)*length)
    fc1_length = int(fraction[2]/sum(fraction)*length)
    fc2_length = int(fraction[3]/sum(fraction)*length)

    final_list = []
    final_list.append(np.random.choice(len(para_cov1), cov1_length, replace = False))
    final_list.append(np.random.choice(len(para_cov2), cov2_length, replace = False))
    final_list.append(np.random.choice(len(para_fc1), fc1_length, replace = False))
    final_list.append(np.random.choice(len(para_fc2), fc2_length, replace = False))

    return final_list

def get_weight_tensor(model, index_list):
    cov1_picked = th_delete(parameters_to_vector(model.conv1.weight.detach()).clone().detach(), index_list[0])
    cov2_picked = th_delete(parameters_to_vector(model.conv2.weight.detach()).clone().detach(), index_list[1])
    fc1_picked = th_delete(parameters_to_vector(model.fc1.weight.detach()).clone().detach(), index_list[2])
    fc2_picked = th_delete(parameters_to_vector(model.fc2.weight.detach()).clone().detach(), index_list[3])
    cat_list = [cov1_picked, cov2_picked, fc1_picked, fc2_picked]

    return torch.cat(cat_list)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    utils.print_exp_details(args)
    
    # data recorders
    file_name = 'hello'
    writer = SummaryWriter('logs/' + file_name)
    cum_poison_acc_mean = 0
        
    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    # fedemnist is handled differently as it doesn't come with pytorch
    if args.data != 'fedemnist':
        user_groups = utils.distribute_data(train_dataset, args)
    
    # poison the validation dataset
    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)                                        
    
    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.data == 'fedemnist': 
            agent = Agent(_id, args)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent) 
        
    # aggregation server and the loss function
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, writer)
    criterion = nn.CrossEntropyLoss().to(args.device)

    export_para_correct = []
    export_para_corrupt = []
    
    export_gradient_correct = []
    export_gradient_corrupt = []

    for i in range(args.num_corrupt):
        export_gradient_corrupt.append([])

    for i in range(args.num_corrupt, args.num_agents):
        export_gradient_correct.append([])
    

    
    # training loop

    para_cov1 = parameters_to_vector(global_model.conv1.weight.detach())
    para_cov2 = parameters_to_vector(global_model.conv2.weight.detach())
    para_fc1 = parameters_to_vector(global_model.fc1.weight.detach())
    para_fc2 = parameters_to_vector(global_model.fc2.weight.detach())
    len_cov1 = int(math.log(len(para_cov1)))
    len_cov2 = int(math.log(len(para_cov2)))
    len_fc1 = int(math.log(len(para_fc1)))
    len_fc2 = int(math.log(len(para_fc2)))
    print('len of cov1 is:')
    print(len(para_cov1))
    print(len_cov1)
    print('len of cov2 is:')
    print(len(para_cov2))
    print(len_cov2)
    print('len of fc1 is:')
    print(len(para_fc1))
    print(len_fc1)
    print('len of fc2 is:')
    print(len(para_fc2))
    print(len_fc2)

    fraction = [len_cov1, len_cov2, len_fc1, len_fc2]
    len_param = len(parameters_to_vector(global_model.parameters()).detach())
    print('length of all parameters is')
    print(len_param)
    print('shape of all parameters:')
    for i in global_model.parameters():
        print(i.shape)
    chosen_parameter = np.random.choice(len_param, 1000, replace = False)
    index_list = parameter_choice(para_cov1, para_cov2, para_fc1, para_fc2, fraction)
    
    previous_round_set_1000 = set()
    previous_conv1_overall_set = set()
    previous_conv2_overall_set = set()
    previous_fc1_overall_set = set()
    previous_fc2_overall_set = set()
    all_round_set = []
    
    temp_1 = 32*1*3*3 + 32
    temp_2 = temp_1 + 64*32*3*3 + 64
    temp_3 = temp_2 + 128*9216 + 128
    temp_4 = temp_3 + 10*128 + 10
    layer_para_number = [temp_1, temp_2, temp_3, temp_4]

    all_train_update = []
    all_train_parameter = []

    for rnd in tqdm(range(1, args.rounds + 1)):
        all_train_update.append([])
        all_train_parameter.append([])

        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        export_para_correct.append([])
        export_para_corrupt.append([])
        agent_updates_dict = {}

        previous_agent_set_1000 = set()
        previous_conv1_set = set()
        previous_conv2_set = set()
        previous_fc1_set = set()
        previous_fc2_set = set()


        set_for_malic = set()
        set_for_benign_1 = set()
        set_for_benign_2 = set()
        set_for_benchmark = set()

        agent_common = []
        agent_conv1_common = []
        agent_conv2_common = []
        agent_fc1_common = []
        agent_fc2_common = []

        overall_update = None
        overall_conv1_update = None
        overall_conv2_update = None
        overall_fc1_update = None
        overall_fc2_update = None


        #for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents*args.agent_frac), replace=False):
        for agent_id in range(args.num_agents):
            #print('current agent is')
            #print(agent_id)
            update = agents[agent_id].local_train(global_model, criterion)

            if rnd in range(80,100):
                with open('./mali_data/all_train_update_{}_{}.pl'.format(rnd, agent_id), "wb") as fp:
                    pickle.dump(update.cpu().detach(), fp)

                with open('./mali_data/all_train_parameter_{}_{}.pl'.format(rnd, agent_id), "wb") as fp:
                    pickle.dump(parameters_to_vector(global_model.parameters()).cpu().detach(),fp)
            '''
            conv1_update = update[0: layer_para_number[0]]
            conv2_update = update[layer_para_number[0]: layer_para_number[1]]
            fc1_update = update[layer_para_number[1]: layer_para_number[2]]
            fc2_update = update[layer_para_number[2]: layer_para_number[3]]
            
            if agent_id == 0:
                overall_update = torch.abs(update)
                #overall_conv1_update = torch.abs(conv1_update)
                #overall_conv2_update = torch.abs(conv2_update)
                #overall_fc1_update = torch.abs(fc1_update)
                #overall_fc2_update = torch.abs(fc2_update)
                #overall_update = torch.abs(parameters_to_vector(global_model.parameters()).clone().detach())
            else:
                overall_update += torch.abs(update)
                #overall_conv1_update += torch.abs(conv1_update)
                #overall_conv2_update += torch.abs(conv2_update)
                #overall_fc1_update += torch.abs(fc1_update)
                #overall_fc2_update += torch.abs(fc2_update)
                #overall_update += torch.abs(parameters_to_vector(global_model.parameters()).clone().detach())
            
            abs_conv1_update = torch.abs(conv1_update)
            current_conv1_set = set(torch.topk(abs_conv1_update, 200).indices.tolist())
            agent_conv1_common.append(current_conv1_set)
            print('common of top 200 between conv1 set of {} and {} is'.format(agent_id - 1, agent_id))
            print(len(current_conv1_set & previous_conv1_set))
            previous_conv1_set = current_conv1_set

            abs_conv2_update = torch.abs(conv2_update)
            current_conv2_set = set(torch.topk(abs_conv2_update, 200).indices.tolist())
            agent_conv2_common.append(current_conv2_set)
            print('common of top 200 between conv2 set of {} and {} is'.format(agent_id - 1, agent_id))
            print(len(current_conv2_set & previous_conv2_set))
            previous_conv2_set = current_conv2_set

            abs_fc1_update = torch.abs(fc1_update)
            current_fc1_set = set(torch.topk(abs_fc1_update, 200).indices.tolist())
            agent_fc1_common.append(current_fc1_set)
            print('common of top 200 between fc1 set of {} and {} is'.format(agent_id - 1, agent_id))
            print(len(current_fc1_set & previous_fc1_set))
            previous_fc1_set = current_fc1_set

            abs_fc2_update = torch.abs(fc2_update)
            current_fc2_set = set(torch.topk(abs_fc2_update, 200).indices.tolist())
            agent_fc2_common.append(current_fc2_set)
            print('common of top 200 between fc2 set of {} and {} is'.format(agent_id - 1, agent_id))
            print(len(current_fc2_set & previous_fc2_set))
            previous_fc2_set = current_fc2_set
            
            
            abs_update = torch.abs(update)
            #abs_update = torch.abs(parameters_to_vector(global_model.parameters()).clone().detach())
            current_set_1000 = set(torch.topk(abs_update, 1000).indices.tolist())
            agent_common.append(current_set_1000)
            print('common of top 1000 between set of {} and {} is'.format(agent_id - 1, agent_id))
            print(len(current_set_1000 & previous_agent_set_1000))
            previous_agent_set_1000 = current_set_1000
            '''
            '''
            if agent_id in range(0, 10):
                set_for_malic = set_for_malic | current_set_1000
            elif agent_id in range(11, 20):
                set_for_benign_1 = set_for_benign_1 | current_set_1000
            elif agent_id in range(21, 30):
                set_for_benign_2 = set_for_benign_2 | current_set_1000
            elif agent_id in range(31, 40):
                set_for_benchmark= set_for_benchmark | current_set_1000
            '''
            '''
            picked_tensor = get_weight_tensor(global_model,index_list)
            #picked_tensor = th_delete(parameters_to_vector(global_model.parameters()).clone().detach(), chosen_parameter)
            picked_gradient = th_delete(update.clone().detach(), chosen_parameter)

            if agent_id < args.num_corrupt:
                export_gradient_corrupt[agent_id].append(picked_gradient)
                export_para_corrupt[rnd - 1].append(picked_tensor)

            else:
                export_gradient_correct[agent_id - args.num_corrupt].append(picked_gradient)
                export_para_correct[rnd - 1].append(picked_tensor)
            '''
            agent_updates_dict[agent_id] = update
            # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        # aggregate params obtained by agents and update the global params
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)

        '''
        print('common between malic and benchmark is')
        print(len(set_for_malic & set_for_benchmark))
        
        print('common between benign1 and benchmark is')
        print(len(set_for_benign_1 & set_for_benchmark))

        print('common between benign2 and benchmark is')
        print(len(set_for_benign_2 & set_for_benchmark))
        
        current_round_set_1000 = set(torch.topk(overall_update, 1000).indices.tolist())

        temp_count = [0,0,0,0]
        for index in current_round_set_1000:
            for layer_num in range(len(layer_para_number)):
                if index < layer_para_number[layer_num]:
                    temp_count[layer_num] += 1
                    break

        print("top number of first conv layer is")
        print(temp_count[0])
        print("top number of second conv layer is")
        print(temp_count[1])
        print("top number of first fc layer is")
        print(temp_count[2])
        print("top number of second fc layer is")
        print(temp_count[3])
        
        if rnd % 5 == 0:
            print('print out the value of top measure set of overall top 1000 parameters:')
            print(parameters_to_vector(global_model.parameters()).detach()[list(current_round_set_1000)])
            
        if rnd >= 5:
            measure_set = current_round_set_1000 & all_round_set[-1]&all_round_set[-2]& all_round_set[-3]&all_round_set[-4]
            print('common of top 1000 compared with previous 4 round is')
            print(len(measure_set))

        all_round_set.append(current_round_set_1000)
        
        
        current_round_set_conv1 = set(torch.topk(overall_conv1_update, 200).indices.tolist())
        print('common of top 200 of conv1 between round {} and {} is'.format(rnd, rnd - 1))
        print(len(current_round_set_conv1 & previous_conv1_overall_set))
        previous_conv1_overall_set = current_round_set_conv1


        current_round_set_conv2 = set(torch.topk(overall_conv2_update, 200).indices.tolist())
        print('common of top 200 of conv2 between round {} and {} is'.format(rnd, rnd - 1))
        print(len(current_round_set_conv2 & previous_conv2_overall_set))
        previous_conv2_overall_set = current_round_set_conv2

        current_round_set_fc1 = set(torch.topk(overall_fc1_update, 200).indices.tolist())
        print('common of top 200 of fc1 between round {} and {} is'.format(rnd, rnd - 1))
        print(len(current_round_set_fc1 & previous_fc1_overall_set))
        previous_fc1_overall_set = current_round_set_fc1


        current_round_set_fc2 = set(torch.topk(overall_fc2_update, 200).indices.tolist())
        print('common of top 200 of fc2 between round {} and {} is'.format(rnd, rnd - 1))
        print(len(current_round_set_fc2 & previous_fc2_overall_set))
        previous_fc2_overall_set = current_round_set_fc2 

        if rnd % 10 == 0:
            print('print out the value of top measure set of overall conv1 update:')
            print(overall_conv1_update[list(current_round_set_conv1)])
            print('print out the value of top measure set of overall conv2 update:')
            print(overall_conv2_update[list(current_round_set_conv2)])
            print('print out the value of top measure set of overall fc1 update:')
            print(overall_fc1_update[list(current_round_set_fc1)])
            print('print out the value of top measure set of overall fc2 update:')
            print(overall_fc2_update[list(current_round_set_fc2)])

        temp_agent_set = agent_conv1_common[0]
        for i in range(1, args.num_agents):
            temp_agent_set = temp_agent_set & agent_conv1_common[i]
        print('intesection between all agents of conv1 is')
        print(len(temp_agent_set))

        temp_agent_set = agent_conv2_common[0]
        for i in range(1, args.num_agents):
            temp_agent_set = temp_agent_set & agent_conv2_common[i]
        print('intesection between all agents of conv2 is')
        print(len(temp_agent_set))

        temp_agent_set = agent_fc1_common[0]
        for i in range(1, args.num_agents):
            temp_agent_set = temp_agent_set & agent_fc1_common[i]
        print('intesection between all agents of fc1 is')
        print(len(temp_agent_set))

        temp_agent_set = agent_fc2_common[0]
        for i in range(1, args.num_agents):
            temp_agent_set = temp_agent_set & agent_fc2_common[i]
        print('intesection between all agents of fc2 is')
        print(len(temp_agent_set))
        '''
        # inference in every args.snap rounds
        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
            
                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd) 
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')





                
    print('Training has finished!')
   

    
    
    
      
              