import pickle
import torch
import torch.nn
import random
import numpy as np

def compare_two_list(list_1, list_2):
    final_distance = 0
    for i in list_1:
        for j in list_2:
            final_distance += torch.dist(i,j)
    return final_distance/(len(list_1) * len(list_2))

def get_para_of_round(round_num):
    round_list = []
    for i in range(40):
        file_name = './src/mali_data/all_train_parameter_{}_{}.pl'.format(round_num, i)
        with open(file_name, "rb") as fp:   # Unpickling
            raw = pickle.load(fp)
            round_list.append(raw)
    return round_list


def get_gradient_of_update(round_num):
    round_list = []
    for i in range(40):
        file_name = './src/mali_data/all_train_update_{}_{}.pl'.format(round_num, i)
        with open(file_name, "rb") as fp:   # Unpickling
            raw = pickle.load(fp)
            round_list.append(raw)
    return round_list

def get_element_of_index(input_tensor, index):
    if(type(index) == set):
        index = list(index)
        index.sort()
    return input_tensor[index]

def split_para_to_layer(input_tensor):
    temp1 = 32*1*3*3 + 32
    temp2 = temp1 + 64*32*3*3 + 64
    temp3 = temp2 + 128*9216 + 128
    temp4 = temp3 + 10*128 + 10
    return [input_tensor[0:temp1],input_tensor[temp1:temp2], input_tensor[temp2:temp3], input_tensor[temp3:temp4]]

def get_topk(input_tensor, k):
    if k > len(input_tensor):
        k = len(input_tensor)
    abs_tensor = abs(input_tensor)
    index_list = torch.topk(abs_tensor, k).indices.tolist()
    #index_list = np.random.choice(len(abs_tensor), k, replace = False)
    return index_list

round_list = get_para_of_round(99)
layer_list = []
for agent in round_list:
    layer_list.append(split_para_to_layer(agent))
'''
for layer_num in range(4):
    layer_mali = []
    layer_be = []
    layer_bench = []
    for i in range(10):
        layer_mali.append(layer_list[i][layer_num])
        layer_be.append(layer_list[i+10][layer_num])
        layer_bench.append(layer_list[i+20][layer_num])
    print('distance between mali & bench on layer {} is'.format(layer_num))
    print(compare_two_list(layer_mali, layer_bench))
    print('distance between benign & bench on layer {} is'.format(layer_num))
    print(compare_two_list(layer_be, layer_bench))
'''

k_list = [1000,1000,1000,1000]
for layer_index in range(len(layer_list[0])):
    k = k_list[layer_index]
    current_layer = layer_list[0][layer_index]
    top_k_overall = set(get_topk(current_layer, k))

    for index in range(1, len(layer_list)):
        top_k_overall = top_k_overall & set(get_topk(layer_list[index][layer_index], k))
    
    print('len of top k overall for layer {}'.format(layer_index))
    print(len(top_k_overall))

for index in range(len(layer_list)):
    print('current comparing {} and {}'.format(index, index + 1))
    for layer_index in range(len(layer_list[index])):
        current_layer = layer_list[index][layer_index]
        previous_layer = layer_list[index - 1][layer_index]
        #print('current layer is')
        #print(current_layer)

        k = k_list[layer_index]
        current_topk = set(get_topk(current_layer, k))
        previous_topk = set(get_topk(previous_layer, k))

        #random_topk = list(np.random.choice(len(current_layer), k, replace = False)).sort()
        final_topk = current_topk & previous_topk

        print('size of merge set is')
        print(len(final_topk))

        #current_topk_layer_random = get_element_of_index(current_layer, random_topk)
        #previous_topk_layer_random = get_element_of_index(previous_layer, random_topk)

        current_topk_layer_merge = get_element_of_index(current_layer, final_topk)
        previous_topk_layer_merge = get_element_of_index(previous_layer, final_topk)
        print('current topk layer is')
        temp_list = list(torch.abs(current_topk_layer_merge).cpu().detach().numpy())
        temp_list.sort()
        print(temp_list)
        #print('element of topk of layer {} is'.format(layer_index))
        #print(get_element_of_index(current_layer, current_topk))
        #print('parameter of current layer is')
        #print(current_layer)
        #print('parameter of previous layer is')
        #print(previous_layer)
        print('distance between two layer of layer {} is'.format(layer_index))
        print(torch.dist(current_layer, previous_layer, 2))
        #print('distance between two topk layer of layer {} is(random_choose)'.format(layer_index))
        #print(torch.dist(current_topk_layer_random, previous_topk_layer_random, 2))
        print('distance between two topk layer of layer {} is(real)'.format(layer_index))
        print(torch.dist(current_topk_layer_merge, previous_topk_layer_merge, 2))       



