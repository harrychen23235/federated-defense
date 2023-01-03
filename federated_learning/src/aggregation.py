import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from defence import *
class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params

        self.cum_net_mov = 0
        
         
    def aggregate_updates(self, global_model, agent_updates_dict, cur_round):
        # adjust LR if robust LR is selected
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates_dict)
        
        
        aggregated_updates = 0

        if self.args.clip != 0:
            self.clip_updates(agent_updates_dict)
            
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr=='comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)
        elif self.args.aggr == 'krum':
            aggregated_updates = self.multi_krum(agent_updates_dict)
        elif self.args.aggr == 'flame':
            aggregated_updates = self.agg_flame(agent_updates_dict)
        if self.args.noise > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=self.args.noise*self.args.clip, size=(self.n_params,)).to(self.args.device))
        
                
        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        vector_to_parameters(new_global_params, global_model.parameters())
        
        # some plotting stuff if desired
        # self.plot_sign_agreement(lr_vector, cur_global_params, new_global_params, cur_round)
        # self.plot_norms(agent_updates_dict, cur_round)
        return           
     
    
    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr                                            
        return sm_of_signs.to(self.args.device)
        
    def multi_krum(self, agent_updates_dict):
        selected_number = self.args.krum_selected_number
        tolerance_number = self.args.krum_tolerance_number
        update_len = len(agent_updates_dict.keys())
        #aggregation method is averaging in this case
        if selected_number >= update_len:
            return self.agg_avg(self, agent_updates_dict)
        else:
            # Compute list of scores
            scores = [list() for i in range(update_len)]
            for i in range(update_len - 1):
                score = scores[i]
                for j in range(i + 1, update_len):
                     # With: 0 <= i < j < nbworkers
                    distance = torch.dist(agent_updates_dict[i], agent_updates_dict[j]).item()
                    #if distance == float('nan'):
                        #distance = float('inf')
                    score.append(distance)
                    scores[j].append(distance)
            nbinscore = update_len - tolerance_number - 2
            for i in range(update_len):
                score = scores[i]
                score.sort()
                scores[i] = sum(score[:nbinscore])
            # Return the average of the m gradients with the smallest score
            pairs = [(agent_updates_dict[i], scores[i]) for i in range(update_len)]
            pairs.sort(key=lambda pair: pair[1])
            result = pairs[0][0]
            for i in range(1, selected_number):
                result += pairs[i][0]
            result /= float(selected_number)
            return result

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            if self.args.data != 'reddit':
                n_agent_data = self.agent_data_sizes[_id]
            else:
                n_agent_data = 1
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    def agg_flame(self, agent_updates_dict):
        """ fed avg with flame """
        update_len = len(agent_updates_dict.keys())
        weights = np.zeros((update_len, np.array(len(agent_updates_dict[0]))))
        for _id, update in agent_updates_dict.items():
            weights[_id] = update.cpu().detach().numpy()  # np.array
        # grad_in = weights.tolist()  #list
        benign_id = flame(weights, cluster_sel=0)
        print('!!!FLAME: remained ids are:')
        print(benign_id)
        accepted_models_dict = {}
        for i in range(len(benign_id)):
            accepted_models_dict[i] = torch.tensor(weights[benign_id[i], :]).to(self.args.device)
        sm_updates, total_data = 0, 0
        for _id, update in accepted_models_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2) 
            update.div_(max(1, l2_update/self.args.clip))
        return
                  
    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])
                              
        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)
        
        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round) 
        return


        