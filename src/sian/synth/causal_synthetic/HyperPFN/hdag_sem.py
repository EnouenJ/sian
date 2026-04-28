

import numpy as np
import torch
import torch.nn as nn




from . import LinearFunction
from . import GaussianNoiseDistribution, StandardGumbelDistribution
from . import HDAG





#TODO: NEED TO UPGRADE THIS TO A PROBABILITY DISTRIBUTION OVER THE META CHOICES
def select_random_from_list(mylist):
    L = len(mylist)
    l = np.random.randint(L)
    return mylist[l]


from . import barabosi, barabosi_hyper #TODO - NEED TO MOVE THIS OUT ASAP

class Doable_HDAG_SEM():
    

    def __init__(self, full_meta_hyperparameters, graph_metainit_seed=None, sem_metainit_seed=None):
    
        self.D = None
        self.hyperparents = None
        self.vars = None
        self.topological_order = None

        self.functions = nn.ModuleDict()
        self.noises = nn.ModuleDict()

        VERBOSE_INIT = True
        
        if graph_metainit_seed is None:
            graph_metainit_seed = 0 #NOTE: could also randomize here
            np.random.seed()
            graph_metainit_seed = np.random.randint(2**31-1)
            print('graph_metainit_seed',graph_metainit_seed)
        self.graph_metainit_seed = graph_metainit_seed
        if sem_metainit_seed is None:
            sem_metainit_seed = 0 #NOTE: could also randomize here
            np.random.seed()
            sem_metainit_seed = np.random.randint(2**31-1)
            print('sem_metainit_seed',sem_metainit_seed)
        self.sem_metainit_seed = sem_metainit_seed


        #################'
        graph_generation_meta_hypers = full_meta_hyperparameters["graph_generation"]
        self.hdag = HDAG(graph_generation_meta_hypers, metainit_seed=graph_metainit_seed)
        self.D = self.hdag.D
        D = self.D
        self.hyperparents = self.hdag.hyperparents
        if VERBOSE_INIT:
            print('hyperparents')
            for j in range(D):
                print(j,':',self.hyperparents[j])
        #################


        np.random.seed(self.sem_metainit_seed)        
        SEM_generation_meta_hypers = full_meta_hyperparameters["SEM_generation"]    
        vars = {}
        for j in range(D):
            cts = (np.random.rand() < SEM_generation_meta_hypers["cts_dic_ratio"])
            if cts:
                vars[j] = ('cts', 1,)
            else:
                Ik = np.random.randint(*SEM_generation_meta_hypers["min_max_discrete_categories"])
                vars[j] = ('disc.onehot', Ik,)
        self.vars = vars
        topological_order = list(range(D))
        self.topological_order = topological_order
                
            


        for j in topological_order:
            var_data_j = self.vars[j]
            var_data_type_j = var_data_j[0]
            hyperparents_j = self.hyperparents[j]

            
            if var_data_type_j=="cts":
                generator_details_j = select_random_from_list(SEM_generation_meta_hypers["cts_function_options_list"])
                noise_details_j = select_random_from_list(SEM_generation_meta_hypers["cts_noise_options_list"])
            elif var_data_type_j=="disc.onehot":
                generator_details_j = select_random_from_list(SEM_generation_meta_hypers["disc_function_options_list"])
                noise_details_j = select_random_from_list(SEM_generation_meta_hypers["disc_noise_options_list"])
            else:
                raise NotImplementedError(f"var_data_type_j={var_data_type_j}")
                
            
            
            self.functions[str(j)] = nn.ModuleDict()
            function_type_j = generator_details_j['function_type']
            for subparents in hyperparents_j:
                subparent_dimensions = [vars[s][1] for s in subparents]
                child_dim = var_data_j[1]

                myfunction = None
                if function_type_j=='multilinear':
                    coef_dist_info = generator_details_j['coef_dist_info']
                    myfunction = LinearFunction(dim_list=subparent_dimensions, d_out=child_dim, coef_dist_info=coef_dist_info)
                else:
                    raise NotImplementedError(f"function_type_j={function_type_j}")

                self.functions[str(j)][str(subparents)] = myfunction

        
            noise_type_j = noise_details_j['noise_type']
            mynoisefunction = None
            if noise_type_j=='gaussian':
                std_dist_info = noise_details_j['std_dist_info']
                mynoisefunction = GaussianNoiseDistribution(std_dist_info)
            elif noise_type_j=='disc.gumbel':
                Ik = var_data_j[1]
                mynoisefunction = StandardGumbelDistribution(Ik)
            else:
                raise NotImplementedError("")
            self.noises[str(j)] = mynoisefunction
        





    #TODO: add metainit and datagen seed
    def generate_data(self, N, datagen_seed=None):

        if datagen_seed is None:
            np.random.seed()
            datagen_seed = np.random.randint(2**31-1)
            print('datagen_seed',datagen_seed)
        self.datagen_seed = datagen_seed

        X_dict = {}

        topological_order = self.topological_order
        if topological_order is None:
            topological_order = list(range( len(self.nodes) ))

        for j in topological_order:
            # print(j)
            var_data_j = self.vars[j]
            # print(var_data_j)

            var_type_j = var_data_j[0]
            hyperparents_j = self.hyperparents[j]
            preX = 0
            for subparents in hyperparents_j:
                X_subparents = torch.concatenate(   [torch.Tensor(X_dict[s]) for s in subparents],  dim=-1  )
                # print('subparents',subparents)
                # print('X_subparents',X_subparents.shape)
                myfunction = self.functions[str(j)][str(subparents)]
                preX += myfunction(X_subparents).detach().cpu().numpy()

            mynoisefunction = self.noises[str(j)]
            preEps = mynoisefunction.generate_noise(N)
            
            X = 0
            if var_type_j=='cts':
                X = preX + preEps
            elif var_type_j=='disc.onehot':
                logitX = preX + preEps
                intX = np.argmax(logitX,axis=1)
                X = np.zeros_like(logitX)
                X[np.arange(N),intX] = 1
            else:
                raise NotImplementedError("")

            X_dict[j] = X

        return X_dict


    def special_generate_data(self):
        raise NotImplementedError("no X,Y,Z,W, A yet")
