
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import torch

import random
import json

from ..synth import DataGenerator



try:
    import igraph as ig
except:
    #TODO
    print("igraph not installed, using XXX instead")








#TODO TODO TODO


# class InputDataGeneratorClass():
#     def __init__(self, D):

#         self.D = D
#         self.generator_parameters = {}
        
#         pass

#     def generate_samples(self, N):

#         X_dict = {}
        
#         #TODO: HOW TO MAKE AN ABSTRACT METHOD  

#         return X_dict




# class OutputDataFunctionClass():
#     def __init__(self, D):
#         pass
#         self.D = D

#         #NOTE: probably needs more of a signature here

#     def generate_output_samples(self, X):
#         PLOTTING = False
#         PLOTTING = True

#         #TODO: HOW TO MAKE AN ABSTRACT METHOD  

#         return output

# #TODO TODO actually use these constructors ^^^







# https://github.com/xunzheng/notears/blob/master/notears/utils.py
def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges."""
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        ###plt.imshow(B_und);plt.show();
        B = _random_acyclic_orientation(B_und)
        ###plt.imshow(B);plt.show();
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    ###return B_perm
    return B #Jam - I can shuffle later after knowing how to generate nonlinear SEM



def jsonify_beta_dict(next_beta_dict):
    new_dict = []
    for tup in next_beta_dict:
        new_dict.append( [   [[str(i) for i in tup]]  ,  next_beta_dict[tup]   ] )
    return new_dict
    
def jsonify_beta_layer(next_beta_layer):
    next_beta_json = {}
    for Dj in next_beta_layer:
        next_layer_j = next_beta_layer[Dj]
        next_layer_j_json = {}
        for beta_dict_str in next_layer_j:
            next_layer_j_json[beta_dict_str] = jsonify_beta_dict(next_layer_j[beta_dict_str])

        next_beta_json[str(Dj)] = next_layer_j_json
    return next_beta_json




#TODO -- use this or use NOTEARS one, make a decision (note there is the difference, especially the IG choice or not)

# def simulate_dag(d, s0, graph_type):
#     def _random_permutation(M):
#         P = np.random.permutation(np.eye(M.shape[0]))
#         return P.T @ M @ P
#     def _random_acyclic_orientation(B_und):
#         return np.tril(_random_permutation(B_und), k=-1)
    
#     if graph_type == 'ER':
#         B_und=np.zeros((d,d))
#         perm = np.random.permutation(d*(d-1))
#         xd=[(i,j) for i in range(d) for j in range(d)]
#         [xd.remove((i,i)) for i in range(d)]
#         xdd=[xd[p] for p in perm[:s0]]
#         B_und[tuple(zip(*xdd))] = 1
#         B = _random_acyclic_orientation(B_und)
#     else:
#         raise ValueError('unknown graph type')
#     return B
    
def numpy_to_json_array(numpy_array):
    return numpy_array.tolist()

def jsonify_theta_dict(next_theta_dict):
    new_dict = []
    for tup in next_theta_dict:
        new_dict.append( [   [[str(i) for i in tup]]  ,  numpy_to_json_array(next_theta_dict[tup])   ] )
    return new_dict
    
def jsonify_theta_layer(next_theta_layer):
    next_theta_json = {}
    for Dj in next_theta_layer:
        next_layer_j = next_theta_layer[Dj]
        next_layer_j_json = {}
        for theta_dict_str in next_layer_j:
            next_layer_j_json[theta_dict_str] = jsonify_theta_dict(next_layer_j[theta_dict_str])

        next_theta_json[str(Dj)] = next_layer_j_json
    return next_theta_json

















# class HCAM_MultilinearSEM_ERgraphSCM_InputDataGenerator():
class HCAM_MultilinearSEM_ERgraphSCM_InputDataGenerator(DataGenerator):
    
    def __init__(self, D, GAM_degree, ER_degree, metainit_seed=None, datagen_seed=None):
        super().__init__(metainit_seed, datagen_seed)

        coefficient_spread_factor = 2.0
        variance_spread_factor = 2.0

        self.D = D
        self.generator_meta_parameters = {
            'GAM_degree' : GAM_degree,
            'ER_degree'  : ER_degree,
            
            'coefficient_spread_factor' : coefficient_spread_factor,
            'variance_spread_factor' : variance_spread_factor,
        }
        self.generator_parameters = {}

        B = simulate_dag(D,D*ER_degree,"ER")
        B = np.array(B)
        self.generator_parameters["B_og"] = B

        # this version of HDAG ensures all hyperedges are exactly degree 2 or 3. 
        B_modified = np.copy(B)
        B_modified[np.sum(B_modified,axis=1)<GAM_degree] = 0 
        self.generator_parameters["B"] = B_modified


        parents = {}
        for i in range(D):
            parents[i] = []
            for j in range(D-1):
                if B_modified[i,j]==1.0:
                    parents[i].append(j)
            print(parents[i])
        self.generator_parameters["parents"] = parents




        SNR_low,SNR_high = -np.log(variance_spread_factor),np.log(variance_spread_factor)
        SNRs_30 = np.exp(SNR_low+(SNR_high-SNR_low)*np.random.rand(D))
        self.generator_parameters["SNRs_30"] = SNRs_30
        print("SNRs_30")
        print(SNRs_30)


        layer_beta_info = {}
        for i in range(D):
            Pi = len(parents[i])
            beta_dict = {}
            final_beta_dict = {}
            if Pi>=GAM_degree:
                parent_tuples = []
                parent_order = np.random.permutation(Pi)
                
                if GAM_degree==2:
                    coef_renorm = 1 / Pi / np.sqrt(3)
                    for p1 in range(Pi-GAM_degree):
                        p_tup = tuple([parent_order[pp] for pp in range(p1,p1+GAM_degree)])
                        parent_tuples.append( p_tup )
                elif GAM_degree==3:
                    coef_renorm = 1 / Pi / np.sqrt(15)
                    for p1 in range(Pi-GAM_degree):
                        p_tup = tuple([parent_order[pp] for pp in range(p1,p1+GAM_degree)])
                        parent_tuples.append( p_tup )
                else:
                    raise NotImplementedError(f"GAM_degree={GAM_degree}")

                for p_tup in parent_tuples:
                    i_tup = tuple([parents[i][pp] for pp in p_tup])
                    
                    coef_low,coef_high = -np.log(coefficient_spread_factor),np.log(coefficient_spread_factor)
                    coef = float((np.random.randint(2)*2-1)* np.exp(coef_low+(coef_high-coef_low)*np.random.rand(1)))
                    beta_dict[i_tup] = coef
                    coef = coef * coef_renorm
                    final_beta_dict[i_tup] = coef

                    
            layer_beta_info[i] = {
                'beta_dict' : beta_dict,
                'final_beta_dict' : final_beta_dict,
            }
            
        self.generator_parameters["layer_beta_info"] = layer_beta_info
            
    
    def save_details(self, file_path):
        #TODO TODO 
        # raise NotImplementedError(f"saving to json format -- should likely do external with \"Datset() constructor wrapper\" ")

        layer_beta_info = self.generator_parameters["layer_beta_info"]
        details_to_save = {
            "layer_beta_info" : jsonify_beta_layer(layer_beta_info),
        }
        with open(file_path+'details.json', 'w', encoding='utf-8') as f:
            json.dump(details_to_save, f, ensure_ascii=False, indent=4)

        np.save(file_path+'B_og.npy',   self.generator_parameters["B_og"])
        np.save(file_path+'B.npy',      self.generator_parameters["B"])
        np.save(file_path+'SNRs_30.npy',self.generator_parameters["SNRs_30"])




    def generate_samples(self, N, return_errors=False, datagen_seed=None):
        if datagen_seed is not None:
            pass
            # self.set_datagen_seed(datagen_seed)
        pass

        D = self.D
        parents = self.generator_parameters["parents"]
        layer_beta_info = self.generator_parameters["layer_beta_info"]
        SNRs_30 = self.generator_parameters["SNRs_30"]
        N30 = np.random.randn(N,D)
        X30 = np.zeros((N,D))


        for i in range(D):
            Pi = len(parents[i])
            final_beta_dict = layer_beta_info[i]['final_beta_dict']

            for i_tup in final_beta_dict:
                print(i_tup)
                coef = final_beta_dict[i_tup]

                thing=1.0
                for i2 in i_tup:
                    thing *= X30[:,i2]
                X30[:,i] += coef * thing
                
            X30[:,i] = X30[:,i] + np.sqrt(SNRs_30[i]) * N30[:,i]
            
        if return_errors:
            return X30, N30
        return X30

    def return_full_readable_labels(self):
        D = self.D
        full_readable_labels = {}
        for d in range(D):
            full_info_d = {
                "label" : "X"+str(d+1),
                "startdim" : d,
                "numdims" : 1,
                "encoding" : "cts.raw",
            }
            full_readable_labels[d] = full_info_d
        full_readable_labels["task_type"] = "generative_dataset"
        full_readable_labels["D0"] = D
        full_readable_labels["D"] = D
        return full_readable_labels







# class HCAM_DiscreteSEM_ERgraphSCM_InputDataGenerator():
class HCAM_DiscreteSEM_ERgraphSCM_InputDataGenerator(DataGenerator):
    
    def __init__(self, D,  I_k, GAM_degree, ER_degree, metainit_seed=None, datagen_seed=None):
        super().__init__(metainit_seed, datagen_seed)

        self.D = D
        self.I_k = I_k
        self.generator_meta_parameters = {
            'GAM_degree' : GAM_degree,
            'ER_degree'  : ER_degree,
        }
        self.generator_parameters = {}


        B = simulate_dag(D,D*ER_degree,"ER")
        B = np.array(B)
        self.generator_parameters["B_og"] = B

        # this version of HDAG ensures all hyperedges are exactly degree 2 or 3. 
        B_modified = np.copy(B)
        B_modified[np.sum(B_modified,axis=1)<GAM_degree] = 0 
        self.generator_parameters["B"] = B_modified


        parents = {}
        for i in range(D):
            parents[i] = []
            for j in range(D-1):
                if B_modified[i,j]==1.0:
                    parents[i].append(j)
            print(parents[i])
        self.generator_parameters["parents"] = parents

        #TODO: later distinguish between the original HCAM data generated in this way and the purified versions
        layer_theta_info = {}
        for i in range(D):
            Pi = len(parents[i])
            theta_dict = {}
            if Pi>=GAM_degree:
                parent_order = np.random.permutation(Pi)
                for p1 in range(Pi-GAM_degree+1):  #oops plus one
                    theta_logits = np.random.randn(*([I_k]*(GAM_degree+1)))
                    p_tup = tuple([parent_order[pp] for pp in range(p1,p1+GAM_degree)])
                    i_tup = tuple([parents[i][pp] for pp in p_tup])
                    theta_dict[i_tup] = theta_logits
            else:
                logits = np.random.randn(I_k)
                theta_dict[()] = logits

            layer_theta_info[i] = {
                'theta_dict' : theta_dict,
            }
        self.generator_parameters["layer_theta_info"] = layer_theta_info


    def save_details(self, file_path):
        #TODO TODO 
        # raise NotImplementedError(f"saving to json format -- should likely do external with \"Datset() constructor wrapper\" ")

        layer_theta_info = self.generator_parameters["layer_theta_info"]
        details_to_save = {
            "layer_theta_info" : jsonify_theta_layer(layer_theta_info),
        }
        with open(file_path+'details.json', 'w', encoding='utf-8') as f:
            json.dump(details_to_save, f, ensure_ascii=False, indent=4)

        np.save(file_path+'B_og.npy',   self.generator_parameters["B_og"])
        np.save(file_path+'B.npy',      self.generator_parameters["B"])



    def generate_samples(self, N, return_cond_probs=False, datagen_seed=None):
        if datagen_seed is not None:
            pass
            # self.set_datagen_seed(datagen_seed)
        pass

        if return_cond_probs:
            #return the u values as well
            raise NotImplementedError(f"not returning conditional probabilities unless I can think of a use case")

        D = self.D
        layer_theta_info = self.generator_parameters["layer_theta_info"]
        X30_dict = {}
    
        for i in layer_theta_info:
            theta_dict = layer_theta_info[i]['theta_dict']
            print(i)

            final_logits = 0.0
            for i_tup in theta_dict:
                print('\t',i_tup)
                X_i_tup = np.ones(N)

                for i2 in i_tup:
                    X_i_tup = X_i_tup[...,None]*X30_dict[i2][tuple([slice(None)]+[None]*(len(X_i_tup.shape)-1))]

                #NOTE: there ought to be a better version than this, right?
                alpha = "abcdefghijklmnopqrstuvwxyz"
                alpha2 = alpha[:len(i_tup)]
                final_logits += np.einsum("I"+alpha2+",N"+alpha2+"->NI",theta_dict[i_tup], X_i_tup)

            final_logits_shift = logsumexp(final_logits,axis=-1)
            final_probs = np.exp(final_logits - final_logits_shift[:,None])

            u_i = torch.Tensor(final_probs)
            Z_i = torch.zeros_like( u_i,dtype=int ).scatter_(1, torch.multinomial(u_i,1), 1.)
            X30_dict[i] = Z_i.numpy()
            
        return X30_dict


    def return_full_readable_labels(self):
        D = self.D
        I_k = self.I_k
        full_readable_labels = {}
        for d in range(D):
            full_info_d = {
                "label" : "X"+str(d+1),
                "startdim" : d*I_k,
                "numdims" : I_k,
                "encoding" : "disc.onehot",
            }
            full_readable_labels[d] = full_info_d
        full_readable_labels["task_type"] = "generative_dataset"
        full_readable_labels["D0"] = D
        full_readable_labels["D"] = D
        return full_readable_labels








