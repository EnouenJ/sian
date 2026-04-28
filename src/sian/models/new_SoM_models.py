import numpy as np
import matplotlib.pyplot as plt
import copy



import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader



from .smooth_sian_or_mnist_sian import MNIST_or_Smooth_SIAN_GAM
from .smooth_sian_or_mnist_sian import MNIST_or_Smooth_ReLU_DNN




class SmoothOrMnist_UnmaskedOrMasked_SIAN(nn.Module):
    def __init__(self,sizes,indices,  dnn_on_or_off=False, small_sizes=[0,16,12,8,1], feature_groups_dict=None,
                                      masking_mode="default_unmasked", parametrization_mode="smooth_parametrization", bias_configuration=None):
        super(SmoothOrMnist_UnmaskedOrMasked_SIAN,self).__init__()
        
        self.sizes = copy.deepcopy(sizes)
        small_sizes = copy.deepcopy(small_sizes)
        small_sizes[-1] = sizes[-1]
        
        subset_indexer = np.zeros((self.sizes[0],len(indices)))
        subset_indexer_const = np.zeros((1,len(indices)))
        
        self.feature_groups_dict = feature_groups_dict        
        
        self.indices = indices
        self.ungrouped_indices = indices
        if self.feature_groups_dict is not None:
            self.ungrouped_indices = []
            for ind in indices:
                new_ind = []
                for i in ind:
                    new_ind.extend(  self.feature_groups_dict[i]  )
                self.ungrouped_indices.append( tuple(new_ind) )

        self.gam = MNIST_or_Smooth_SIAN_GAM(sizes[0], self.ungrouped_indices, small_sizes=small_sizes, parametrization_mode=parametrization_mode, bias_configuration=bias_configuration)
        self.masking_mode = masking_mode # "default_unmasked"  or "insta_masked"

        self.precompute_off()
        self.value = 0

        for ii,ind in enumerate(indices):
            if self.feature_groups_dict is None:
                for i in ind:
                    subset_indexer[i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
                subset_indexer_const[0,ii] = 1 / (len(ind)+0.5)
            else:
                for i in ind:
                    sub_i = self.feature_groups_dict[i][0] #just use the first dimension (NOTE: assumes the full mask is initialized correctly)
                    subset_indexer[sub_i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
                subset_indexer_const[0,ii] = 1 / (len(ind)+0.5)

        self.subset_indexer_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer).float(), requires_grad=False )
        self.subset_indexer_const_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer_const).float(), requires_grad=False )

        self.debug_verbose = False #01/29/2025



    def forward(self, x):
        if self.masking_mode == "default_unmasked":
            return self.vanilla_forward(x)
        elif self.masking_mode == "insta_masked":
            return self.insta_masked_forward(x)
        else:
            raise NotImplementedError(f"self.masking_mode={self.masking_mode}")
        
    def vanilla_forward(self, x):
        gam_h, shape_loss = self.gam(x)
        dnn_h = torch.zeros_like(gam_h)
        if self.debug_verbose:
            print('SoM_Un_SIAN.forward()','dnn_h',dnn_h.shape,'gam_h',gam_h.shape,'shape_loss',shape_loss.shape)
        return dnn_h, gam_h, shape_loss

    def insta_masked_forward(self, xx):
        x,S = xx
        x = x * S + self.value * (1-S)

        if self.precomputed_shapes:
            all_shapes = self.all_shapes
        else:
            all_shapes = self.gam.forward_shapes(x)

        shape_fn_mask = (torch.matmul(S.float(),self.subset_indexer_tensor)+self.subset_indexer_const_tensor>1).float()
        shape_fn_mask = shape_fn_mask[:,:,None]
        
        if self.gam.biases_on:
            all_shapes = all_shapes + (self.gam.biases_scaling*self.gam.biases)[None]
        if self.gam.bias_on:
            all_shapes = all_shapes + (self.gam.bias_scaling*self.gam.bias)[None,None]
        
        all_shapes = all_shapes*shape_fn_mask
        if self.debug_verbose:
            print(all_shapes)

        gam_h = torch.sum(all_shapes, dim=1)
        dnn_h = torch.zeros_like(gam_h)

        shape_loss = torch.zeros(len(self.indices), device=x.device)
        if self.gam.compute_shapeloss_while_training:
            # shape_loss = torch.mean(  torch.sum( torch.abs(all_shapes),dim=2),  dim=0)
            shape_loss = torch.mean(  torch.sum( (all_shapes**2),dim=2),  dim=0) #TODO
        if self.debug_verbose:
            print(gam_h)
            print(dnn_h)
            print('SoM_Insta_SIAN.forward()','dnn_h',dnn_h.shape,'gam_h',gam_h.shape,'shape_loss',shape_loss.shape)
        return dnn_h,gam_h,shape_loss

        

    def precompute_on(self, x):
        self.all_shapes = self.gam.forward_shapes(x)
        self.precomputed_shapes = True
    def precompute_off(self):
        self.all_shapes = None
        self.precomputed_shapes = False

    def collectParameters(self):
        gam_params = self.gam.collectParameters()
        return gam_params
    def forward_shapes(self, x):
        return self.gam.forward_shapes(x)
    def to(self, *args, **kwargs):
        self.gam = self.gam.to(*args, **kwargs)
        self = super().to(*args, **kwargs) 
        return self
    def compress(self):
        self.gam.compress()
    def blocksparse(self):
        self.gam.blocksparse()






class SmoothOrMnist_UnmaskedOrMasked_MLP(nn.Module):
    def __init__(self, sizes, masking_mode="default_unmasked", parametrization_mode="smooth_parametrization"):
        super(SmoothOrMnist_UnmaskedOrMasked_MLP,self).__init__()
        
        if masking_mode=="default_masked":
            sizes[0] = 2*sizes[0]
        self.sizes = copy.deepcopy(sizes)

        self.dnn = MNIST_or_Smooth_ReLU_DNN(sizes, parametrization_mode=parametrization_mode)
        self.masking_mode = masking_mode # "default_unmasked"  or "default_masked"
        self.value = 0 #masked value

    def forward(self, x):
        if self.masking_mode == "default_unmasked":
            return self.vanilla_forward(x)
        elif self.masking_mode == "default_masked":
            return self.masked_forward(x)
        else:
            raise NotImplementedError(f"self.masking_mode={self.masking_mode}")
        
    def vanilla_forward(self, x):
        dnn_h = self.dnn(x)
        gam_h = torch.zeros_like(dnn_h)
        return dnn_h,gam_h,None

    def masked_forward(self, xx):
        x,S = xx
        x = x * S + self.value * (1-S)
        if True: #appending
            x = torch.cat((x,S),dim=1)
        dnn_h = self.dnn(x)
        gam_h = torch.zeros_like(dnn_h)
        return dnn_h,gam_h,None

    def collectParameters(self):
        dnn_params = self.dnn.collectParameters()
        return torch.cat([dnn_params])

    def precompute_on(self, x): #do nothing for MLP
        pass
    def precompute_off(self):
        pass


