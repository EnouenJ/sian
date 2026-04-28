import copy
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F



from .models import Blocksparse_Deep_Relu_GAM
from .models import FeedForwardDNN




#From January 2024 -- copied here on 08/28/2024
class InstaSHAPMasked_SIAN(nn.Module):
    def __init__(self,sizes,indices,  dnn_on_or_off=False,small_sizes=[0,16,12,8,1],feature_groups_dict=None):
        super(InstaSHAPMasked_SIAN,self).__init__()
        
        self.sizes = copy.deepcopy(sizes)
        small_sizes = copy.deepcopy(small_sizes)
        small_sizes[-1] = sizes[-1]
        
        subset_indexer = np.zeros((self.sizes[0],len(indices)))
        subset_indexer_const = np.zeros((1,len(indices)))
        
        self.feature_groups_dict = feature_groups_dict #04/13/2025

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
        
        
        self.indices = indices
        self.ungrouped_indices = indices
        if self.feature_groups_dict is not None:
            self.ungrouped_indices = []
            for ind in indices:
                new_ind = []
                for i in ind:
                    new_ind.extend(  self.feature_groups_dict[i]  )
                self.ungrouped_indices.append( tuple(new_ind) )
        self.value = 0
        self.gam = Blocksparse_Deep_Relu_GAM(sizes[0],self.ungrouped_indices,small_sizes=small_sizes) #04/13/2025
        self.precompute_off()

        self.debug_verbose = False #01/29/2025

    def forward(self, xx):
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
        

        all_shapes = all_shapes * shape_fn_mask
        if self.debug_verbose:
            print(all_shapes)

        gam_h = torch.sum(all_shapes,dim=1)
        dnn_h = torch.zeros_like(gam_h)
        shape_loss = torch.zeros(1)

        if True: #TURNING ON SHAPE_LOSS
            shape_loss = torch.sum(torch.abs(all_shapes),dim=1) #MIGHT BLOW UP FOR LARGE # OF INDICES IF I USE sum()
        if self.debug_verbose:
            print(gam_h)
            print(dnn_h)
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
    def returnLinearNorms(self):
        return self.gam.returnLinearNorms()
    def returnQuadraticNorms(self):
        return self.gam.returnQuadraticNorms()
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

        


#08/28/2024
class FastSHAPMasked_GAM(nn.Module):
    def __init__(self,sizes,indices,  dnn_on_or_off=False,small_sizes=[0,16,12,8,1]):
        super(FastSHAPMasked_GAM,self).__init__()
                
        self.sizes = copy.deepcopy(sizes)
        small_sizes = copy.deepcopy(small_sizes)
        small_sizes[-1] = sizes[-1]
        
        subset_indexer = np.zeros((self.sizes[0],len(indices)))
        subset_indexer_const = np.zeros((1,len(indices)))

        for ii,ind in enumerate(indices):
            for i in ind:
                subset_indexer[i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
            subset_indexer_const[0,ii] = 1 / (len(ind)+0.5)
        self.subset_indexer_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer).float(), requires_grad=False )
        self.subset_indexer_const_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer_const).float(), requires_grad=False )
        
        
        self.indices = indices
        self.value = 0

        self.fast_nets = torch.nn.ModuleList()
        for ii,ind in enumerate(indices):
            small_sizes[0]=sizes[0]
            self.fast_nets.append(  MLP(small_sizes) )

    def forward(self, xx):
        x,S = xx
        x = x * S + self.value * (1-S)
            
        if False:
            all_shapes = self.gam.forward_shapes(x)
        else:
            fast_pred = [self.fast_nets[ii](x)   for ii,ind in enumerate(self.indices)]
            fast_pred = torch.concatenate([logits[0]+logits[1] for logits in fast_pred],dim=1) #TODO: double check for classification 
            all_shapes = fast_pred

        shape_fn_mask = (torch.matmul(S.float(),self.subset_indexer_tensor)+self.subset_indexer_const_tensor>1).float()
        if False: #turning off for regression for now
            shape_fn_mask = shape_fn_mask[:,:,None] #needed for classification tensor 

        all_shapes = all_shapes * shape_fn_mask
        gam_h = torch.sum(all_shapes,dim=1) #classification version?
        if True: #for regression, adding back an empty dimension
            gam_h=gam_h[:,None]
        dnn_h = torch.zeros_like(gam_h)
        shape_loss = torch.zeros(1)
        if True: #TURNING ON SHAPE_LOSS
            shape_loss = torch.sum(torch.abs(all_shapes),dim=1) #MIGHT BLOW UP FOR LARGE # OF INDICES IF I USE sum()
        return dnn_h,gam_h,shape_loss
        
    def forward_shapes(self, x):
            fast_pred = [self.fast_nets[ii](x)   for ii,ind in enumerate(self.indices)]
            fast_pred = torch.concatenate([logits[0]+logits[1] for logits in fast_pred],dim=1)
            all_shapes = fast_pred
            return all_shapes

    def collectParameters(self):
        raise NotImplementedError()
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        return self




class MaskedMLP(nn.Module):
    def __init__(self,sizes,small_sizes=[0,16,12,8,1]):
        super(MaskedMLP,self).__init__()
        sizes[0] = 2*sizes[0]
        self.sizes = sizes
        self.dnn = FeedForwardDNN(sizes)
        self.value = 0 #masked value

    def forward(self, xx):
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

    def precompute_on(self, x): #DO nothing for MLP, just to look pretty (for usage by GAM)
        pass
    def precompute_off(self):
        pass











