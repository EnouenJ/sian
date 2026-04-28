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




from .models import FeedForwardDNN







def plot_weights(w):
    plt.imshow(w, cmap='coolwarm');

    B = np.max(np.abs(w))
    plt.clim(-B,B)
    
    plt.colorbar();
    plt.show();



DEBUGGING = True
DEBUGGING = False

GRADNORMS=True
GRADNORMS=False



# 07/27/2025 -- new approach to improving the stability of mnist SIAN
#               it seems to have very minimal effect, especially for relatively
#               small LR's, but I can't imagine any possible reasons for it to be
#               harmful, so I think I will highly likely keep it around.
RELU_FLIPPING = False
RELU_FLIPPING = True



class MNIST_or_Smooth_SIAN_GAM(nn.Module):
    def __init__(self, feat_in, all_indices, small_sizes=[0, 16, 12, 8, 1], parametrization_mode="smooth_parametrization", bias_configuration=None):
        super(MNIST_or_Smooth_SIAN_GAM, self).__init__()

        
        if bias_configuration is None:
            bias_configuration = { #NOTE: should have a global for getting default
                "biases_on" : False,
                "bias_on" : True,
                "biases_scaling" : 1.0,
                "bias_scaling" : 1.0,
                "initialize_bias" : False,
            }

        self.all_indices = all_indices
        self.used_indices = list(range(len(all_indices)))

        self.parametrization_mode = parametrization_mode
        # print(self.parametrization_mode, "INSIDE DEBUGGING MERGED")


        if self.parametrization_mode == "mnist_parametrization": #JAM: changing these to string literals for more readable code
            pass
            # print("=== Using MNIST SIAN ===")
            # self.parametrization_mode = "mnist"
        elif self.parametrization_mode == "smooth_parametrization":
            pass
            # print("Using Smooth SIAN")
            # self.parametrization_mode = "smooth"
        else:
            raise NotImplementedError(f"self.parametrization_mode={self.parametrization_mode}")
            
        self.all_sizes = [small_sizes for _ in all_indices]
        self.activation = F.relu
        self.sizes = []
        self.length = len(self.all_sizes[0]) - 1
        for k in range(self.length + 1):
            size_k = 0
            if k == 0:
                size_k = feat_in
            else:
                for j, index in enumerate(self.all_indices):
                    size_k += self.all_sizes[j][k]
            self.sizes.append(size_k)

        self.mode = "blocksparse"  # or "compressed"
        self.models = None

        if True:
            self.n_groups = len(self.all_indices)
            self.output_dim = small_sizes[-1]
        
        self.grad_masks = []
        for k in range(self.length):
            size_k1 = 0
            size_k2 = 0
            grad_mask = torch.zeros((self.sizes[k + 1], self.sizes[k]))

            for j, index in enumerate(self.all_indices):
                curr_k2 = self.all_sizes[j][k + 1]
                if k == 0:
                    for i in index:
                        grad_mask[size_k2 : size_k2 + curr_k2, i] = 1
                else:
                    curr_k1 = self.all_sizes[j][k]
                    grad_mask[size_k2:size_k2 + curr_k2, size_k1:size_k1 + curr_k1] = 1
                    size_k1 += curr_k1
                size_k2 += curr_k2

            self.grad_masks.append(torch.Tensor(grad_mask))

        self.hiddens = nn.ModuleList()
        for k in range(self.length):
            self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k + 1]))


        #08/16/2025 @ 1:00am
        self.bias = torch.nn.Parameter(torch.zeros(self.output_dim))
        self.biases = torch.nn.Parameter(torch.zeros(len(self.all_indices),self.output_dim))
        
        self.biases_on = bias_configuration["biases_on"]
        self.bias_on = bias_configuration["bias_on"]
        self.biases_scaling = bias_configuration["biases_scaling"]
        self.bias_scaling = bias_configuration["bias_scaling"]
        self.initialize_bias = bias_configuration["initialize_bias"]
        if self.initialize_bias:
            raise NotImplementedError("initialize_bias is not implememnted here, requires training statistics")

        self.compute_shapeloss_while_training = False
            


        def get_sparse_hook(param_idx, model):
            def hook(grad):
                grad = grad.clone() #don't change the given gradient in place
                grad_mask = model.grad_masks[param_idx]
                grad = grad * grad_mask #gradient mask will be on same device b/c of IDK register

                #DEBUG
                if DEBUGGING:
                    # plot_weights(grad.detach().cpu().numpy())
                    print('grad.shape',grad.shape)
                    plot_weights(grad.detach().cpu().numpy()[:200,:200])

                if GRADNORMS:
                    xd = grad.detach().cpu().numpy()
                    print(np.mean(xd**2))
                    plt.figure(figsize=(12,7.5))

                    plot_weights(grad.detach().cpu().numpy()[:20,:100])
                    plot_weights(grad.detach().cpu().numpy()[-20:,-100:])

                return grad
            return hook
        

        for k in range(self.length):
            if self.parametrization_mode == "smooth_parametrization":
                self.hiddens[k].weight = torch.nn.Parameter(self.hiddens[k].weight.data * self.grad_masks[k])
                if k!=0:
                    self.hiddens[k].weight = torch.nn.Parameter( self.hiddens[k].weight.data * self.grad_masks[k]  *np.sqrt(len(self.all_indices)) )
                    self.hiddens[k].bias   = torch.nn.Parameter( self.hiddens[k].bias.data  *np.sqrt(len(self.all_indices)) )


            elif self.parametrization_mode == "mnist_parametrization":

                # 07/27/25                
                if self.parametrization_mode == "mnist_parametrization": #the actual muP parameter inits
                    if k==0:
                        torch.nn.init.kaiming_normal_(self.hiddens[k].weight.data,a=0,mode='fan_out', nonlinearity='relu')
                    else:
                        torch.nn.init.kaiming_normal_(self.hiddens[k].weight.data,a=0,mode='fan_in', nonlinearity='relu')

                    if k==0:

                        self.hiddens[k].weight = torch.nn.Parameter(self.hiddens[k].weight.data * self.grad_masks[k] * np.sqrt(len(self.all_indices)))

                        if True: #DEBUGGING THIS RN
                            pass
                            scale_mask = []
                            for j, index in enumerate(self.all_indices):
                                print('j',j,'index',index)
                                curr_k1 = len(index)
                                curr_k2 = self.all_sizes[j][k + 1]
                                scale_mask.extend(  [curr_k1]*curr_k2  )

                    else:
                        self.hiddens[k].weight = torch.nn.Parameter(self.hiddens[k].weight.data * self.grad_masks[k] * np.sqrt(len(self.all_indices)))

                    self.hiddens[k].bias = torch.nn.Parameter(self.hiddens[k].bias.data * 0)

                    if False: #TRY RENORMALIZING FROM THE SPHERE AND NOT JUST GAUSSIANS (it seems overpowered weights off the sphere may have a larger impact in this discrete setting)
                        pass
                        l2norm = torch.sum( self.hiddens[k].weight.data**2, axis=1, keepdims=True)
                        print('shapes',self.hiddens[k].weight.data.shape)
                        print('l2norm',l2norm.shape)
                        self.hiddens[k].weight = torch.nn.Parameter(self.hiddens[k].weight.data * self.grad_masks[k] / l2norm * np.sqrt(len(self.all_indices)))

                        pass

            self.hiddens[k].weight.register_hook(get_sparse_hook(k, self))

        if False: #for debugging
            print("NAMING MYSELF")
            for name,param in self.named_parameters():
                print(name,param.shape)
            for name,param in self.named_parameters():
                print(param)
                
    

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if not self.grad_masks is None:
            new_grad_masks = [grad_mask.to(*args, **kwargs) for grad_mask in self.grad_masks]
            self.grad_masks = new_grad_masks
        return self


    def forward(self, x):
        shape_loss = torch.zeros(len(self.all_indices), device=x.device)
        if self.mode == "blocksparse":

            h = x

            if self.parametrization_mode == "mnist_parametrization":
                h = h * np.sqrt(self.sizes[1] / len(self.all_indices))  


            for k in range(self.length):
                h = self.hiddens[k](h)

                if k==0 and self.parametrization_mode == "mnist_parametrization":
                    output_rescaler = []
                    for j, index in enumerate(self.all_indices):
                        my_len = ( (len(index)) )
                        if my_len==0:
                            my_len=1
                        output_rescaler.extend(  [my_len]*self.all_sizes[j][1]  ) 
                    output_rescaler = torch.Tensor(output_rescaler)[None,:].to(h.device)

                    h = h / output_rescaler


                if DEBUGGING: #DEBUGGING BIAS TERM
                    mybs = h.shape[0]
                    for sh in range(h.shape[1]):
                        xd = h[:,sh].cpu().detach().numpy()
                        plt.scatter(np.arange(mybs),   xd)
                        plt.plot(np.arange(mybs),   xd)
                    plt.show()

                    plt.scatter(np.arange(h.shape[1]),np.mean(h[:,:].cpu().detach().numpy(),axis=0), c='gray')
                    plt.show()

                if k != self.length - 1:
                    h = self.activation(h)

                    if RELU_FLIPPING and self.parametrization_mode == "mnist_parametrization":
                        flip_scaler = np.arange(h.shape[1])
                        flip_scaler = (-np.ones(1)) ** flip_scaler
                        flip_scaler = torch.Tensor(flip_scaler)[None,:].to(h.device)
                        h = h * flip_scaler


            if self.parametrization_mode == "mnist_parametrization":
                h = h / np.sqrt(self.sizes[k] / len(self.all_indices))
            h = h.reshape(h.shape[0], self.n_groups, self.output_dim)

            if DEBUGGING: #DEBUG
                mybs = h.shape[0]
                if False:
                    for sh in range(h.shape[1]):
                        xd = h[:,sh,0].cpu().detach().numpy()
                        plt.scatter(np.arange(mybs),   xd)
                        plt.plot(np.arange(mybs),   xd)
                    plt.show()
                if True:
                    var_list = []
                    mean_list = []
                    color_list = []

                    for j, index in enumerate(self.all_indices[::-1]):
                        colors=["C3","C0","C2"]
                        newlen = (len(index)-1) // 5
                        xd = h[:,j,0].cpu().detach().numpy()
                        plt.scatter(np.arange(mybs),   xd, c=colors[newlen])
                        plt.plot(np.arange(mybs),   xd, c=colors[newlen])

                        var_list.append(np.var(xd))
                        mean_list.append(np.mean(xd))
                        color_list.append(colors[newlen])
                    plt.show()
                        
                    plt.figure(figsize=(8,5))
                    plt.scatter(np.arange(len(self.all_indices)),var_list, c=color_list)
                    if True:
                        plt.yscale('log')
                    plt.show()

                    plt.figure(figsize=(8,5))
                    plt.title('mean')
                    plt.scatter(np.arange(len(self.all_indices)),mean_list, c=color_list)
                    plt.show()
            
            if self.biases_on: 
                h = h + (self.biases_scaling*self.biases)[None]
            if self.compute_shapeloss_while_training:
                # shape_loss = torch.mean(  torch.sum(torch.sum( torch.abs(h),dim=2),dim=1),  dim=0)
                #shape_loss = torch.mean(  torch.sum(torch.sum( (h**2),dim=2),dim=1),  dim=0) #TODO
                shape_loss = torch.mean(  (torch.sum( (h**2),dim=2)),  dim=0) #TODO (change to no sum over shapes, same as unmasked w/ forward_shapes)
            out = torch.sum(h, dim=1) 
            if self.bias_on:
                out = out + (self.bias_scaling*self.bias)[None]

        elif self.mode == "compressed":
            if self.parametrization_mode == "mnist_parametrization":
                raise NotImplementedError("need to fix shapes for .biases in compressed version")
            
            features = [] #NOTE: this sequential computation can take a long time
            for j, index in enumerate(self.all_indices):
                inputs = []
                for ind in index:
                    inputs.append(x.narrow(1, ind, 1))
                inputs = torch.cat(inputs, dim=1)
                ft = self.models[j](inputs)  # (batch_size, C)
                features.append(ft)

            if len(features) > 0:
                if self.parametrization_mode == "smooth_parametrization":
                    features = torch.stack(features, dim=1)  # (batch_size, n_groups, C)
                    if self.biases_on:
                        features = features + (self.biases_scaling*self.biases)[None]
                    if self.compute_shapeloss_while_training:
                        shape_loss = torch.mean(  torch.sum( torch.abs(h),dim=2),  dim=0)
                    out = torch.sum(features, dim=1)  # (batch_size, C)
                    if self.bias_on:
                        out = out + (self.bias_scaling*self.bias)[None]
                elif self.parametrization_mode == "mnist_parametrization":
                    raise NotImplementedError("")
                    features = torch.cat(features, dim=1)
                    out = torch.sum(features, dim=1)
                    out = out.unsqueeze(-1)
                    out = out + (self.bias).repeat(x.shape[0]).unsqueeze(-1)
            else:
                out = torch.zeros(x.shape[0], self.output_dim, device=x.device)

        return out, shape_loss
    
    def forward_shapes(self, x):
        if self.mode == "blocksparse":
            h = x
            for k in range(self.length):
                h = self.hiddens[k](h)
                if k != self.length - 1:
                    h = self.activation(h)
            h = h.view(h.shape[0], self.n_groups, self.output_dim)
            if self.biases_on:
                h = h + (self.biases_scaling*self.biases)[None]
            return h

        elif self.mode == "compressed":
            features = []
            for j, index in enumerate(self.all_indices):
                inputs = []
                for ind in index:
                    inputs.append(x.narrow(1, ind, 1))
                inputs = torch.cat(inputs, dim=1)
                ft = self.models[j](inputs)  
                features.append(ft)
            features = torch.stack(features, dim=1) 
            if self.biases_on:
                features = features + (self.biases_scaling*self.biases)[None]
            return features
        return None

    def blocksparse(self):
        if self.mode == "compressed":
            self.mode = "blocksparse"
            device = self.bias.device

            self.grad_masks = [] #NOTE: regenerated each time, but maybe for some applications it is better to keep them

            for k in range(self.length):
                size_k1 = 0
                size_k2 = 0
                grad_mask = torch.zeros((self.sizes[k + 1], self.sizes[k]))

                for j, index in enumerate(self.all_indices):
                    curr_k2 = self.all_sizes[j][k + 1]
                    if k == 0:
                        for i in index:
                            grad_mask[size_k2 : size_k2 + curr_k2, i] = 1
                    else:
                        curr_k1 = self.all_sizes[j][k]
                        grad_mask[size_k2:size_k2 + curr_k2, size_k1:size_k1 + curr_k1] = 1
                        size_k1 += curr_k1

                    size_k2 += curr_k2

                self.grad_masks.append(torch.Tensor(grad_mask))
            
            self.hiddens = nn.ModuleList()
            for k in range(self.length):
                self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k + 1]))

            def get_sparse_hook(param_idx, model):
                def hook(grad):
                    grad = grad.clone()
                    grad_mask = model.grad_masks[param_idx]
                    grad = grad * grad_mask
                    return grad
                return hook

            for k in range(self.length):
                size_k1 = 0
                size_k2 = 0
                layer_weight = torch.zeros((self.sizes[k + 1], self.sizes[k]))
                layer_bias = torch.zeros(self.sizes[k + 1])

                for j, index in enumerate(self.all_indices):
                    
                    curr_k2 = self.all_sizes[j][k + 1]

                    if k == 0:
                        for ii, i in enumerate(index):
                            layer_weight[size_k2:size_k2 + curr_k2, i] = self.models[j].hiddens[k].weight.data[:, ii]
                        layer_bias[size_k2:size_k2 + curr_k2] = self.models[j].hiddens[k].bias.data
                    else:
                        curr_k1 = self.all_sizes[j][k]
                        layer_weight[size_k2:size_k2 + curr_k2, size_k1:size_k1 + curr_k1] = self.models[j].hiddens[k].weight.data
                        layer_bias[size_k2:size_k2 + curr_k2] = self.models[j].hiddens[k].bias.data
                        size_k1 += curr_k1

                    size_k2 += curr_k2
                
                self.hiddens[k].weight = torch.nn.Parameter(layer_weight.to(device))
                self.hiddens[k].bias = torch.nn.Parameter(layer_bias.to(device))
                self.hiddens[k].weight.register_hook(get_sparse_hook(k, self))

            del self.models
            self.models = None
            print("blocksparsed :]")

    def compress(self):
        if self.mode == "blocksparse":
            self.mode = "compressed"
            device = self.bias.device

            self.models = nn.ModuleList()
            for j, index in enumerate(self.all_indices):
                sizes = self.all_sizes[j]
                sizes[0] = len(index)
                self.models.append(FeedForwardDNN(sizes).to(device))

            for k in range(self.length):
                size_k1 = 0
                size_k2 = 0

                for j, index in enumerate(self.all_indices):
                    curr_k2 = self.all_sizes[j][k+1]
                    curr_k1 = self.all_sizes[j][k]

                    if k == 0:
                        firstlayer = torch.zeros((curr_k2, len(index)))
                        for ii, i in enumerate(index):
                            firstlayer[:, ii] = self.hiddens[k].weight.data[size_k2 : size_k2 + curr_k2, i]

                        self.models[j].hiddens[k].weight = torch.nn.Parameter(firstlayer.to(device))
                        self.models[j].hiddens[k].bias = torch.nn.Parameter(self.hiddens[k].bias.data[size_k2:size_k2+curr_k2].to(device))
                    else:
                        self.models[j].hiddens[k].weight = torch.nn.Parameter(self.hiddens[k].weight.data[size_k2:size_k2+curr_k2, size_k1:size_k1+curr_k1].to(device))
                        self.models[j].hiddens[k].bias = torch.nn.Parameter(self.hiddens[k].bias.data[size_k2:size_k2+curr_k2].to(device))

                    size_k1 += curr_k1
                    size_k2 += curr_k2
                
            del self.hiddens
            self.hiddens = None
            del self.grad_masks
            self.grad_masks = None
            print("compressed :)")


    def addIndices(self, index, small_sizes=[-1,16,12,8,-1]):
        self.all_indices.append(index)

        if self.mode == "compressed":
            small_sizes[0] = len(index)
            small_sizes[-1] = self.output_dim
            self.models.append(FeedForwardDNN(small_sizes)) #TODO: add MNIST_or_Smooth_ReLU_DNN() and parametrization
            self.all_sizes.append(small_sizes)
            for k in range(1, len(self.sizes)):
                self.sizes[k] += sizes[k]   #TODO: rename to fullsizes?
        else:
            raise NotImplementedError("") #TODO: finish addIndices()



    def collectParameters(self):
        all_param_list = []
        if self.mode == "blocksparse":
            for k in range(self.length):
                for x in self.hiddens[k].parameters():
                    all_param_list.append(x.view(-1))
        elif self.mode == "compressed":
            for model in self.models:
                model_params = model.collectParameters()
                all_param_list.append(model_params)
            if len(all_param_list) > 0:
                return torch.cat(all_param_list)
            else:
                return torch.zeros(1)
        return torch.cat(all_param_list)



class MNIST_or_Smooth_ReLU_DNN(nn.Module):
    def __init__(self, sizes, parametrization_mode="smooth_parametrization"):
        super(MNIST_or_Smooth_ReLU_DNN, self).__init__()

        self.parametrization_mode = parametrization_mode

        if self.parametrization_mode == "mnist_parametrization":
            print("=== Using MNIST MLP ===")
        if self.parametrization_mode == "smooth_parametrization":
            print("Using Smooth MLP")

        self.sizes = sizes
        self.length = len(self.sizes)-1

        self.activation = F.relu
        
        
        self.hiddens = nn.ModuleList()
        for k in range(self.length):
            self.hiddens.append(nn.Linear(self.sizes[k], self.sizes[k+1]))

        if self.parametrization_mode == "mnist_parametrization": #the actual muP parameter inits
            for ll,layer in enumerate(self.hiddens): #Greg muP (fan in except for first layer)
                if ll==0:
                    torch.nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                else:
                    torch.nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                layer.bias.data.fill_(0)
        
        if self.parametrization_mode == "old_mnist_parametrization":
            for ll,layer in enumerate(self.hiddens): #Greg muP (fan in except for first layer)
                torch.nn.init.kaiming_normal_(layer.weight.data,a=0,mode='fan_in', nonlinearity='relu') 
                layer.bias.data.fill_(0)

        
    def forward(self, x):
        h = x
        for k in range(self.length):
            h = self.hiddens[k](h)

            if self.parametrization_mode == "mnist_parametrization" or self.parametrization_mode == "old_mnist_parametrization":
                #Greg muP -- pre-input upscaling
                if k==0:
                    h *= np.sqrt(self.sizes[1]) 
            
            if k!= self.length-1:
                h = self.activation(h)

                if RELU_FLIPPING: #09/20/25 @ 5:00am
                    flip_scaler = np.arange(h.shape[1])
                    flip_scaler = (-np.ones(1)) ** flip_scaler
                    flip_scaler = torch.Tensor(flip_scaler)[None,:].to(h.device)
                    h = h * flip_scaler
            else:
                if self.parametrization_mode == "mnist_parametrization" or self.parametrization_mode == "old_mnist_parametrization":
                    #Greg muP -- post-output downscaling
                    h /= np.sqrt(self.sizes[k])
        
        return h

    def collectParameters(self):
        all_param_list = []
        for k in range(self.length):
            for x in self.hiddens[k].parameters():
                all_param_list.append(x.reshape(-1))        
        return torch.cat(all_param_list)
        
































# class Smooth_Or_MNIST_SIAN(nn.Module):
#     def __init__(self, sizes, indices, dnn_on_or_off=False, small_sizes=[0, 16, 12, 8, 1], feature_groups_dict=None, mnist_on_or_off="smooth", bias_configuration=None):
#         super(Smooth_Or_MNIST_SIAN, self).__init__()

#         self.dnn_on = dnn_on_or_off
#         if not self.dnn_on:
#             sizes = [sizes[0], sizes[-1]]
#         self.sizes = sizes

#         self.dnn = MNIST_or_Smooth_ReLU_DNN(sizes, mnist_on_or_off=mnist_on_or_off)

#         self.feature_groups_dict = feature_groups_dict
#         self.indices = indices
#         self.ungrouped_indices = indices
#         if self.feature_groups_dict is not None:
#             self.ungrouped_indices = []
#             for ind in indices:
#                 new_ind = []
#                 for i in ind:
#                     new_ind.extend(self.feature_groups_dict[i])
#                 self.ungrouped_indices.append(tuple(new_ind))

#         self.gam = MNIST_or_Smooth_SIAN_GAM(sizes[0],self.ungrouped_indices,small_sizes=small_sizes, mnist_on_or_off=mnist_on_or_off, bias_configuration=bias_configuration)

#     def forward(self, x):
#         if self.dnn_on:
#             dnn_h = self.dnn(x)
#         else:
#             dnn_h = self.dnn(x) * 0
#         gam_h, shape_loss = self.gam(x)
#         print('SoM_Un_SIAN.forward()','dnn_h',dnn_h.shape,'gam_h',gam_h.shape,'shape_loss',shape_loss.shape)
#         return dnn_h, gam_h, shape_loss

#     def collectParameters(self):
#         dnn_params = self.dnn.collectParameters()
#         gam_params = self.gam.collectParameters()
#         return torch.cat([dnn_params, gam_params])

#     def forward_shapes(self, x):
#         return self.gam.forward_shapes(x)

#     def to(self, *args, **kwargs):
#         self.gam = self.gam.to(*args, **kwargs)
#         self = super().to(*args, **kwargs)
#         return self

#     def compress(self):
#         self.gam.compress()

#     def blocksparse(self):
#         self.gam.blocksparse()









# #08/16/2025
# class SmoothOrMnist_InstaSHAPMasked_SIAN(nn.Module):
#     def __init__(self,sizes,indices,  dnn_on_or_off=False,small_sizes=[0,16,12,8,1],feature_groups_dict=None,mnist_on_or_off="smooth", bias_configuration=None):
#         super(SmoothOrMnist_InstaSHAPMasked_SIAN,self).__init__()
        
#         self.sizes = copy.deepcopy(sizes)
#         small_sizes = copy.deepcopy(small_sizes)
#         small_sizes[-1] = sizes[-1]
        
#         subset_indexer = np.zeros((self.sizes[0],len(indices)))
#         subset_indexer_const = np.zeros((1,len(indices)))
        
#         self.feature_groups_dict = feature_groups_dict #04/13/2025

#         for ii,ind in enumerate(indices):
#             if self.feature_groups_dict is None:
#                 for i in ind:
#                     subset_indexer[i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
#                 subset_indexer_const[0,ii] = 1 / (len(ind)+0.5)
#             else:
#                 for i in ind:
#                     sub_i = self.feature_groups_dict[i][0] #just use the first dimension (NOTE: assumes the full mask is initialized correctly)
#                     subset_indexer[sub_i,ii] = 1 / (len(ind)+0.5) #meaning total should be greater than 1.0 (in presence of all k elements)
#                 subset_indexer_const[0,ii] = 1 / (len(ind)+0.5)

#         self.subset_indexer_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer).float(), requires_grad=False )
#         self.subset_indexer_const_tensor = torch.nn.Parameter( torch.from_numpy(subset_indexer_const).float(), requires_grad=False )
        
        
#         self.indices = indices
#         self.ungrouped_indices = indices
#         if self.feature_groups_dict is not None:
#             self.ungrouped_indices = []
#             for ind in indices:
#                 new_ind = []
#                 for i in ind:
#                     new_ind.extend(  self.feature_groups_dict[i]  )
#                 self.ungrouped_indices.append( tuple(new_ind) )
#         self.value = 0

#         self.gam = MNIST_or_Smooth_SIAN_GAM(sizes[0],self.ungrouped_indices,small_sizes=small_sizes,mnist_on_or_off=mnist_on_or_off,bias_configuration=bias_configuration)
#         self.precompute_off()

#         self.debug_verbose = False #01/29/2025

#     def forward(self, xx):
#         x,S = xx
#         x = x * S + self.value * (1-S)


#         if self.precomputed_shapes:
#             all_shapes = self.all_shapes
#         else:
#             all_shapes = self.gam.forward_shapes(x)


#         shape_fn_mask = (torch.matmul(S.float(),self.subset_indexer_tensor)+self.subset_indexer_const_tensor>1).float()
#         shape_fn_mask = shape_fn_mask[:,:,None]

        
        
#         if self.gam.biases_on:
#             all_shapes = all_shapes + (self.gam.biases_scaling*self.gam.biases)[None]

#         if self.gam.bias_on:
#             all_shapes = all_shapes + (self.gam.bias_scaling*self.gam.bias)[None,None]
        

#         all_shapes = all_shapes*shape_fn_mask
#         if self.debug_verbose:
#             print(all_shapes)

#         gam_h = torch.sum(all_shapes, dim=1)
#         dnn_h = torch.zeros_like(gam_h)

#         shape_loss = torch.zeros(len(self.indices), device=x.device)
#         if self.gam.compute_shapeloss_while_training:
#             shape_loss = torch.mean(  torch.sum( torch.abs(all_shapes),dim=2),  dim=0)
#         if self.debug_verbose:
#             print(gam_h)
#             print(dnn_h)
#         print('SoM_Insta_SIAN.forward()','dnn_h',dnn_h.shape,'gam_h',gam_h.shape,'shape_loss',shape_loss.shape)
#         return dnn_h,gam_h,shape_loss

        

#     def precompute_on(self, x):
#         self.all_shapes = self.gam.forward_shapes(x)
#         self.precomputed_shapes = True
#     def precompute_off(self):
#         self.all_shapes = None
#         self.precomputed_shapes = False

#     def collectParameters(self):
#         gam_params = self.gam.collectParameters()
#         return gam_params
#     def forward_shapes(self, x):
#         return self.gam.forward_shapes(x)
#     def to(self, *args, **kwargs):
#         self.gam = self.gam.to(*args, **kwargs)
#         self = super().to(*args, **kwargs) 
#         return self
#     def compress(self):
#         self.gam.compress()
#     def blocksparse(self):
#         self.gam.blocksparse()

        