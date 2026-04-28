



import torch
import torch.nn as nn







class Mnist_CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= nn.Sequential(
                nn.Conv2d(1, 16, 3, 1),
                nn.ELU(inplace = True),
                nn.Conv2d(16, 32, 3, 1),
                nn.ELU(inplace = True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, 1),
                nn.ELU(inplace = True),
                nn.Conv2d(64, 128, 3, 1),
                nn.ELU(inplace = True),
                nn.MaxPool2d(2, 2),
                nn.Flatten(start_dim = 1),
                nn.Linear(2048, 256),
                nn.ELU(inplace = True),
                nn.Linear(256, 10))
    def forward(self, x):
        # print('x',x.shape)
        x = x.reshape(-1, 1, 28, 28) #09/18/25 -- allowing flat inputs right now
        # print('x',x.shape)
        #### return self.model(x)
        y = self.model(x)
        # print('y',y.shape)
        #### return y
        return y, torch.zeros_like(y), 0.0    #TODO: still werid, no?
    
    def collectParameters(self):
        all_param_list = []
        for thing in self.model:
            for x in thing.parameters():
                all_param_list.append(x.reshape(-1))
        return torch.cat(all_param_list)






class Mnist_CnnGam1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_list = nn.ModuleList()

        for i in range(49):
            self.model_list.append(
              nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, padding=1),
                nn.ELU(inplace = True),
                nn.Conv2d(16, 32, 3, 1, padding=1),
                nn.ELU(inplace = True),
                nn.MaxPool2d(2, 2),
                # nn.Conv2d(32, 64, 3, 1),
                # nn.ELU(inplace = True),
                # nn.Conv2d(64, 128, 3, 1),
                # nn.ELU(inplace = True),
                # nn.MaxPool2d(2, 2),
                nn.Flatten(start_dim = 1),
                ## nn.Linear(2048, 256),
                ## nn.ELU(inplace = True),
                ## nn.Linear(256, 10))
                nn.Linear(128, 64),
                nn.ELU(inplace = True),
                nn.Linear(64, 10))
            )

    def forward(self, x):
        # print('x',x.shape)
        x = x.reshape(-1, 1, 28, 28) #09/18/25 -- allowing flat inputs right now
        patch_size = 4
        x_patches = []
        for h in range(7):
            for w in range(7):
                x_patch = x[:,:,patch_size*h:patch_size*h+patch_size, patch_size*w:patch_size*w+patch_size]
                x_patches.append( x_patch )

        y = 0.0
        for mm,model in enumerate(self.model_list):
            y += model(x_patches[mm])
        return y, torch.zeros_like(y), 0.0    #TODO: still werid, no?
    
    def collectParameters(self):
        all_param_list = []
        for model in self.model_list:
            for thing in model:
                for x in thing.parameters():
                    all_param_list.append(x.reshape(-1))
        return torch.cat(all_param_list)





class Mnist_CnnGam2x2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_list = nn.ModuleList()

        for i in range(36):
            self.model_list.append(
              nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, padding=1),
                nn.ELU(inplace = True),
                nn.Conv2d(32, 64, 3, 1, padding=1),
                nn.ELU(inplace = True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 1, 1, padding=0),
                nn.ELU(inplace = True),
                nn.Conv2d(128, 256, 1, 1, padding=0),
                nn.ELU(inplace = True),
                nn.MaxPool2d(2, 2),
                nn.Flatten(start_dim = 1),
                ## nn.Linear(2048, 256),
                ## nn.ELU(inplace = True),
                ## nn.Linear(256, 10))
                nn.Linear(1024, 256),
                nn.ELU(inplace = True),
                nn.Linear(256, 10))
            )

    def forward(self, x):
        # print('x',x.shape)
        x = x.reshape(-1, 1, 28, 28) #09/18/25 -- allowing flat inputs right now
        patch_size = 4
        x_patches = []
        K=2
        for h in range(7-K+1):
            for w in range(7-K+1):
                x_patch = x[:,:,patch_size*h:patch_size*h+K*patch_size, patch_size*w:patch_size*w+K*patch_size]
                x_patches.append( x_patch )

        y = 0.0
        for mm,model in enumerate(self.model_list):
            y += model(x_patches[mm])
        return y, torch.zeros_like(y), 0.0    #TODO: still werid, no?
    
    def collectParameters(self):
        all_param_list = []
        for model in self.model_list:
            for thing in model:
                for x in thing.parameters():
                    all_param_list.append(x.reshape(-1))
        return torch.cat(all_param_list)




# from .new_inflated_models import SemiInflated_SmoothOrMnist_UnmaskedOrMasked_SIAN

# class Mnist_CnnGam2x2_plusLongRange(nn.Module):
#     def __init__(self, FIS_interactions, feature_groups_dict):
#         super().__init__()
#         print('Mnist_CnnGam2x2_plusLongRange()')


#         self.model_list = nn.ModuleList()

#         for i in range(36):
#             self.model_list.append(
#               nn.Sequential(
#                 nn.Conv2d(1, 32, 3, 1, padding=1),
#                 nn.ELU(inplace = True),
#                 nn.Conv2d(32, 64, 3, 1, padding=1),
#                 nn.ELU(inplace = True),
#                 nn.MaxPool2d(2, 2),
#                 nn.Conv2d(64, 128, 1, 1, padding=0),
#                 nn.ELU(inplace = True),
#                 nn.Conv2d(128, 256, 1, 1, padding=0),
#                 nn.ELU(inplace = True),
#                 nn.MaxPool2d(2, 2),
#                 nn.Flatten(start_dim = 1),
#                 ## nn.Linear(2048, 256),
#                 ## nn.ELU(inplace = True),
#                 ## nn.Linear(256, 10))
#                 nn.Linear(1024, 256),
#                 nn.ELU(inplace = True),
#                 nn.Linear(256, 10))
#             )

#         sizes  = [-1, 256, 128, 64, -1]
#         small_sizes = [-1, 32, 24, 16, -1]
#         sizes  = [784, 256, 128, 64, 10]
#         small_sizes = [-1, 32, 24, 16, 10]
#         max_inflation_amount = 64
#         self.sian = SemiInflated_SmoothOrMnist_UnmaskedOrMasked_SIAN(sizes, FIS_interactions, small_sizes=small_sizes, feature_groups_dict=feature_groups_dict, max_inflation_amount=64)
    

#     def forward(self, x):
#         # print('x',x.shape)
#         x_sq = x.reshape(-1, 1, 28, 28) #09/18/25 -- allowing flat inputs right now
#         patch_size = 4
#         x_patches = []
#         K=2
#         for h in range(7-K+1):
#             for w in range(7-K+1):
#                 x_patch = x_sq[:,:,patch_size*h:patch_size*h+K*patch_size, patch_size*w:patch_size*w+K*patch_size]
#                 x_patches.append( x_patch )

#         y = 0.0
#         for mm,model in enumerate(self.model_list):
#             y += model(x_patches[mm])
#         dnn_y,gam_y,shapeloss = self.sian(x)
#         return dnn_y, y+gam_y, shapeloss 
    
#     def collectParameters(self):
#         all_param_list = []
#         for model in self.model_list:
#             for thing in model:
#                 for x in thing.parameters():
#                     all_param_list.append(x.reshape(-1))
#         return torch.cat(all_param_list)
    
#     def to(self, *args, **kwargs):
#         self.sian = self.sian.to(*args, **kwargs) #grad mask needs to be moved
#         self = super().to(*args, **kwargs) 
#         return self