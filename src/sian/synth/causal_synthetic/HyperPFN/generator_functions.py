






import copy
import numpy as np
import torch
import torch.nn as nn






class GeneratorFunction(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        raise NotImplementedError("absract class")





class LinearFunction(GeneratorFunction):
    def __init__(self, dim_list, d_out, coef_dist_info):
        super().__init__()

        self.dim_list = dim_list
        self.d_out = d_out
        full_dim_list = copy.deepcopy(dim_list)
        full_dim_list.append(d_out)
        self.full_dim_list = full_dim_list
        self.coef_dist_info = coef_dist_info

        coef_dist_type = coef_dist_info['type']
        if coef_dist_type == 'log_uniform':
            coef_low  = np.log(coef_dist_info['low'])
            coef_high = np.log(coef_dist_info['high'])

            coef_sign = 2*np.random.randint(2,size=tuple(full_dim_list))-1
            coef = coef_sign * np.exp(  np.random.uniform(coef_low,coef_high,size=tuple(full_dim_list))  )

        elif coef_dist_type == 'gaussian':
            std = coef_dist_info['std']
            coef = np.random.normal(0.0,std,size=tuple(full_dim_list))

        else:
            raise NotImplementedError()

        self.linear_coef = nn.Parameter( torch.Tensor(coef) )

    def forward(self, X):
        dim_list = self.dim_list
        d_in = len(dim_list)

        if d_in>0:
            i = 0
            I_k = dim_list[i]
            X_i = X[:,:I_k]
            result = torch.einsum('na,a...->n...', X_i, self.linear_coef)
            ii = I_k
            for i in range(1,d_in):
                I_k = dim_list[i]
                X_i = X[:,ii:ii+I_k]
                result = torch.einsum('nb,nb...->n...', X_i, result)
                ii += I_k
                pass
        else:
            #NOTE: hope this can work even without passing "N"
            result = torch.einsum('na,a...->n...', torch.ones(X.shape[0],1), self.linear_coef)
            

        return result
        



    
