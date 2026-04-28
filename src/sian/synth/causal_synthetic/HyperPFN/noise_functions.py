




import numpy as np
import torch
import torch.nn as nn




class NoiseDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    pass
    def generate_noise(self, N):
        raise NotImplementedError("absract class")



class GaussianNoiseDistribution(NoiseDistribution):
    def __init__(self, std_dist_info):
        self.std_dist_info = std_dist_info
        std_dist_type = std_dist_info['type']
        if std_dist_type == 'log_uniform':
            std_low  = np.log(std_dist_info['low'])
            std_high = np.log(std_dist_info['high'])
            std = np.exp(  np.random.uniform(std_low,std_high) )
            
        else:
            raise NotImplementedError()
        
        self.std = std #NOTE: consider torch-ifying and consider log-scaling

    def generate_noise(self, N):
        # return  np.random.randn(N)*self.std[:,None]
        return  (np.random.randn(N)*self.std)[:,None] #torch.Tensor()



class StandardGumbelDistribution(NoiseDistribution):
    def __init__(self, d_out):
        self.d_out = d_out

    def generate_noise(self, N):
        return  np.random.gumbel(size=(N,self.d_out))


