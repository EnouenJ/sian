import random
import numpy as np
import torch
import matplotlib.pyplot as plt






#NOTE: probably needs more of a signature here for these two classes

# class InputDataGeneratorClass(DataGenerator):
#     def __init__(self, D, metainit_seed=None, datagen_seed=None):
#         super().__init__(metainit_seed, datagen_seed)

#         self.D = D
#         self.generator_meta_parameters = {}
#         self.generator_parameters = {}

#     def generate_samples(self, N):
#         raise NotImplementedError("please implement generate_samples() for your class extension.")



# class OutputDataFunctionClass(DataGenerator):
#     def __init__(self, D, metainit_seed=None, datagen_seed=None):
#         super().__init__(metainit_seed, datagen_seed)

#         self.D = D
#         self.generator_meta_parameters = {}
#         self.generator_parameters = {}


#     def generate_output_samples(self, X):
#         raise NotImplementedError("please implement generate_output_samples() for your class extension.")




class DataGenerator():
    def __init__(self, metainit_seed=None, datagen_seed=None):
        if metainit_seed is None:
            metainit_seed = random.randint()
        if datagen_seed is None:
            datagen_seed = metainit_seed

        self.metainit_seed = metainit_seed
        self.datagen_seed = datagen_seed
        self.set_seed(self.metainit_seed)

    # Assuming these three are enough for now
    def set_seed(self, seed):
        #having this as a private method is rather odd, but is better extendible to using personal sources of randomness
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # https://stackoverflow.com/questions/76993193/automatically-calling-class-function-after-init
    def __call__(cls, *args, **kwargs):
        new_obj = type.__call__(cls, *args, **kwargs)
        new_obj.__post_init__()
        return new_obj

    def __post_init_(self):
        self.set_seed(self.datagen_seed)







