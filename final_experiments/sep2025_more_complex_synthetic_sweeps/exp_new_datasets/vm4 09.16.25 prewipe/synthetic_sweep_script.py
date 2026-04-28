# please write code here for sweeping across these datasets
import itertools
import numpy as np
import os
import json


import sys
sys.path.append(".") #TODO: this does not work for me generally
from configurate import setup_config, save_json_and_shell 







# dataset_preproc_pairs = []
# seeds = [0,1,2]
# depths = [0,1,2]
# for seed in seeds:
#     for depth in depths:
#         dataset_preproc_pairs.append( (f"SYNTH_simple_discrete_synthwave_v11_D9_Ik5_depth{depth}_s3.9.27_conditionalHookerANOVA_seed{seed}", None) )
        

dataset_preproc_pairs = []
seeds = [0,1,2]
depths = [0,1,2]
seeds = [0,1] #GCLOUD
depths = [0] #GCLOUD

hered_style_list = ['strongestheredity', 'unshuffled']
hered_style_list = ['strongestheredity', ] #gcloud, 09/15/25 @ 10:30
hered_style_list = ['unshuffled', ] #other gcloud, 09/15/25 @ 10:30
for seed in seeds:
    for depth in depths:
        for hered_style in hered_style_list:
            ###dataset_preproc_pairs.append( (f"SYNTH_simple_discrete_synthwave_v11_D9_Ik5_depth{depth}_s3.9.27_conditionalHookerANOVA_seed{seed}", None) )
            dataset_preproc_pairs.append( (f"SYNTH_simple_discrete_synthwave_v12_D9_Ik5_depth{depth}_s4.16.64_conditionalHookerANOVA_{hered_style}_seed{seed}", None) ) #09/15/25 @ 10:30pm
        



# masked_combinations = [(False, False), (True, True)]  
# use_mnist_scaling = ["mnist", "smooth"]
# FIS_style_list = ["batchwise", "layerwise", "maximal"]

masked_combinations = [(False, False), (True, True)]
use_mnist_scaling = ["smooth", "mnist"]
FIS_style_list = ["maximal"]

masked_combinations = [(False, False),] #GCLOUD
use_mnist_scaling = ["smooth",]
FIS_style_list = ["maximal"]

BS_list = [32]
LR_list = [5e-3]
STEP_list = [1024*128]

MAX_K_list = [3, 2, 1] #better GPU scheduling


target_sample_sizes = list((np.power(2.0, np.arange(-2, 11)) * 100).astype(int))
target_sample_sizes = [int(x) for x in list((np.power(2.0, np.arange(-2, 11)) * 100))] #int64 issue

target_sample_sizes = [int(x) for x in list((np.power(2.0, np.arange(0, 7)) * 100))] #med-dim find
target_sample_sizes = [int(x) for x in list((np.power(2.0, np.arange(0, 7)) * 100))] #med-dim find

shell_save_dir = "shell_scripts/"
data_base_path = "data/"
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)


count_number_of_jobs = 0
for MAX_K, (is_masked_mlp, is_masked_sian), FIS_style, use_mnist, BS, LR, STEP in itertools.product(
    MAX_K_list, masked_combinations, FIS_style_list, use_mnist_scaling, BS_list, LR_list, STEP_list
):
    for dataset_str, preproc_owner in dataset_preproc_pairs:

        job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_BS{BS}_LR{LR}_STEP{STEP}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}"
        json_data = {}

        json_data["data_base_path"] = data_base_path
        json_data["dataset_str"] = dataset_str
        json_data["preproc_owner"] = preproc_owner

        json_data["is_masked_mlp"] = is_masked_mlp
        json_data["is_masked_sian"] = is_masked_sian
        json_data["use_mnist_scaling"] = use_mnist
        json_data["FIS_style"] = FIS_style
        json_data["MAX_K"] = MAX_K
        json_data["number_of_rounds"] = None
        json_data["inters_per_round"] = None

        json_data["BS"] = BS
        json_data["LR"] = LR
        json_data["STEP"] = STEP

        json_data["target_sample_sizes"] = target_sample_sizes

    
        if FIS_style == "maximal": 
            pass
        else:
            raise NotImplementedError(f"FIS_style={FIS_style} not implemented right now")


        if True:
            save_json_and_shell(shell_save_dir, json_data, job_name, 'run_exp_synthsweep.py')
            count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")

