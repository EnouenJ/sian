# please write code here for sweeping across these datasets
import itertools
import numpy as np
import os
import json
import copy

import sys
sys.path.append(".") #TODO: this does not work for me generally
from configurate import setup_config, generate_slurm_script, save_json_and_shell 






dataset_preproc_pairs = [
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed1", None),
    # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed2", None),
    # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed3", None),
    # # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed4", None),
]



masked_combinations = [(False, False), (True, True)]  
use_mnist_scaling = ["mnist", "smooth"]
FIS_style_list = ["batchwise", "layerwise", "maximal"]
FIS_style_list = ["maximal"]

masked_combinations = [(False, False), ]
use_mnist_scaling = ["smooth", ]
FIS_style_list = ["maximal"]

MAX_K_list = [3, 2] #better GPU scheduling
MAX_K_list = [2]




shell_save_dir = "shell_scripts/"
data_base_path = "data/"
data_base_path = "../../../sian-private-sep13thTestingL4GPUs/final_experiments/sep2025_more_complex_synthetic_sweeps/data/" 
data_base_path = "data/"
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)

#TARGETING THE DOUBLE DESCENT POINT FOR {v9_D10_Ik5_s3.9.27, K=2}
descent_point_2D = 761
descent_point_2D_close = 800
target_sample_sizes = np.round(np.exp( np.log(descent_point_2D_close) + np.linspace(-2.0,2.0,5)/8.0*np.log(2) )).astype(int)
target_sample_sizes = np.round(np.exp( np.log(descent_point_2D_close) + np.linspace(-4.0,4.0,9)/8.0*np.log(2) )).astype(int)
print('target_sample_sizes',target_sample_sizes)
target_sample_sizes = [int(trn_n) for trn_n in target_sample_sizes]
print('target_sample_sizes',target_sample_sizes) # [566, 617, 673, 734, 800, 872, 951, 1037, 1131]


descent_point_3D = 8441
descent_point_3D_close = 6400,12800
descent_point_3D_close = np.sqrt(6400*12800)
print('descent_point_3D_close',descent_point_3D_close)
target_sample_sizes = np.round(np.exp( np.log(descent_point_3D_close) + np.linspace(-4.0,4.0,9)/8.0*np.log(2) )).astype(int)
print('target_sample_sizes',target_sample_sizes)
target_sample_sizes = [int(trn_n) for trn_n in target_sample_sizes]
print('target_sample_sizes',target_sample_sizes) # [6400, 6979, 7611, 8300, 9051, 9870, 10763, 11738, 12800]


target_sample_sizes = [566, 617, 673, 734, 800, 872, 951, 1037]
target_sample_sizes = [566, 673, 800, 951]
BS_list = [32, 128]





target_sample_sizes = [566, 673, 800, 951]
BS_list = [128]
compute_shapeloss = True


#GCLOUD @ 1:50pm on 09/15/2025
target_sample_sizes = [566, 673, 800, 951]
BS_list = [512]
compute_shapeloss = True
#GCLOUD ==> turning up L1 shapeloss (5e-5 ==> 5e-4)


#GCLOUD @ 6:50pm on 09/15/2025
target_sample_sizes = [566, 673, 800, 951]
BS_list = [512]
compute_shapeloss = True
LR = 1.0e-2
#GCLOUD ==> turning up L1 shapeloss (5e-4 ==> 5e-3)


#GCLOUD @ 7:10pm on 09/16/2025
lambda1=0.0
target_sample_sizes = [566, 673, 800, 951]
BS_list = [512]
compute_shapeloss = False
LR = 1.0e-2


#GCLOUD @ 10:00pm on 09/16/2025
target_sample_sizes = np.round(np.exp( np.log(descent_point_2D_close) + np.linspace(-4.0,4.0,9)/8.0*np.log(2) )).astype(int)
target_sample_sizes = np.round(np.exp( np.log(descent_point_2D_close) + np.linspace(-2.0,2.0,5)/4.0*np.log(2) )).astype(int)
target_sample_sizes = np.round(np.exp( np.log(descent_point_2D_close) + np.linspace( 1.0,4.0,4)/4.0*np.log(2) )).astype(int)
target_sample_sizes = [int(trn_n) for trn_n in target_sample_sizes]
print('target_sample_sizes',target_sample_sizes)



count_number_of_jobs = 0
for MAX_K, (is_masked_mlp, is_masked_sian), FIS_style, use_mnist in itertools.product(
    MAX_K_list, masked_combinations, FIS_style_list, use_mnist_scaling
):
    for dataset_str, preproc_owner in dataset_preproc_pairs:

        for TRN_N in target_sample_sizes:
            for BS in BS_list:

                job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}_BS{BS}_N{TRN_N}"
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
                json_data["STEP"] = 1024 * 1024 * 8
                json_data["BS"] = BS

                json_data["TARGET_TRN_N"] = TRN_N

                json_data["compute_shapeloss"] = compute_shapeloss
                json_data["LR"] = LR   #GCLOUD @ 6:50pm on 09/15/2025
        
                json_data["lambda1"] = lambda1
        
        
                if FIS_style == "maximal": 
                    save_json_and_shell(shell_save_dir, json_data, job_name, 'run_exp_longrun.py')
                    count_number_of_jobs += 1
                else:
                    raise NotImplementedError(f"FIS_style={FIS_style} not implemented right now")


print(f"Total number of jobs generated: {count_number_of_jobs}")

