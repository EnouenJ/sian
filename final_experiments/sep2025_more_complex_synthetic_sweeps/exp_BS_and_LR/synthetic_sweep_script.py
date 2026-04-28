# please write code here for sweeping across these datasets
import itertools
import numpy as np
import os
import json


import sys
sys.path.append(".") #TODO: this does not work for me generally
from configurate import setup_config, save_json_and_shell, setup_debugging_config, CommonConfig, CommonDatasetConfigs


DEBUGGING_FLAG = setup_debugging_config()


class HYPERPARAMETERS:
    dataset_preproc_pairs = [
        CommonDatasetConfigs.synthetic_sweep_BS_LR_and_tester_datasets[0],
        CommonDatasetConfigs.synthetic_sweep_BS_LR_and_tester_datasets[1],
        CommonDatasetConfigs.synthetic_sweep_BS_LR_and_tester_datasets[2],
    ]

    # masked_combinations = [(False, False), (True, True)]  
    # use_mnist_scaling = ["mnist", "smooth"]
    # FIS_style_list = ["batchwise", "layerwise", "maximal"]

    masked_combinations = [(False, False), ]
    use_mnist_scaling = ["smooth", ]
    FIS_style_list = ["maximal"]

    BS_list = [8, 16, 32, 64, 128]
    LR_list = [1.25e-3, 2.5e-3, 5e-3, 10e-3, 20e-3]

    if DEBUGGING_FLAG:
        STEP_list = [10]
    else:
        STEP_list = [1024*256]

    MAX_K_list = [3, 2, 1] #better GPU scheduling

    # number_of_rounds_list_dict = {
    #     1 : [3, 6, 9],
    #     2 : [3, 6, 9],
    #     3 : [3, 6, 9],
    # }
    # inters_per_round_list_dict = {
    #     1 : [1],
    #     2 : [3],
    #     3 : [10],
    # }


    type_of_experiment = "synthetic_sweep_BS_and_LR"
    target_sample_sizes = [800, 900]

    USE_LONG_JOB_NAME = CommonConfig.USE_LONG_JOB_NAME


shell_save_dir = CommonConfig.shell_save_dir
data_base_path = CommonConfig.data_base_path
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)

# Generate all combinations for each dataset
count_number_of_jobs = 0
for MAX_K, (is_masked_mlp, is_masked_sian), FIS_style, use_mnist, BS, LR, STEP in itertools.product(
    HYPERPARAMETERS.MAX_K_list, HYPERPARAMETERS.masked_combinations, HYPERPARAMETERS.FIS_style_list, HYPERPARAMETERS.use_mnist_scaling, HYPERPARAMETERS.BS_list, HYPERPARAMETERS.LR_list, HYPERPARAMETERS.STEP_list
):
    for dataset_str, preproc_owner in HYPERPARAMETERS.dataset_preproc_pairs:       
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
        json_data["type_of_experiment"] = HYPERPARAMETERS.type_of_experiment
        json_data["target_sample_sizes"] = HYPERPARAMETERS.target_sample_sizes
    
        if FIS_style == "maximal": 
            pass
        else:
            raise NotImplementedError(f"FIS_style={FIS_style} not implemented right now")

        if HYPERPARAMETERS.USE_LONG_JOB_NAME:
            job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_BS{BS}_LR{LR}_STEP{STEP}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}"
        else:
            job_name = f"exp_BS_LR_job{count_number_of_jobs}"
            
        save_json_and_shell(shell_save_dir, json_data, job_name, 'run_specific_experiment.py')
        count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")

