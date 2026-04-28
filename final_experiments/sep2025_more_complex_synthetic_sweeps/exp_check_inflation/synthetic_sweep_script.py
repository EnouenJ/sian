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
    dataset_preproc_pairs = CommonDatasetConfigs.check_inflation_dataset

    # masked_combinations = [(False, False), (True, True)]  
    # use_mnist_scaling = ["mnist", "smooth"]
    # FIS_style_list = ["batchwise", "layerwise", "maximal"]

    masked_combinations = [(False, False), ]
    use_mnist_scaling = ["smooth", ]
    FIS_style_list = ["maximal"]

    BS_list = [8, 16, 32, 64, 128]
    BS_list = [8, 32, 128]
    LR_list = [5e-3]
    # STEP_list = [32]
    STEP_list = [8]

    MAX_K_list = [3, 2, 1] #better GPU scheduling
    MAX_K_list = [2]

    max_part_size_list = list((np.power(2.0, np.arange(0, 21)/2) * 8).astype(int))
    max_part_size_list = list((np.power(2.0, np.arange(0, 15)/2) * 8).astype(int))

    ##max_part_size_list = list((np.power(2.0, np.arange(1, 21)/2) * 1).astype(int))
    ##max_part_size_list = [int(x) for x in max_part_size_list]
    ##max_part_size_list = list((np.power(2.0, np.arange(4, 21)/2) * 1).astype(int))
    max_part_size_list = np.round((np.power(2.0, np.arange(4, 21)/2) * 1))
    max_part_size_list = [int(x) for x in max_part_size_list]
    max_part_size_list = [1,2] + max_part_size_list

    #09/18/25 @ 11:30pm
    max_inters_list = [100,300,1000]
    BS_list = [32, 128]
    max_part_size_list = np.round((np.power(2.0, np.arange(0, 11)) * 1))
    max_part_size_list = [int(x) for x in max_part_size_list]
    STEP_list = [4]

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

        for max_part_size in HYPERPARAMETERS.max_part_size_list:
            for max_number_of_interactions in HYPERPARAMETERS.max_inters_list:
                
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
                
                json_data["max_part_size"] = max_part_size
                json_data["max_number_of_interactions"] = max_number_of_interactions

                json_data["target_sample_sizes"] = [100, 1000, 10000]
                json_data["type_of_experiment"] = "test_inflation"
            
                if FIS_style == "maximal": 
                    pass
                else:
                    raise NotImplementedError(f"FIS_style={FIS_style} not implemented right now")

                if HYPERPARAMETERS.USE_LONG_JOB_NAME:
                    job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_BS{BS}_LR{LR}_STEP{STEP}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}"
                    job_name += f"_part{max_part_size}_inters{max_number_of_interactions}"
                else:
                    job_name = f"inflation_job{count_number_of_jobs}"

                save_json_and_shell(shell_save_dir, json_data, job_name, 'run_specific_experiment.py')
                count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")

