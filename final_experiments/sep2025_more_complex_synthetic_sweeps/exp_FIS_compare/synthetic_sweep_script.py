import itertools
import numpy as np
import os
import json
import sys
sys.path.append(".")
from configurate import save_json_and_shell, setup_debugging_config, CommonConfig, CommonDatasetConfigs


DEBUGGING_FLAG = setup_debugging_config()


class HYPERPARAMETERS:
    dataset_preproc_pairs = CommonDatasetConfigs.FIS_compare_dataset
    
    masked_combinations = [(False, False), (True, True)]
    use_mnist_scaling = ["smooth", "mnist"]
    use_mnist_scaling = ["smooth"]
    FIS_style_list = ["layerwise", "batchwise"]
    FIS_style_list = ["batchwise"]
    explainer_score_types = ["arch", "inc", "rem"]
    explainer_score_types = ["arch"]
    
    BS_list = [32]
    LR_list = [5e-3]
    
    STEP_list = [1024*128]
    STEP_list = [10]
    MAX_K_list = [1, 2, 3, 4, 5]
    MAX_K_list = [5, 4, 3, 2, 1] #better GPU scheduling
    
    tau_dict_list = [
        {1: 1.0, 2: 0.5, 3: 0.33, 4: 0.67, 5: 0.8},
        {1: 0.9, 2: 0.3, 3: 0.1, 4: 0.33, 5: 0.8},
    ]
    
    theta_dict_list = [
        {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.2, 5: 0.8},
    ]
    
    rounds_dict_list = [
        {1: 3, 2: 3, 3: 3, 4: 5, 5: 1},
    ]
    
    inters_dict_list = [
        {1: 1, 2: 3, 3: 10, 4: 3, 5: 1},
    ]
    
    #JAM
    actual_MAX_K = 5
    theta_dict_list = []
    theta_dict = {}
    for k in range(actual_MAX_K):
        theta_dict[k+1] = 0.5
    theta_dict_list.append(theta_dict)
    theta_dict = {}
    for k in range(actual_MAX_K):
        theta_dict[k+1] = 0.5**(k+1)
    theta_dict_list.append(theta_dict)
    theta_dict = {}
    for k in range(actual_MAX_K):
        theta_dict[k+1] = 0.8**(k+1)
    theta_dict_list.append(theta_dict)
    
    tau_dict_list = []
    eps=1e-5
    tau_dict = {}
    for k in range(1, actual_MAX_K + 1):  # Prevent division by zero error
        tau_dict[k] = 1.0 / k - eps
    tau_dict_list.append(tau_dict)
    tau_dict = {}
    for k in range(actual_MAX_K):
        tau_dict[k+1] = 1.0
    tau_dict_list.append(tau_dict)
    tau_dict = {}
    for k in range(actual_MAX_K):
        tau_dict[k+1] = 0.5
    tau_dict_list.append(tau_dict)
    
    #{9 choose k} = 9, 36, 84, 126, 126
    rounds_dict_list = [
        # {1: 9, 2: 36, 3: 100, 4: 100, 5: 100},
        {1: 2, 2: 7, 3: 20, 4: 20, 5: 20},
    ]
    
    inters_dict_list = [
        # {1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        {1: 5, 2: 5, 3: 5, 4: 5, 5: 5},
    ]
    
    if DEBUGGING_FLAG:
        MAX_K_list = [1,2] #debugging
        STEP_list = [10] #debugging
    
    USE_LONG_JOB_NAME = CommonConfig.USE_LONG_JOB_NAME

    TARGET_SAMPLE_SIZES = [100, 800, 6400, 102400]

shell_save_dir = CommonConfig.shell_save_dir
data_base_path = CommonConfig.data_base_path
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)


count_number_of_jobs = 0
for MAX_K, (is_masked_mlp, is_masked_sian), FIS_style, use_mnist, BS, LR, STEP, explainer_score_type in itertools.product(
    HYPERPARAMETERS.MAX_K_list, HYPERPARAMETERS.masked_combinations, HYPERPARAMETERS.FIS_style_list, HYPERPARAMETERS.use_mnist_scaling, HYPERPARAMETERS.BS_list, HYPERPARAMETERS.LR_list, HYPERPARAMETERS.STEP_list, HYPERPARAMETERS.explainer_score_types
):
    for dataset_str, preproc_owner in HYPERPARAMETERS.dataset_preproc_pairs:
        hyperparameter_combo_list = []
        
        arange_K = list(range(1,MAX_K+1))
        if FIS_style == "layerwise":
            for tau_dict, theta_dict in itertools.product(HYPERPARAMETERS.tau_dict_list, HYPERPARAMETERS.theta_dict_list):
                if all(k in tau_dict for k in arange_K) and all(k in theta_dict for k in arange_K):
                    hyperparam_str = f"t{''.join(str(tau_dict[k]) for k in arange_K)}_th{''.join(str(theta_dict[k]) for k in arange_K)}"
                    hyperparam_combo_dict = {
                        "FIS_style": "layerwise",
                        "MAX_K": MAX_K,
                        "explainer_score_type": explainer_score_type,
                        "tau_thresholds": tau_dict,
                        "theta_thresholds": theta_dict,
                        "hyperparam_str": hyperparam_str,
                    }
                    hyperparameter_combo_list.append(hyperparam_combo_dict)
        
        elif FIS_style == "batchwise":
            for tau_dict, rounds_dict, inters_dict in itertools.product(HYPERPARAMETERS.tau_dict_list, HYPERPARAMETERS.rounds_dict_list, HYPERPARAMETERS.inters_dict_list):
                if all(k in tau_dict for k in arange_K) and all(k in rounds_dict for k in arange_K) and all(k in inters_dict for k in arange_K):
                    number_of_rounds = rounds_dict[MAX_K]
                    inters_per_round = inters_dict[MAX_K]
                    hyperparam_str = f"t{''.join(str(tau_dict[k]) for k in arange_K)}_r{number_of_rounds}_i{inters_per_round}"
                    hyperparam_combo_dict = {
                        "FIS_style": "batchwise",
                        "MAX_K": MAX_K,
                        "explainer_score_type": explainer_score_type,
                        "tau_thresholds": tau_dict,
                        "number_of_rounds": number_of_rounds,
                        "inters_per_round": inters_per_round,
                        "hyperparam_str": hyperparam_str
                    }
                    hyperparameter_combo_list.append(hyperparam_combo_dict)
        
        else:
            raise NotImplementedError(f"FIS_style={FIS_style} is not supported")
        
        json_data = {
            "data_base_path": data_base_path,
            "dataset_str": dataset_str,
            "preproc_owner": preproc_owner,
            "is_masked_mlp": is_masked_mlp,
            "use_mnist_scaling": use_mnist,
            "BS": BS,
            "LR": LR,
            "STEP": STEP,
            "hyperparameter_combo_list": hyperparameter_combo_list,
            "type_of_experiment": "compare_fis",
            "target_sample_sizes": HYPERPARAMETERS.TARGET_SAMPLE_SIZES,
        }
        
        if HYPERPARAMETERS.USE_LONG_JOB_NAME:
            job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_BS{BS}_LR{LR}_ST{STEP}_MLP{'T' if is_masked_mlp else 'F'}_SN{'T' if is_masked_sian else 'F'}_MNT{'mn' if use_mnist == 'mnist' else 'sm'}_FIS{'l' if FIS_style == 'layerwise' else 'b'}_K{MAX_K}_EST{explainer_score_type}"
        else:
            job_name = f"script_generator_SIAN_job{count_number_of_jobs}"
        
        save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name='run_specific_experiment.py')
        count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")