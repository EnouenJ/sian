# please write code here for sweeping across these datasets
import itertools
import numpy as np
import os
import json
import copy

import sys
sys.path.append(".") #TODO: this does not work for me generally
from configurate import setup_config, save_json_and_shell, setup_debugging_config, CommonConfig, CommonDatasetConfigs


DEBUGGING_FLAG = setup_debugging_config()


class HYPERPARAMETERS:
    # dataset_preproc_pairs = [
    #     ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0", None),
    #     # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed1", None),
    #     # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed2", None),
    #     # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed3", None),
    #     # # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed4", None),
    # ]

    dataset_preproc_pairs = [
        CommonDatasetConfigs.synthetic_sweep_BS_LR_and_tester_datasets[0],
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

    type_of_experiment = "longrun"
    if DEBUGGING_FLAG:
        STEP = 20
    else:
        STEP = 1024 * 1024 * 8

    USE_LONG_JOB_NAME = CommonConfig.USE_LONG_JOB_NAME

shell_save_dir = CommonConfig.shell_save_dir
data_base_path = CommonConfig.data_base_path
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

# target_sample_sizes = [6400, 7611, 9051, 10763]
# BS_list = [128]
# MAX_K_list = [3]

compute_shapeloss = False



target_sample_sizes = np.round(np.exp( np.log(descent_point_2D_close) + np.linspace(-4.0,4.0,9)/8.0*np.log(2) )).astype(int)
target_sample_sizes = np.round(np.exp( np.log(descent_point_2D_close) + np.linspace(-2.0,2.0,5)/4.0*np.log(2) )).astype(int)
target_sample_sizes = np.round(np.exp( np.log(descent_point_2D_close) + np.linspace( 1.0,4.0,4)/4.0*np.log(2) )).astype(int)
target_sample_sizes = [int(trn_n) for trn_n in target_sample_sizes]
print('target_sample_sizes',target_sample_sizes)
 # [6400, 6979, 7611, 8300, 9051, 9870, 10763, 11738, 12800]


count_number_of_jobs = 0
for MAX_K, (is_masked_mlp, is_masked_sian), FIS_style, use_mnist in itertools.product(
    HYPERPARAMETERS.MAX_K_list, HYPERPARAMETERS.masked_combinations, HYPERPARAMETERS.FIS_style_list, HYPERPARAMETERS.use_mnist_scaling
):
    for dataset_str, preproc_owner in HYPERPARAMETERS.dataset_preproc_pairs:

        for TRN_N in target_sample_sizes:
            for BS in BS_list:
                if HYPERPARAMETERS.USE_LONG_JOB_NAME:
                    job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}_BS{BS}_N{TRN_N}"
                else:
                    job_name = f"exp_double_descent_job{count_number_of_jobs}"

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
                json_data["STEP"] = HYPERPARAMETERS.STEP
                json_data["BS"] = BS

                json_data["TARGET_TRN_N"] = TRN_N

                json_data["compute_shapeloss"] = compute_shapeloss

                json_data["type_of_experiment"] = HYPERPARAMETERS.type_of_experiment
                
                # Uncommenting for now @James. `sian_training_args.lambda1 = lambda1` in LongrunMixin -> run_longrun - Rahil
                json_data["lambda1"] = 0.0
        
        
                if FIS_style == "maximal": 
                    save_json_and_shell(shell_save_dir, json_data, job_name, 'run_specific_experiment.py')
                    count_number_of_jobs += 1
                else:
                    raise NotImplementedError(f"FIS_style={FIS_style} not implemented right now")


print(f"Total number of jobs generated: {count_number_of_jobs}")

