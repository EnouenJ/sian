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
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik5_depth0_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik5_depth1_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik5_depth2_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik10_depth0_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik10_depth1_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik10_depth2_s3.9.27_conditionalHookerANOVA_seed0", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed1", None),
    # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed2", None),
    # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed3", None),
    # ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed4", None),
]



masked_combinations = [(False, False), (True, True)]  
use_mnist_scaling = ["mnist", "smooth"]
FIS_style_list = ["batchwise", "layerwise", "maximal"]
FIS_style_list = ["maximal"]

# masked_combinations = [(False, False), ]
# use_mnist_scaling = ["smooth", ]
# FIS_style_list = ["maximal"]

MAX_K_list = [3, 2, 1] #better GPU scheduling

number_of_rounds_list_dict = {
    1 : [3, 6, 9],
    2 : [3, 6, 9],
    3 : [3, 6, 9],
}
inters_per_round_list_dict = {
    1 : [1],
    2 : [3],
    3 : [10],
}


if True:
    masked_combinations = [(True, True)]  
    use_mnist_scaling = ["smooth"]
    FIS_style_list = ["batchwise"]
    
if True:
    masked_combinations = [(False, False)]  
    use_mnist_scaling = ["mnist"]
    FIS_style_list = ["layerwise"]
    MAX_K_list = [2] 



shell_save_dir = "shell_scripts/"
data_base_path = "data/"
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)

# Generate all combinations for each dataset
count_number_of_jobs = 0
for MAX_K, (is_masked_mlp, is_masked_sian), FIS_style, use_mnist in itertools.product(
    MAX_K_list, masked_combinations, FIS_style_list, use_mnist_scaling
):
    for dataset_str, preproc_owner in dataset_preproc_pairs:

        job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}"
        job_name="XDDDDD_tooLongForWindows"
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
        json_data["STEP"] = 4
    
    
        if FIS_style == "batchwise":
            number_of_rounds_list = number_of_rounds_list_dict.get(MAX_K, [])
            inters_per_round_list = inters_per_round_list_dict.get(MAX_K, [])
            for number_of_rounds in number_of_rounds_list:
                for inters_per_round in inters_per_round_list:
                    job_name2 = job_name + f"_Rounds{number_of_rounds}_IntersPerRound{inters_per_round}"
                    json_data2 = copy.deepcopy(json_data)
                    json_data2["number_of_rounds"] = number_of_rounds
                    json_data2["inters_per_round"] = inters_per_round

                    save_json_and_shell(shell_save_dir, json_data2, job_name2, 'run_exp_synthsweep.py')
                    count_number_of_jobs += 1
                    pass
            number_of_rounds=number_of_rounds,
            inters_per_round=inters_per_round,
        elif FIS_style == "layerwise":
            #TODO: still need to implement, this will just run default theta params
            save_json_and_shell(shell_save_dir, json_data, job_name, 'run_exp_synthsweep.py')
            count_number_of_jobs += 1
            # raise NotImplementedError("need to implement")
        elif FIS_style == "maximal": 
            save_json_and_shell(shell_save_dir, json_data, job_name, 'run_exp_synthsweep.py')
            count_number_of_jobs += 1
        else:
            raise NotImplementedError(f"FIS_style={FIS_style} not implemented right now")


print(f"Total number of jobs generated: {count_number_of_jobs}")

