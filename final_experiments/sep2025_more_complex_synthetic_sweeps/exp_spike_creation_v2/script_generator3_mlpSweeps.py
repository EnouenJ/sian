import itertools
import os
import sys
import json
import numpy as np

sys.path.append(".")
from configurate import save_json_and_shell



dataset_preproc_pairs = [
    # ("SYNTH_simple_discrete_synthwave_v12_D10_Ik5_depth0_s3.9.27_conditionalHookerANOVA_strongestheredity_seed0", None), #testing
    ("SYNTH_simple_discrete_synthwave_v12_D20_Ik5_depth0_s3.9.27_conditionalHookerANOVA_nonhereditary_seed0", None)
]



BS_list = [128,]
compute_shapeloss = False
lambda1 = 0.0

STEP_list = [1024*1024//4]
# STEP_list = [2] #testing
LR_list = [5e-3]





target_sample_sizes = list((np.power(2.0, np.arange(-2, 11)) * 100).astype(int))
target_sample_sizes = list((np.power(2.0, np.arange(-4, 11)) * 100).astype(int))
target_sample_sizes = [int(trn_n) for trn_n in target_sample_sizes]
print('target_sample_sizes',target_sample_sizes)

##model_init_seed_list = [0,1,2] #TODO: merge model_init and master seeds
##print('model_init_seed_list',model_init_seed_list)



#FILL IN THE FIS ChOICES AND ADD HERE
class HYPERPARAMETERS:
    
    MAX_K_list = [1,2,3,5]

    use_mnist_scaling = ["smooth"]
    FIS_style_list = ["batchwise"]
    masked_combinations = [(True, False)]
    explainer_score_types = ["arch"]
    
    
    tau_dict_list = [ #weak hered
        {1: 1.0, 2: 0.5, 3: 0.33, 4: 0.25, 5: 0.20},
    ]

    ## theta_dict_list = [
    ##     {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.2, 5: 0.8},
    ## ]

    # rounds_dict_list = [
    #     {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
    #     # {1: 300, 2: 3, 3: 3, 4: 3, 5: 3}, #testing
    # ]

    # inters_dict_list = [ #SHOULD THIS BE A DICTIONARY?
    #     {1: 10, 2: 10, 3: 10, 4: 10, 5: 10},
    # ]

    rounds_dict_list = [
        {1: 20, 2: 20, 3: 20, 4: 20, 5: 20},
    ]

    inters_dict_list = [ 
        {1: 5, 2: 5, 3: 5, 4: 5, 5: 5},
    ]


ALL_SEEDS = [0,1,2]

shell_save_dir = "shell_scripts/"
data_base_path = "data/"
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)

count_number_of_jobs = 0


# for master_seed in ALL_SEEDS:
#     for ((is_masked_mlp, is_masked_sian), FIS_style, use_mnist, BS, LR, STEP, explainer_score_type,) in itertools.product(HYPERPARAMETERS.masked_combinations, HYPERPARAMETERS.FIS_style_list, HYPERPARAMETERS.use_mnist_scaling, HYPERPARAMETERS.BS_list, HYPERPARAMETERS.LR_list, HYPERPARAMETERS.STEP_list, HYPERPARAMETERS.explainer_score_types):
#         for dataset_str, preproc_owner in dataset_preproc_pairs:

for master_seed in ALL_SEEDS:
    for ((is_masked_mlp, is_masked_sian), FIS_style, use_mnist, BS, LR, STEP, explainer_score_type,) in itertools.product(HYPERPARAMETERS.masked_combinations, HYPERPARAMETERS.FIS_style_list, HYPERPARAMETERS.use_mnist_scaling, BS_list, LR_list, STEP_list, HYPERPARAMETERS.explainer_score_types):
        for dataset_str, preproc_owner in dataset_preproc_pairs:
            hyperparameter_combo_list = []

            for MAX_K in HYPERPARAMETERS.MAX_K_list:
                arange_K = list(range(1, MAX_K + 1))

                if FIS_style == "layerwise":
                    for tau_dict, theta_dict in itertools.product(HYPERPARAMETERS.tau_dict_list, HYPERPARAMETERS.theta_dict_list):
                        if all(k in tau_dict for k in arange_K) and all(k in theta_dict for k in arange_K):
                            hyperparam_str = (f"t{''.join(str(tau_dict[k]) for k in arange_K)}_th{''.join(str(theta_dict[k]) for k in arange_K)}")
                            hyperparameter_combo_list.append(
                                {
                                    "FIS_style": FIS_style,
                                    "MAX_K": MAX_K,
                                    "explainer_score_type": explainer_score_type,
                                    "tau_thresholds": tau_dict,
                                    "theta_thresholds": theta_dict,
                                    "hyperparam_str": hyperparam_str,
                                    "number_of_rounds": None,
                                    "inters_per_round": None,
                                }
                            )

                elif FIS_style == "batchwise":
                    for tau_dict, rounds_dict, inters_dict in itertools.product(HYPERPARAMETERS.tau_dict_list, HYPERPARAMETERS.rounds_dict_list, HYPERPARAMETERS.inters_dict_list):
                        if all(k in tau_dict for k in arange_K) and all(k in rounds_dict for k in arange_K) and all(k in inters_dict for k in arange_K):
                            number_of_rounds = rounds_dict[MAX_K]
                            inters_per_round = inters_dict[MAX_K]
                            hyperparam_str = (
                                f"t{''.join(str(tau_dict[k]) for k in arange_K)}"
                                f"_r{number_of_rounds}_i{inters_per_round}"
                            )
                            hyperparameter_combo_list.append(
                                {
                                    "FIS_style": FIS_style,
                                    "MAX_K": MAX_K,
                                    "explainer_score_type": explainer_score_type,
                                    "tau_thresholds": tau_dict,
                                    "number_of_rounds": number_of_rounds,
                                    "inters_per_round": inters_per_round,
                                    "hyperparam_str": hyperparam_str,
                                    "theta_thresholds": None,
                                }
                            )
                else:
                    raise NotImplementedError(f"FIS_style={FIS_style} not supported")

            json_data = {
                "data_base_path": data_base_path,
                "dataset_str": dataset_str,
                "preproc_owner": preproc_owner,
                "is_masked_mlp": is_masked_mlp,
                "is_masked_sian": is_masked_sian,
                "use_mnist_scaling": use_mnist,
                "BS": BS,
                "LR": LR,
                "STEP": STEP,
                "MAX_K_list" : HYPERPARAMETERS.MAX_K_list,
                "hyperparameter_combo_list": hyperparameter_combo_list,

                "master_seed" : master_seed,
                "target_sample_sizes" : target_sample_sizes,

                "lambda1" : lambda1,
                "compute_shapeloss" : compute_shapeloss,
            }

            # for combo in hyperparameter_combo_list:
            #     for key, value in combo.items():
            #         if key not in json_data:
            #             json_data[key] = value

            #job_name = (f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_BS{BS}_LR{LR}_ST{STEP}_MLP{'T' if is_masked_mlp else 'F'}_SN{'T' if is_masked_sian else 'F'}_MNT{'mn' if use_mnist == 'mnist' else 'sm'}_FIS{'l' if FIS_style == 'layerwise' else 'b'}_K{MAX_K}_EST{explainer_score_type}")
            #job_name = (f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_step12_BS{BS}_LR{LR}_ST{STEP}_MLP{'T' if is_masked_mlp else 'F'}_SN{'T' if is_masked_sian else 'F'}_MNT{'mn' if use_mnist == 'mnist' else 'sm'}_seed{master_seed}")
            job_name = f"{dataset_str}_step12_MLP{'m' if is_masked_mlp else 'u'}_SIAN{'m' if is_masked_sian else 'u'}_" + \
                        f"{use_mnist}Scal_FIS{FIS_style}_K{MAX_K}_BS{BS}_ST{STEP}_seed{master_seed}"
            # job_name = f"xxxx_seed{master_seed}"
            
            # save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name="run_exp_realworld.py")
            save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name="run_save_mlp_and_save_FIS.py")
            count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")