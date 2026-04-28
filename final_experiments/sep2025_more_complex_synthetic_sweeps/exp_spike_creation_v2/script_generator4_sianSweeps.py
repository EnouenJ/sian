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
    # FIS_style_list = ["batchwise"]
    masked_combinations = [(True, False)]
    # explainer_score_types = ["arch"]
    
    


ALL_SEEDS = [0,1,2]

shell_save_dir = "shell_scripts/"
data_base_path = "data/"
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)






# hardcoded_fis_json_locations = {
#     0 : "mlp_and_fis_good_runs_seed_0.json",
#     1 : "mlp_and_fis_good_runs_seed_1.json",
#     2 : "mlp_and_fis_good_runs_seed_2.json",
# }
hardcoded_fis_json_location = "results_sian_review_experiments_hyperparam_sweep_sep30th_mlpAndFisSweeps.json"




count_number_of_jobs = 0

for master_seed in ALL_SEEDS:
    for ((is_masked_mlp, is_masked_sian), use_mnist, BS, LR, STEP) in itertools.product(HYPERPARAMETERS.masked_combinations, HYPERPARAMETERS.use_mnist_scaling, BS_list, LR_list, STEP_list):
        for dataset_str, preproc_owner in dataset_preproc_pairs:
            for MAX_K in HYPERPARAMETERS.MAX_K_list:
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
                    # Hardcoded FIS experiment path - Rahil
                    # "fis_location": "/home/rahil/sian-private-SIAN-expanding-experiments_sep22_james_meet/sian-private-SIAN-expanding-experiments/final_experiments/sep2025_more_complex_synthetic_sweeps/results/20250923_185141_demo_simple_testing/20250923_185141_N800_batchwise_K2_arch_t1.00.5_r100_i10results.json"
                    # "fis_json_location": hardcoded_fis_json_locations[master_seed],
                    "fis_json_location" : hardcoded_fis_json_location,
                    "MAX_K": MAX_K,
                        

                    "master_seed" : master_seed,
                    "target_sample_sizes" : target_sample_sizes,

                    "lambda1" : lambda1,
                    "compute_shapeloss" : compute_shapeloss,
                }

                # job_name = (
                #     f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_step3_BS{BS}_LR{LR}_ST{STEP}"
                #     f"_MLP{'T' if is_masked_mlp else 'F'}_SN{'T' if is_masked_sian else 'F'}_MNT{'mn' if use_mnist == 'mnist' else 'sm'}"
                # )
                # job_name = (
                #     f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_step3_BS{BS}_LR{LR}_ST{STEP}"
                #     f"_MLP{'T' if is_masked_mlp else 'F'}_SN{'T' if is_masked_sian else 'F'}_MNT{'mn' if use_mnist == 'mnist' else 'sm'}_K{MAX_K}_seed{master_seed}"
                # )
                job_name = f"{dataset_str}_step3_MLP{'m' if is_masked_mlp else 'u'}_SIAN{'m' if is_masked_sian else 'u'}_" + \
                            f"{use_mnist}Scal_K{MAX_K}_BS{BS}_ST{STEP}_seed{master_seed}"
                # job_name = f"xxxx_K{MAX_K}_seed{master_seed}"

                save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name="run_save_sian.py")
                count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")