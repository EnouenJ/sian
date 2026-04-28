import itertools
import os
import sys
import json

sys.path.append(".")
from configurate import save_json_and_shell, setup_debugging_config, CommonConfig, CommonDatasetConfigs


DEBUGGING_FLAG = setup_debugging_config()


class HYPERPARAMETERS:
    if DEBUGGING_FLAG:
        dataset_preproc_pairs = CommonDatasetConfigs.realworld_debugging
    else:
        dataset_preproc_pairs = CommonDatasetConfigs.realworld_all_datasets
    
    use_mnist_scaling = ["smooth"]
    masked_combinations = [(True, False)]
    BS_list = [32]
    BS_list = [128]
    LR_list = [5e-3]
    STEP_list = [10] #testing
    STEP_list = [1024*1024*1]
    STEP_list = [1024*1024//8]
    STEP_list = [10]

    MAX_K_list = [1,2,3,5]
    MAX_K_list = [1]

    USE_LONG_JOB_NAME = CommonConfig.USE_LONG_JOB_NAME

    TYPE_OF_EXPERIMENT = "train_sian_sweep"
    if TYPE_OF_EXPERIMENT == "train_sian_benchmark":
        TARGET_SAMPLE_SIZES = ["get_N"]
    else:
        TARGET_SAMPLE_SIZES = [100, 800, "get_N"]


ALL_SEEDS = [0,1,2]
ALL_SEEDS = [0] #partial results




shell_save_dir = CommonConfig.shell_save_dir
data_base_path = CommonConfig.data_base_path
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)




hardcoded_fis_json_locations = {
    0 : "mlp_and_fis_good_runs_seed_0.json",
    1 : "mlp_and_fis_good_runs_seed_1.json",
    2 : "mlp_and_fis_good_runs_seed_2.json",
}

# Can be adjust based on the dataset - Rahil
target_sample_sizes = [50, 100, 1000]


count_number_of_jobs = 0

for master_seed in ALL_SEEDS:
    for ((is_masked_mlp, is_masked_sian), use_mnist, BS, LR, STEP) in itertools.product(
        HYPERPARAMETERS.masked_combinations,
        HYPERPARAMETERS.use_mnist_scaling,
        HYPERPARAMETERS.BS_list,
        HYPERPARAMETERS.LR_list,
        HYPERPARAMETERS.STEP_list
    ):
        for dataset_str, preproc_owner in HYPERPARAMETERS.dataset_preproc_pairs:
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
                    "fis_json_location": hardcoded_fis_json_locations[master_seed],
                    "MAX_K": MAX_K,
                    "type_of_experiment": HYPERPARAMETERS.TYPE_OF_EXPERIMENT,
                    "target_sample_sizes": HYPERPARAMETERS.TARGET_SAMPLE_SIZES,
                    "master_seed" : master_seed,
                }

                if HYPERPARAMETERS.USE_LONG_JOB_NAME:
                    job_name = (
                        f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_step3_BS{BS}_LR{LR}_ST{STEP}"
                        f"_MLP{'T' if is_masked_mlp else 'F'}_SN{'T' if is_masked_sian else 'F'}_MNT{'mn' if use_mnist == 'mnist' else 'sm'}_K{MAX_K}_seed{master_seed}"
                    )
                else:
                    job_name = (f"script_generator_real_world_sweep_SIAN_job{count_number_of_jobs}")

                save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name="run_specific_experiment.py")
                count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")