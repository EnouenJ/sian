import itertools
import os
import sys
import json
from datetime import datetime

sys.path.append(".")
from configurate import save_json_and_shell, setup_debugging_config, CommonConfig, CommonDatasetConfigs

DEBUGGING_FLAG = setup_debugging_config()

class HYPERPARAMETERS:
    if DEBUGGING_FLAG:
        dataset_preproc_pairs = CommonDatasetConfigs.realworld_debugging
    else:
        dataset_preproc_pairs = CommonDatasetConfigs.realworld_all_datasets

    masked_combinations = [(True, False)]
    use_mnist_scaling = ["smooth"]
    BS_list = [32]
    LR_list = [5e-3]
    STEP_list = [1024 * 1024 * 1]
    if DEBUGGING_FLAG:
        STEP_list = [10]

    MAX_K_list = [1, 2, 3, 5]
    MAX_K_list = [2]
    FIS_style_list = ["batchwise", "layerwise"]
    explainer_score_types = ["arch"]

    tau_dict_list = [
        {1: 1.0, 2: 0.5, 3: 0.33, 4: 0.25, 5: 0.20},
    ]
    theta_dict_list = [
        {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
    ]
    rounds_dict_list = [
        {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
    ]
    if DEBUGGING_FLAG:
        rounds_dict_list = [{1: 3, 2: 3, 3: 3, 4: 3, 5: 3}]

    inters_dict_list = [
        {1: 10, 2: 10, 3: 10, 4: 10, 5: 10},
    ]

    TARGET_SAMPLE_SIZES = [100, 800, "get_N"] if not DEBUGGING_FLAG else [100, "get_N"]
    TARGET_SAMPLE_SIZES = [100, "get_N"]

    ALL_SEEDS = [0, 1, 2] if not DEBUGGING_FLAG else [0]
    ALL_SEEDS = [0]

    USE_LONG_JOB_NAME = CommonConfig.USE_LONG_JOB_NAME

shell_save_dir = CommonConfig.shell_save_dir
data_base_path = CommonConfig.data_base_path
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)

count_number_of_jobs = 0


for master_seed in HYPERPARAMETERS.ALL_SEEDS:
    for (is_masked_mlp, is_masked_sian), use_mnist, BS, LR, STEP, FIS_style, explainer_score_type in itertools.product(
        HYPERPARAMETERS.masked_combinations,
        HYPERPARAMETERS.use_mnist_scaling,
        HYPERPARAMETERS.BS_list,
        HYPERPARAMETERS.LR_list,
        HYPERPARAMETERS.STEP_list,
        HYPERPARAMETERS.FIS_style_list,
        HYPERPARAMETERS.explainer_score_types
    ):
        for dataset_str, preproc_owner in HYPERPARAMETERS.dataset_preproc_pairs:
            hyperparameter_combo_list = []

            for MAX_K in HYPERPARAMETERS.MAX_K_list:
                arange_K = list(range(1, MAX_K + 1))

                if FIS_style == "layerwise":
                    for tau_dict, theta_dict in itertools.product(
                        HYPERPARAMETERS.tau_dict_list,
                        HYPERPARAMETERS.theta_dict_list
                    ):
                        if not (all(k in tau_dict for k in arange_K) and
                                all(k in theta_dict for k in arange_K)):
                            continue
                        hyperparam_str = (
                            f"t{''.join(f'{tau_dict[k]:.2f}'[:4] for k in arange_K)}"
                            f"_th{''.join(f'{theta_dict[k]:.2f}'[:4] for k in arange_K)}"
                        )
                        hyperparameter_combo_list.append({
                            "FIS_style": FIS_style,
                            "MAX_K": MAX_K,
                            "explainer_score_type": explainer_score_type,
                            "tau_thresholds": tau_dict,
                            "theta_thresholds": theta_dict,
                            "number_of_rounds": None,
                            "inters_per_round": None,
                            "hyperparam_str": hyperparam_str,
                        })

                elif FIS_style == "batchwise":
                    for tau_dict, rounds_dict, inters_dict in itertools.product(
                        HYPERPARAMETERS.tau_dict_list,
                        HYPERPARAMETERS.rounds_dict_list,
                        HYPERPARAMETERS.inters_dict_list
                    ):
                        if not (all(k in tau_dict for k in arange_K) and
                                all(k in rounds_dict for k in arange_K) and
                                all(k in inters_dict for k in arange_K)):
                            continue

                        number_of_rounds = rounds_dict[MAX_K]
                        inters_per_round = inters_dict[MAX_K]
                        hyperparam_str = (
                            f"t{''.join(f'{tau_dict[k]:.2f}'[:4] for k in arange_K)}"
                            f"_r{number_of_rounds}_i{inters_per_round}"
                        )

                        hyperparameter_combo_list.append({
                            "FIS_style": FIS_style,
                            "MAX_K": MAX_K,
                            "explainer_score_type": explainer_score_type,
                            "tau_thresholds": tau_dict,
                            "number_of_rounds": number_of_rounds,
                            "inters_per_round": inters_per_round,
                            "theta_thresholds": None,
                            "hyperparam_str": hyperparam_str,
                        })

                else:
                    raise ValueError(f"Unknown FIS_style: {FIS_style}")

            if not hyperparameter_combo_list:
                continue

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
                "target_sample_sizes": HYPERPARAMETERS.TARGET_SAMPLE_SIZES,
                "hyperparameter_combo_list": hyperparameter_combo_list,
                "master_seed": master_seed,
                "type_of_experiment": "real_world_dataset_experiment",
            }

            if HYPERPARAMETERS.USE_LONG_JOB_NAME:
                job_name = (f"REALWORLD_{dataset_str}_PO{preproc_owner or 'none'}_BS{BS}_LR{LR:.5f}_ST{STEP}_MLP{'T' if is_masked_mlp else 'F'}_SIAN{'T' if is_masked_sian else 'F'}_MNT{use_mnist[:2]}_FIS{FIS_style[0]}_seed{master_seed}")
            else:
                job_name = f"realworld_job_{count_number_of_jobs}"

            save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name="run_specific_experiment.py")
            count_number_of_jobs += 1

print(f"\nTotal jobs generated: {count_number_of_jobs}")
print(f"Shell scripts saved in: {shell_save_dir}")