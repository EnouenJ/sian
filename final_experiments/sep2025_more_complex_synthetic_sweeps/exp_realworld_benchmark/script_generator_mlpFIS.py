import itertools
import os
import sys
import json

sys.path.append(".")
from configurate import save_json_and_shell, setup_debugging_config, CommonConfig, CommonDatasetConfigs




DEBUGGING_FLAG = setup_debugging_config()



#FILL IN THE FIS ChOICES AND ADD HERE
class HYPERPARAMETERS:
    if DEBUGGING_FLAG:
        dataset_preproc_pairs = CommonDatasetConfigs.realworld_debugging
    else:
        dataset_preproc_pairs = CommonDatasetConfigs.realworld_all_datasets
        
    # dataset_preproc_pairs = [
    #     ("UCI_275_bike_sharing_dataset", None),
    #     # ("otherSource_cal_housing", None),
    #     # ("UCI_186_wine_quality", None),
    #     # ("UCI_374_appliances_energy_prediction", None),
        
    #     # ("UCI_31_tree_cover_type_dataset", None),
    #     # ("UCI_203_yearpredictionMSD", None),
    #     # ("UCI_280_higgs_boson_dataset", None),
        
    #     # ("UCI_2_adults_dataset", None),
    #     # ("UCI_1_abalone_dataset", None),
    #     # ("UCI_332_online_news_popularity", None),
    #     # ("UCI_171_madelon_dataset", None),
    #     # ("UCI_572_taiwanese_bankruptcy_dataset", None),

    #     # ("otherSource_Microsoft_search_queries", None),
    #     # ("Kaggle_blastchar_telco_customer_churn", None),
    #     # ("Kaggle_mlgulb_credit_card_fraud_dataset", None),
    #     # ("Kaggle_ishadss_eucalyptus_dataset", None),
    # ]
    
    MAX_K_list = [1,2,3,5]

    use_mnist_scaling = ["smooth"]
    FIS_style_list = ["batchwise"]
    masked_combinations = [(True, False)]
    explainer_score_types = ["arch"]

    BS_list = [32]
    # BS_list = [128]
    LR_list = [5e-3]
    # STEP_list = [1024*1024*1]
    # STEP_list = [1024*1024*8]
    STEP_list = [1024*1024*1]
    if DEBUGGING_FLAG:
        STEP_list = [10]

    
    tau_dict_list = [ #weak hered
        {1: 1.0, 2: 0.5, 3: 0.33, 4: 0.25, 5: 0.20},
    ]

    ## theta_dict_list = [
    ##     {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.2, 5: 0.8},
    ## ]

    rounds_dict_list = [
        {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
    ]
    if DEBUGGING_FLAG:
        rounds_dict_list = [
            {1: 300, 2: 3, 3: 3, 4: 3, 5: 3}, #debugging
        ]

    inters_dict_list = [ #SHOULD THIS BE A DICTIONARY?
        {1: 10, 2: 10, 3: 10, 4: 10, 5: 10},
    ]

    USE_LONG_JOB_NAME = CommonConfig.USE_LONG_JOB_NAME


ALL_SEEDS = [0,1,2]

shell_save_dir = CommonConfig.shell_save_dir
data_base_path = CommonConfig.data_base_path
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)

count_number_of_jobs = 0


for master_seed in ALL_SEEDS:
    for ((is_masked_mlp, is_masked_sian), FIS_style, use_mnist, BS, LR, STEP, explainer_score_type,) in itertools.product(HYPERPARAMETERS.masked_combinations, HYPERPARAMETERS.FIS_style_list, HYPERPARAMETERS.use_mnist_scaling, HYPERPARAMETERS.BS_list, HYPERPARAMETERS.LR_list, HYPERPARAMETERS.STEP_list, HYPERPARAMETERS.explainer_score_types):
        for dataset_str, preproc_owner in HYPERPARAMETERS.dataset_preproc_pairs:
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
                "hyperparameter_combo_list": hyperparameter_combo_list,
                "master_seed" : master_seed,
                "type_of_experiment": "train_mlp_and_fis",
            }

            # for combo in hyperparameter_combo_list:
            #     for key, value in combo.items():
            #         if key not in json_data:
            #             json_data[key] = value

            #job_name = (f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_BS{BS}_LR{LR}_ST{STEP}_MLP{'T' if is_masked_mlp else 'F'}_SN{'T' if is_masked_sian else 'F'}_MNT{'mn' if use_mnist == 'mnist' else 'sm'}_FIS{'l' if FIS_style == 'layerwise' else 'b'}_K{MAX_K}_EST{explainer_score_type}")

            if HYPERPARAMETERS.USE_LONG_JOB_NAME:
                job_name = (f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_step12_BS{BS}_LR{LR}_ST{STEP}_MLP{'T' if is_masked_mlp else 'F'}_SN{'T' if is_masked_sian else 'F'}_MNT{'mn' if use_mnist == 'mnist' else 'sm'}_seed{master_seed}")
            else:
                job_name = (f"script_generator_MLP_FIS_job{count_number_of_jobs}")

            # save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name="run_exp_realworld.py")
            save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name="run_specific_experiment.py")
            count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")