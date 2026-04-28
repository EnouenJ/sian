import os
import copy
import time
import pickle
import sys
from ..configurate import setup_config
CONFIGURATION_FLAG = setup_config() 

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from sian.utils import gettimestamp
from sian.data import Final_TabularDataset
from sian.models import TrainingArgs
from sian.fis import layerwise_FIS_Hyperparameters, batchwise_FIS_Hyperparameters
from sian.interpret import unmasked_FID_Hyperparameters, masked_FID_Hyperparameters
from sian import initalize_the_explainer
from sian import train_mlp_final, do_the_fis_final, train_sian_final
from sian.interpret import plot_all_GAM_functions
from itertools import combinations

import json
import argparse


BS = 32
EP = 100
LR = 5e-3

#JAM: stability when we change the dataset sizes (this varies steps according to epochs)
STEP = 1024*16; EP=None
STEP = 1024*256; EP=None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_training_args = TrainingArgs(BS, EP, LR, device, STEP)
sian_training_args = TrainingArgs(BS, EP, LR, device, STEP)


parser = argparse.ArgumentParser(description="SIAN Model Training and Interpretation")
parser.add_argument(
    '--dataset_str',
    type=str,
    default="UCI_275_bike_sharing_dataset",
    help="Dataset name (default: UCI_275_bike_sharing_dataset)"
)
parser.add_argument(
    '--preproc_owner',
    type=str,
    default=None,
    help="Preprocessing owner identifier (default: None)"
)
parser.add_argument(
    '--is_masked_mlp',
    type=lambda s: s.lower() == 'true',
    default=True,
    help="Whether to use masked MLP (default: True)"
)
parser.add_argument(
    '--is_masked_sian',
    type=lambda s: s.lower() == 'true',
    default=True,
    help="Whether to use masked SIAN (default: True)"
)
parser.add_argument(
    '--use_mnist_scaling',
    type=lambda s: s.lower() if s.lower() in ['mnist', 'smooth'] else s,
    default="smooth",
    help="Scaling mode: 'mnist' or 'smooth' (default: 'smooth')"
)
parser.add_argument(
    '--FIS_style',
    type=str,
    choices=['batchwise', 'layerwise', 'maximal'],
    default='layerwise',
    help="FIS style: batchwise or layerwise (default: layerwise)"
)
parser.add_argument(
    '--MAX_K',
    type=int,
    default=5,
    help="Maximum K value (default: 5)"
)
parser.add_argument(
    '--number_of_rounds',
    type=int,
    default=None,
    help="Number of rounds for batchwise FIS (default: None)"
)
parser.add_argument(
    '--inters_per_round',
    type=int,
    default=None,
    help="Number of interactions per round in batchwise FIS (default: None)"
)

args = parser.parse_args()
dataset_str = args.dataset_str
preproc_owner = args.preproc_owner if args.preproc_owner != "None" else None
is_masked_mlp = args.is_masked_mlp
is_masked_sian = args.is_masked_sian
use_mnist_scaling = args.use_mnist_scaling
FIS_style = args.FIS_style
MAX_K = args.MAX_K
number_of_rounds = args.number_of_rounds
inters_per_round = args.inters_per_round

print("Dataset used:", dataset_str)
print("Preprocessing Owner:", preproc_owner)
if FIS_style == "batchwise":
    print("Number of rounds:", number_of_rounds)
    print("Interactions per round:", inters_per_round)

data_base_path = "data/"
load_dataset_path = data_base_path
save_dataset_path = data_base_path + dataset_str + "/"

cwd = os.getcwd()
results_path = cwd + "/" + "results/"
exp_datetimestr = gettimestamp()
if dataset_str.startswith("SYNTH_"):
    exp_folder = results_path + exp_datetimestr + '_' + "synthetic_sampleComplexity_sweep/"
else:
    exp_folder = results_path + exp_datetimestr + '_' + "demo" + '_simple_testing/'
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)
print(exp_folder)

BASE_TRN_N = 700 * 1000
target_sample_sizes = list((np.power(2.0, np.arange(-2, 11)) * 100).astype(int))
print('target_sample_sizes', target_sample_sizes)

mlp_training_args.model_config.net_name = "MLP"
mlp_training_args.model_config.sizes = [-1, 256, 128, 64, -1]
mlp_training_args.model_config.is_masked = is_masked_mlp
mlp_training_args.model_config.use_mnist_scaling = use_mnist_scaling
mlp_training_args.saving_settings.exp_folder = exp_folder
mlp_training_args.saving_settings.results_to_save = ['best_net','final_net']
mlp_training_args.return_settings.return_val_tensor = True
mlp_training_args.return_settings.return_final_net = True
mlp_training_args.return_settings.return_best_net = True
mlp_training_args.EVALUATE_EVERY_K_EPOCHS = None
mlp_training_args.VERBOSE_TRAINING = None

sian_training_args.model_config.net_name = "SIAN-K"
sian_training_args.model_config.sizes = [-1, 256, 128, 64, -1]
sian_training_args.model_config.small_sizes = [-1, 32, 24, 16, -1]
sian_training_args.model_config.is_masked = is_masked_sian
sian_training_args.model_config.use_mnist_scaling = use_mnist_scaling
sian_training_args.saving_settings.exp_folder = exp_folder
sian_training_args.saving_settings.results_to_save = ['best_net','final_net']
sian_training_args.return_settings.return_val_tensor = True
sian_training_args.return_settings.return_final_net = True
sian_training_args.return_settings.return_best_net = True
sian_training_args.EVALUATE_EVERY_K_EPOCHS = None
sian_training_args.VERBOSE_TRAINING = None


if os.path.exists(save_dataset_path):
    print(f"Loading existing dataset from {save_dataset_path}")
else:
    print(f"Creating new dataset at {save_dataset_path}")

dataset_obj = Final_TabularDataset(
    dataset_str,
    preproc_owner=preproc_owner,
    load_dataset_path=load_dataset_path,
    save_dataset_path=save_dataset_path
)

D = dataset_obj.get_D()
readable_labels = dataset_obj.get_readable_labels()
print(readable_labels)


all_tau1_values = [1.0]
all_tau2_values = [0.5]
all_tau3_values = [0.33]

all_theta1_perc_values = [0.2,] #RAHIL RAHIL RAHIL, I think this is perhaps a 'bug' for the layerwise, let's turn it off for now
all_theta2_perc_values = [0.2]
all_theta3_perc_values = [0.6]

tau_values_list = []
theta_values_list = []
if MAX_K >= 1:
    tau_values_list.append(all_tau1_values)
    theta_values_list.append(all_theta1_perc_values)
if MAX_K >= 2:
    tau_values_list.append(all_tau2_values)
    theta_values_list.append(all_theta2_perc_values)
if MAX_K >= 3:
    tau_values_list.append(all_tau3_values)
    theta_values_list.append(all_theta3_perc_values)



if FIS_style == "layerwise":
    hyperparam_combinations = list(itertools.product(*tau_values_list, *theta_values_list))
else:
    hyperparam_combinations = list(itertools.product(*tau_values_list))

print(hyperparam_combinations)
print(len(hyperparam_combinations))


# for TARGET_TRN_N in [102400]:
# for TARGET_TRN_N in [100, 800, 102400]:
# for TARGET_TRN_N in target_sample_sizes:
for TARGET_TRN_N in [100, 6400, 102400]:
    print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
    trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
    mlp_training_args.trainval_reduction_percentage = trn_reduc_perc
    sian_training_args.trainval_reduction_percentage = trn_reduc_perc

    dataset_obj = Final_TabularDataset(
        dataset_str,
        preproc_owner=preproc_owner,
        load_dataset_path=load_dataset_path,
        save_dataset_path=save_dataset_path
    )
    
    # sian_training_args.model_config.model_init_seed = model_init_seed #NOTE: JAM -- isnt this also necessary to set? even if we reuse it?



    output_type = dataset_obj.get_task_type()
    grouped_features_dict = dataset_obj.get_grouped_feature_dict()
    if FIS_style in ["layerwise", "batchwise"]: #08/21/2025
        print(10 * "=", f"Training MLP with sample size {TARGET_TRN_N}", 10 * "=")
        mlp_results = train_mlp_final(dataset_obj, mlp_training_args)
        trained_mlp = mlp_results["best_net"]
        val_tensor = mlp_results["val_tensor"]

        # output_type = dataset_obj.get_task_type()
        # grouped_features_dict = dataset_obj.get_grouped_feature_dict()
        if is_masked_mlp:
            fid_masking_style = "masking_based"
            print("FID masking style", fid_masking_style)
            score_type_name = "new_arch_inter_sobol_score"
            inc_rem_pel_list = ['inc_inter_sobol_score', 'rem_inter_sobol_score', 'new_arch_inter_sobol_score']
            fis_valX = val_tensor
            my_FID_hypers = masked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list,
                                                    grouped_features_dict)
        else:
            fid_masking_style = "triangle_marginal"
            print("FID masking style", fid_masking_style)
            score_type_name = "old_arch_inter_score"
            inc_rem_pel_list = ['inc_inter_score', 'rem_inter_score', 'old_arch_inter_score']
            fis_valX = val_tensor.detach().cpu().numpy()
            my_FID_hypers = unmasked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list, device, grouped_features_dict)
    elif FIS_style == "maximal":
        mlp_training_args.number_of_epochs = 0
        mlp_training_args.number_of_steps = None
        mlp_results = train_mlp_final(dataset_obj, mlp_training_args)
    else:
        raise Exception(f"FIS_style={FIS_style} not recognized")



    for hyperparam_idx, hyperparam_combo in enumerate(hyperparam_combinations):
        datetimestr = gettimestamp()
        if FIS_style in ["layerwise", "batchwise"]: #08/21/2025
            if FIS_style == "layerwise":
                tau_values = hyperparam_combo[:MAX_K]
                theta_values = hyperparam_combo[MAX_K:]
                tau_thresholds = {k+1: tau_values[k] for k in range(min(MAX_K, len(tau_values)))}
                theta_thresholds = {k+1: theta_values[k] for k in range(min(MAX_K, len(theta_values)))}
                hyperparam_str = f"tau{'_'.join(str(t) for t in tau_values)}_theta{'_'.join(str(t) for t in theta_values)}"
                my_FIS_hypers = layerwise_FIS_Hyperparameters(
                    MAX_K, tau_thresholds, theta_thresholds, None, theta_percentile_mode=True
                )
            elif FIS_style == "batchwise":
                tau_values = hyperparam_combo
                tau_thresholds = {k+1: tau_values[k] for k in range(min(MAX_K, len(tau_values)))}
                if number_of_rounds is None or inters_per_round is None:
                    raise ValueError("number_of_rounds and inters_per_round must be specified for batchwise FIS")
                hyperparam_str = f"tau{'_'.join(str(t) for t in tau_values)}_rounds{number_of_rounds}_inters_per_round{inters_per_round}"
                my_FIS_hypers = batchwise_FIS_Hyperparameters(
                    MAX_K, tau_thresholds, number_of_rounds=number_of_rounds, interactions_per_round=inters_per_round,
                    explainer=None, tuples_initialization=None, pick_underlings=False, fill_underlings=False, PLOTTING=True)

            sian_paths_prefix = f"{exp_folder}{datetimestr}_MAXK{MAX_K}_TRN_N{TARGET_TRN_N}_{hyperparam_str}_"
            saved_results_path = sian_paths_prefix + 'results.json'
            print(f"Processing hyperparameter combination {hyperparam_idx+1}/{len(hyperparam_combinations)} for sample size {TARGET_TRN_N}: {hyperparam_str}")
            print(sian_paths_prefix)

            jam_arch = initalize_the_explainer(trained_mlp, my_FID_hypers)
            my_FIS_hypers.add_the_explainer(jam_arch)

            FIS_algorithm_start_time = time.time()
            FIS_interactions = do_the_fis_final(my_FIS_hypers, fis_valX, AGG_K=100)
            FIS_algorithm_time_taken = time.time() - FIS_algorithm_start_time
        elif FIS_style == "maximal":
            hyperparam_str = f"maximal_interactions"
            sian_paths_prefix = f"{exp_folder}{datetimestr}_MAXK{MAX_K}_TRN_N{TARGET_TRN_N}_{hyperparam_str}_"
            saved_results_path = sian_paths_prefix + 'results.json'
            print(f"Processing hyperparameter combination {hyperparam_idx+1}/{len(hyperparam_combinations)} for sample size {TARGET_TRN_N}: {hyperparam_str}")
            print(sian_paths_prefix)

            FIS_algorithm_start_time = time.time()
            D0 = grouped_features_dict["D0"]
            FIS_interactions = []
            for k in range(1,MAX_K+1):
                FIS_interactions.extend(   list(combinations(list(range(D0)),k))   )
            FIS_algorithm_time_taken = time.time() - FIS_algorithm_start_time
        else:
            raise Exception(f"FIS_style={FIS_style} not recognized")
        print("FIS_algorithm_time_taken", FIS_algorithm_time_taken)
        print("FIS_interactions")
        print(FIS_interactions)

        data_seed = dataset_str.split("_")[-1] #NOTE: looks like bad practice
        data_seed = data_seed.replace("seed", "")
        data_seed = int(data_seed)

        model_init_seed = data_seed
        sian_training_args.model_config.model_init_seed = model_init_seed

        print(f"Data seed: {data_seed}, Model init seed: {model_init_seed}")

        print(10 * "=", f"Training SIAN-{MAX_K} with sample size {TARGET_TRN_N}", 10 * "=")

        sian_training_args.model_config.FIS_interactions = FIS_interactions
        sian_results = train_sian_final(dataset_obj, sian_training_args)
        trained_sian = sian_results["best_net"]
        val_tensor = sian_results["val_tensor"]
        print(f"SIAN Results: {sian_results}")

        def get_device_details(device, VERBOSE=True):
            torch_version = torch.__version__
            if VERBOSE:
                print(f"Using device: {device}")
            if device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(device)
                cuda_version = torch.version.cuda
                if VERBOSE:
                    print("GPU Name:", gpu_name)
                    print("CUDA Version:", cuda_version)
            else:
                gpu_name = "CPU"
                cuda_version = None
                if VERBOSE:
                    print("No GPU available.")
            return_dictionary = {
                "gpu_name": gpu_name,
                "cuda_version": cuda_version,
                "torch_version": torch_version,
            }
            return return_dictionary

        gpu_details = get_device_details(device)

        if dataset_str in ["Kaggle_blastchar_telco_customer_churn", "Kaggle_ishadss_eucalyptus_dataset"]:
            test_metric_value = round(float(final_test_mse["roc_auc"]), 5)
        else:
            test_metric_value = round(float(sian_results["final_test_mse"]["mse"]), 5)

        last_epoch = max(sian_results["epoch_data"].keys())
        data = {
            "exp_folder": exp_folder,
            "dataset_str": dataset_str,
            "batch_size": BS,
            "epochs": EP,
            "learning_rate": LR,
            "preproc_owner": preproc_owner,
            "FIS_style": FIS_style,
            "is_masked_mlp": is_masked_mlp,
            "is_masked_sian": is_masked_sian,
            "use_mnist_scaling": use_mnist_scaling,
            "model_init_seed": model_init_seed,
            "data_seed": data_seed,
            "MAX_K": MAX_K,
            "TARGET_TRN_N": int(TARGET_TRN_N),
            "number_of_rounds": number_of_rounds if FIS_style == "batchwise" else None,
            "inters_per_round": inters_per_round if FIS_style == "batchwise" else None,
            # "tau_thresholds": tau_thresholds,
            "tau_thresholds": tau_thresholds if (FIS_style in ["batchwise","layerwise"]) else None,
            "theta_thresholds": theta_thresholds if FIS_style == "layerwise" else None,
            "FIS_algorithm_time_taken": FIS_algorithm_time_taken,
            "FIS_interactions": FIS_interactions,

            "MLP_total_training_time": round(mlp_results["total_training_time"], 2),
            "mlp_results.step_data" : mlp_results["step_data"],
            "mlp_results.epoch_data" : mlp_results["epoch_data"],
            "sian_results.step_data" : sian_results["step_data"],
            "sian_results.epoch_data" : sian_results["epoch_data"],

            "SIAN_total_training_time": round(sian_results["total_training_time"], 2),
            "train_mse": round(float(sian_results["epoch_data"][last_epoch]["trn_metric"]), 5),
            "val_mse": round(float(sian_results["epoch_data"][last_epoch]["val_metric"]), 5),
            "final_test_mse.mse": round(float(sian_results["final_test_mse"]["mse"]), 5),
            "GPU_name": gpu_details["gpu_name"]
        }
        print(data)
        with open(saved_results_path, "w") as f:
            json.dump(data, f, indent=4)

        print("JSON results saved successfully")