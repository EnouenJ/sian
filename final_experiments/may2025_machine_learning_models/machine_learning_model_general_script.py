#!/usr/bin/env python
# coding: utf-8

# In[58]:


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

from sian.utils import gettimestamp
from sian.data import Final_TabularDataset
from sian.models import TrainingArgs
from sian.models.machine_learning_models import train_and_get_ml_model_results, test_ml_model_results

from sian.fis import layerwise_FIS_Hyperparameters, batchwise_FIS_Hyperparameters
from sian.interpret import unmasked_FID_Hyperparameters, masked_FID_Hyperparameters
from sian import initalize_the_explainer

from sian import train_mlp_final, do_the_fis_final, train_sian_final #steps 1, 2, and 3
from sian.interpret import plot_all_GAM_functions #step 4

import optuna
import argparse

import warnings
warnings.filterwarnings("ignore")

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[59]:


parser = argparse.ArgumentParser(description="Select dataset, preprocessing owner and number of hyperparameter tuning trials.")
parser.add_argument('--model_to_train', type=str, required=True, help='Model to be trained (e.g., XGB, Random Forest, SVM)')
parser.add_argument('--dataset_str', type=str, required=True, help='Dataset string (e.g., UCI_31_tree_cover_type_dataset)')
parser.add_argument('--preproc_owner', type=str, required=True, help='Preprocessing owner (e.g., InstaSHAP2025)')
parser.add_argument('--use_optuna', type=str, required=True, help='Whether or not to use Optuna for hyperparameter tuning (or use the default hyperparameters instead)')
parser.add_argument('--optuna_n_trials', type=int, required=True, help='Number of hyperparameter tuning trails for Optuna (e.g., 100)')
parser.add_argument('--seed', type=int, default=0, required=True, help='Random seed for reproducibility (default: 0)')

args = parser.parse_args()


# In[114]:


BS = 32
# EP = 100
EP = 2
LR = 5e-3

# if True:
#     dataset_str = "UCI_275_bike_sharing_dataset"
#     preproc_owner = "SIAN2022"
# if True:
#     dataset_str = "UCI_186_wine_quality"
#     preproc_owner = "SIAN2022"
# if True:
#     dataset_str = "UCI_2_adults_dataset"
#     preproc_owner = "InstaSHAP2025"
# if True:
#     dataset_str = "UCI_31_tree_cover_type_dataset"
#     preproc_owner = "InstaSHAP2025"
# if True:
#     dataset_str = "otherSource_cal_housing"
#     preproc_owner = "SIAN2022"
# if False:
#     dataset_str = "UCI_203_yearpredictionMSD"
#     preproc_owner = "SIAN2022"
# if True:
#     dataset_str = "UCI_1_abalone_dataset"
#     preproc_owner="FIS2025"
# if True:
#     dataset_str = "Kaggle_blastchar_telco_customer_churn"
#     preproc_owner = "FIS2025"
# if True:
#     dataset_str = "UCI_332_online_news_popularity"
#     preproc_owner = "FIS2025"
# if True:
#     dataset_str = "Kaggle_mlgulb_credit_card_fraud_dataset"
#     preproc_owner = "FIS2025"
# if False:
#     dataset_str = "otherSource_Microsoft_search_queries"
#     preproc_owner = "FIS2025"



dataset_str = args.dataset_str
preproc_owner = args.preproc_owner
if preproc_owner=="None":
    preproc_owner = None
N_TRIALS = args.optuna_n_trials

SEED = args.seed

if args.use_optuna in ["true","True","TRUE"]:
    TUNING_ON = True
else:
    TUNING_ON = False
print('TUNING_ON',TUNING_ON)

data_base_path = "../../data/"
load_dataset_path = data_base_path
save_dataset_path = data_base_path+dataset_str+"/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_training_args = TrainingArgs(BS, EP, LR, device)
sian_training_args = TrainingArgs(BS, EP, LR, device)




is_masked_mlp = True;  is_masked_sian = True;
# is_masked_mlp = False; is_masked_sian = False;

FIS_style = 'batchwise'
# FIS_style = 'layerwise'    

MAX_K = None
MAX_K = 3
MAX_K = 2 #quicker for marginal which is hella slow


# In[115]:

cwd = os.getcwd()
results_path = cwd + "/" + "results/"
exp_datetimestr = gettimestamp()
###exp_folder = results_path+exp_datetimestr +'_'+ "demo" +'_simple_testing/'
exp_folder = results_path+exp_datetimestr +'_'+ "sklearn_baseline_results/"
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)
print(exp_folder)


# In[116]:


if True: #DEFAULT MODEL PARAMETRIZATION
    mlp_training_args.model_config.net_name = "MLP"
    mlp_training_args.model_config.sizes = [-1, 256, 128, 64, -1]
    mlp_training_args.model_config.is_masked = is_masked_mlp
    mlp_training_args.saving_settings.exp_folder = exp_folder
    mlp_training_args.return_settings.return_val_tensor = True #for step 2
    mlp_training_args.return_settings.return_final_net = True #for step 2
    
    sian_training_args.model_config.net_name = "SIAN-K"
    sian_training_args.model_config.sizes = [-1, 256, 128, 64, -1]
    sian_training_args.model_config.small_sizes = [-1, 32, 24, 16, -1]
    sian_training_args.model_config.is_masked = is_masked_sian
    sian_training_args.saving_settings.exp_folder = exp_folder
    sian_training_args.return_settings.return_val_tensor = True #for step 4
    sian_training_args.return_settings.return_final_net = True #for step 4


# In[117]:


dataset_obj = \
    Final_TabularDataset(dataset_str, preproc_owner=preproc_owner,
                       load_dataset_path=load_dataset_path, 
                       save_dataset_path=save_dataset_path)     


# In[118]:


D = dataset_obj.get_D()
readable_labels = dataset_obj.get_readable_labels()
print(readable_labels)


# @TODO: @Rahil - Uncomment lines 171 - 253 while running
# # SIAN Step 1: Train Masked MLP

# # In[119]:


# mlp_results = train_mlp_final(dataset_obj, mlp_training_args)
# trained_mlp = mlp_results["final_net"]
# val_tensor = mlp_results["val_tensor"]


# # # SIAN Step 2: Masked Archipelago FIS

# # ### setup FID hypers

# # In[120]:


###output_type = dataset_obj.get_task_type()
# grouped_features_dict = dataset_obj.get_grouped_feature_dict()
# if is_masked_mlp: 
#     fid_masking_style = "masking_based"
#     score_type_name = "new_arch_inter_sobol_score"
#     inc_rem_pel_list = ['inc_inter_sobol_score', 'rem_inter_sobol_score', 'new_arch_inter_sobol_score',] #NOTE: only for batchwise plots
#     fis_valX = val_tensor

#     my_FID_hypers = masked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list,
#                                                grouped_features_dict)
# else:    
#     fid_masking_style = "triangle_marginal"
#     score_type_name = "old_arch_inter_score"
#     inc_rem_pel_list = ['inc_inter_score', 'rem_inter_score', 'old_arch_inter_score',] #NOTE: only for batchwise plots
#     fis_valX = val_tensor.detach().cpu().numpy()
    
#     my_FID_hypers = unmasked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list,
#                                                device, grouped_features_dict)


# # ### setup FIS hypers

# # In[121]:


# if FIS_style=="batchwise":
#     max_number_of_rounds = 5
#     inters_per_round = 1
#     tau_tup=(1.0,0.5,0.33)
    
#     tau_thresholds = {}
#     for k in range(MAX_K): #NOTE: no good MAX_K = None support yet
#         tau_thresholds[k+1] = tau_tup[k]
    
#     my_FIS_hypers = batchwise_FIS_Hyperparameters(MAX_K, tau_thresholds, max_number_of_rounds, inters_per_round,
#                    # jam_arch, 
#                    None, 
#                    tuples_initialization=None,pick_underlings=False,fill_underlings=False,PLOTTING=True)

# elif FIS_style=="layerwise":

#     theta_percentile_mode=True
#     theta_tup=(0.8,0.4,0.2)
#     tau_tup=(1.0,0.5,0.33)
    
#     tau_thresholds, theta_thresholds = {}, {}
#     for k in range(MAX_K):
#         tau_thresholds[k+1] = tau_tup[k]
#         theta_thresholds[k+1] = theta_tup[k]

#     my_FIS_hypers = layerwise_FIS_Hyperparameters(MAX_K, tau_thresholds, theta_thresholds, 
#                    # jam_arch, 
#                    None, 
#                    theta_percentile_mode=theta_percentile_mode)
# else:
#     raise Exception(f"FIS_style={FIS_style} not recognized")


# # ### finalize the FID and FIS hypers

# # In[122]:


# jam_arch = initalize_the_explainer(trained_mlp, my_FID_hypers)
# my_FIS_hypers.add_the_explainer(jam_arch)


# ### run the actual FIS

# In[123]:


# FIS_algorithm_start_time = time.time()
# FIS_interactions = do_the_fis_final(my_FIS_hypers, fis_valX, AGG_K=100)
# FIS_algorithm_time_taken = time.time() - FIS_algorithm_start_time
# print("FIS_algorithm_time_taken",FIS_algorithm_time_taken)


# # SIAN Step 3: Train the InstaSHAP GAM

# In[124]:


# print("FIS_interactions")
# print(FIS_interactions)


# In[125]:


# sian_training_args.model_config.FIS_interactions = FIS_interactions
# sian_results = train_sian_final(dataset_obj, sian_training_args)
# trained_sian = sian_results["final_net"]
# val_tensor = sian_results["val_tensor"]


# # SIAN Step 4: Plotting Learned Shapes

# In[126]:


# full_readable_labels = dataset_obj.get_full_readable_labels()
# plot_all_GAM_functions(trained_sian.cpu(), val_tensor.detach().cpu().numpy(),     full_readable_labels)


# In[127]:


# pass #n_tokens


# # Training and Testing ML Models

# In[128]:


output_type = dataset_obj.get_task_type()
# trnval_shuffle_seed = 0  #TODO: needs to be added
# trnval_per = 0.70 #NOTE: default
# dataset_obj.shuffle_and_split_trnval(trnval_shuffle_seed=trnval_shuffle_seed,trnval_split_percentage=trnval_per)



output_type

print('TUNING_ON',TUNING_ON)

# In[130]:


# models = ["XGB"] # You can add "Random Forest", "SVM" "CatBoost", "LightGBM", and "EBM" as required
models = []
models.append(args.model_to_train)


# JAM: commenting this out because I think the argument solves this problem (potentially better until we figure out a better way to pass arguments?)

# best_models_default = {}
# best_params_dict_default = {}
# best_val_scores_default = {}

# for model_name in models:

#     model, name, params, val_score = train_and_get_ml_model_results(
#         model_name=model_name,
#         dataset_obj=dataset_obj,
#         task_type=output_type,
#         exp_folder=exp_folder,
#         use_optuna=False,
#         use_default=True

#     )

#     best_models_default[f"{model_name}_default"] = model
#     best_params_dict_default[f"{model_name}_default"] = params
#     best_val_scores_default[f"{model_name}_default"] = val_score

# for model_name in models:

#     test_ml_model_results(best_models_default[model_name + "_default"], 
#                           model_name, 
#                           dataset_obj, 
#                           task_type=output_type, 
#                           exp_folder=exp_folder,
#                           use_default=True)


best_models = {}
best_params_dict = {}
best_val_scores = {}

for model_name in models:
    model, name, params, val_score = train_and_get_ml_model_results(
        model_name=model_name,
        dataset_obj=dataset_obj,
        task_type=output_type,
        exp_folder=exp_folder,
        use_optuna=TUNING_ON,
        n_trials=N_TRIALS,
        seed=SEED,
        device=device
    )
    best_models[model_name] = model
    best_params_dict[model_name] = params
    best_val_scores[model_name] = val_score

print("\n===== Summary of Best Models =====")
for model_name in models:
    print(f"Best Model: {model_name}")
    print(f"Best Validation {'MSE' if output_type == 'regression' else 'ROC AUC'}: {best_val_scores[model_name]:.3f}")
    print(f"Best Hyperparameters: {best_params_dict[model_name]}")
    print()


# In[131]:

print('TUNING_ON',TUNING_ON)
for model_name in models:
    test_ml_model_results(best_models[model_name], 
                          "RF" if model_name == "Random Forest" else model_name,
                          dataset_obj, 
                          task_type=output_type, 
                          exp_folder=exp_folder,
                          use_default=(not TUNING_ON))


