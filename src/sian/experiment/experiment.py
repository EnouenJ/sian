

import os
import copy
import time
import pickle
import sys
from abc import ABC, abstractmethod

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from sian.utils import gettimestamp, get_device_details
from sian.data import Final_TabularDataset
from sian.models import TrainingArgs
from sian.fis import layerwise_FIS_Hyperparameters, batchwise_FIS_Hyperparameters, maximal_FIS_Hyperparameters
from sian.interpret import unmasked_FID_Hyperparameters, masked_FID_Hyperparameters, fake_FID_Hyperparameters
###from sian import initalize_the_explainer
###from sian import train_mlp_final, do_the_fis_final, train_sian_final
from sian.interpret import plot_all_GAM_functions
# from .helpers import return_score_type_name, prepare_for_FIS, prepare_for_FIS_v2, prepare_for_FIS_v2 #jAM TODO

# from sian.models import SmoothOrMnist_UnmaskedOrMasked_MLP #TODO: circular import

import json
import argparse



from .helpers import initalize_the_explainer
from .helpers import train_mlp_final, do_the_fis_final, train_sian_final

from sian.utils import convert_dict_keys_to_int






def return_score_type_name(is_masked_mlp, explainer_score_type):
    print('Inside function return_score_type_name is_masked_mlp', is_masked_mlp)
    if is_masked_mlp:
        if explainer_score_type=="arch":
            # score_type_name = "new_arch_inter_sobol_score"
            score_type_name = "new_arch_inter_var_score"
        elif explainer_score_type=="inc":
            # score_type_name = "inc_inter_sobol_score"
            score_type_name = "inc_inter_var_score"
        elif explainer_score_type=="rem":
            # score_type_name = "rem_inter_sobol_score"
            score_type_name = "rem_inter_var_score"
        else:
            raise NotImplementedError(f'explainer_score_type={explainer_score_type} not in [\'arch\', \'inc\', \'rem\']')
    else:
        if explainer_score_type=="arch":
            score_type_name = "old_arch_inter_score"
        elif explainer_score_type=="inc":
            score_type_name = "inc_inter_score"
        elif explainer_score_type=="rem":
            score_type_name = "rem_inter_score"
        else:
            raise NotImplementedError(f'explainer_score_type={explainer_score_type} not in [\'arch\', \'inc\', \'rem\']')

    return score_type_name


# def prepare_for_FIS(FIS_style, dataset_obj, mlp_training_args):
# def prepare_for_FIS(are_we_traning_the_mlp, dataset_obj, mlp_training_args):
def prepare_for_FIS(are_we_traning_the_mlp, dataset_obj, mlp_training_args, explainer_score_type):
    output_type = dataset_obj.get_task_type()
    grouped_features_dict = dataset_obj.get_grouped_feature_dict()
    is_masked_mlp = mlp_training_args.model_config.is_masked
    score_type_name = return_score_type_name(is_masked_mlp, explainer_score_type)

    if are_we_traning_the_mlp:
        print(10 * "=", f"Training MLP", 10 * "=")
        mlp_results = train_mlp_final(dataset_obj, mlp_training_args)
        trained_mlp = mlp_results["best_net"]
        val_tensor = mlp_results["val_tensor"]

        if is_masked_mlp:
            fid_masking_style = "masking_based"
            print("FID masking style", fid_masking_style)
            if True: #TODO: need to add TASK_TYPE = REGRESSION 
                inc_rem_pel_list = ['inc_inter_sobol_score', 'rem_inter_sobol_score', 'new_arch_inter_sobol_score']
                inc_rem_pel_list = ['inc_inter_var_score', 'rem_inter_var_score', 'new_arch_inter_var_score']
            else:
                inc_rem_pel_list = ['inc_inter_KL_sobol_score', 'rem_inter_KL_sobol_score', 'new_arch_inter_KL_sobol_score']
                inc_rem_pel_list = ['inc_inter_var_score', 'rem_inter_var_score', 'new_arch_inter_var_score']
            fis_valX = val_tensor
            my_FID_hypers = masked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list,
                                                    grouped_features_dict)
        else:
            fid_masking_style = "triangle_marginal"
            print("FID masking style", fid_masking_style)
            inc_rem_pel_list = ['inc_inter_score', 'rem_inter_score', 'old_arch_inter_score']
            fis_valX = val_tensor.detach().cpu().numpy()
            my_FID_hypers = unmasked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list, mlp_training_args.device, grouped_features_dict)
            
        jam_arch = initalize_the_explainer(trained_mlp, my_FID_hypers)
    else:
        mlp_training_args.number_of_epochs = 0
        mlp_training_args.number_of_steps = None
        mlp_results = train_mlp_final(dataset_obj, mlp_training_args)
        
        my_FID_hypers = fake_FID_Hyperparameters(grouped_features_dict)
        trained_mlp = None
        jam_arch = initalize_the_explainer(trained_mlp, my_FID_hypers) #fake jam archipelago
        fis_valX = None
    
    return jam_arch, fis_valX, mlp_results



def prepare_for_FIS_v2(trained_mlp, is_masked_mlp, dataset_obj, explainer_score_type):
    are_we_traning_the_mlp = True
    output_type = dataset_obj.get_task_type()
    grouped_features_dict = dataset_obj.get_grouped_feature_dict()
    score_type_name = return_score_type_name(is_masked_mlp, explainer_score_type)

    if are_we_traning_the_mlp:
        #val_tensor = mlp_results["val_tensor"]

        if is_masked_mlp:
            fid_masking_style = "masking_based"
            print("FID masking style", fid_masking_style)
            if True: #TODO: need to add TASK_TYPE = REGRESSION 
                inc_rem_pel_list = ['inc_inter_sobol_score', 'rem_inter_sobol_score', 'new_arch_inter_sobol_score']
                inc_rem_pel_list = ['inc_inter_var_score', 'rem_inter_var_score', 'new_arch_inter_var_score']
            else:
                inc_rem_pel_list = ['inc_inter_KL_sobol_score', 'rem_inter_KL_sobol_score', 'new_arch_inter_KL_sobol_score']
                inc_rem_pel_list = ['inc_inter_var_score', 'rem_inter_var_score', 'new_arch_inter_var_score']
            # fis_valX = val_tensor
            my_FID_hypers = masked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list, grouped_features_dict)
        else:
            fid_masking_style = "triangle_marginal"
            print("FID masking style", fid_masking_style)
            inc_rem_pel_list = ['inc_inter_score', 'rem_inter_score', 'old_arch_inter_score']
            # fis_valX = val_tensor.detach().cpu().numpy()
            my_FID_hypers = unmasked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list, trained_mlp.device, grouped_features_dict)
            
        jam_arch = initalize_the_explainer(trained_mlp, my_FID_hypers)
    else:
        my_FID_hypers = fake_FID_Hyperparameters(grouped_features_dict)
        trained_mlp = None
        jam_arch = initalize_the_explainer(trained_mlp, my_FID_hypers) #fake jam archipelago
        fis_valX = None
    
    return jam_arch




BATCHWISE_PLOTTING = True
# BATCHWISE_PLOTTING = False #TODO -- pull out somewhere




#TODO: rename
#TODO: pull all of these out into the experiment choices to sweep over?
# def xxxxxxxxxxx(FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX,                  exp_folder, datetimestr, TARGET_TRN_N):
# def xxxxxxxxxxx(FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round,                  exp_folder, datetimestr, TARGET_TRN_N):
def really_do_the_fis(FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round, explainer_score_type,                 exp_folder, datetimestr, TARGET_TRN_N):
    # explainer_score_type = hyperparam_combo["explainer_score_type"] #TODO @ rahil, this version would be fine but right now it breaks the other versions just to generate the path prefix
    if FIS_style in ["layerwise", "batchwise"]:
        if FIS_style == "layerwise":
            tau_thresholds = {int(k): v for k, v in hyperparam_combo["tau_thresholds"].items()}
            theta_thresholds = {int(k): v for k, v in hyperparam_combo["theta_thresholds"].items()}
            hyperparam_str = f"tau{'_'.join(str(tau_thresholds[t]) for t in tau_thresholds)}_theta{'_'.join(str(theta_thresholds[t]) for t in theta_thresholds)}"
            my_FIS_hypers = layerwise_FIS_Hyperparameters(
                MAX_K, tau_thresholds, theta_thresholds, None, theta_percentile_mode=True
            )
        elif FIS_style == "batchwise":
            tau_thresholds = hyperparam_combo["tau_thresholds"]
            tau_thresholds = convert_dict_keys_to_int(  hyperparam_combo["tau_thresholds"]  )
            if number_of_rounds is None or inters_per_round is None:
                raise ValueError("number_of_rounds and inters_per_round must be specified for batchwise FIS")
            hyperparam_str = f"tau{'_'.join(str(tau_thresholds[t]) for t in tau_thresholds)}_rounds{number_of_rounds}_inters_per_round{inters_per_round}"
            my_FIS_hypers = batchwise_FIS_Hyperparameters(
                MAX_K, tau_thresholds, number_of_rounds=number_of_rounds, interactions_per_round=inters_per_round,
                explainer=None, tuples_initialization=None, pick_underlings=False, fill_underlings=False, PLOTTING=BATCHWISE_PLOTTING)

        sian_paths_prefix = f"{exp_folder}{datetimestr}_EST{explainer_score_type}_K{MAX_K}_N{TARGET_TRN_N}_{hyperparam_str}"
        saved_results_path = sian_paths_prefix + 'results.json'
        print(sian_paths_prefix)

        my_FIS_hypers.add_the_explainer(jam_arch)

        FIS_algorithm_start_time = time.time()
        FIS_interactions, FIS_other_stuff = do_the_fis_final(my_FIS_hypers, fis_valX, AGG_K=100)
        FIS_algorithm_time_taken = time.time() - FIS_algorithm_start_time
    elif FIS_style == "maximal":
        hyperparam_str = f"maximal_interactions"
        sian_paths_prefix = f"{exp_folder}{datetimestr}_MAXK{MAX_K}_TRN_N{TARGET_TRN_N}_{hyperparam_str}_"
        saved_results_path = sian_paths_prefix + 'results.json'
        print(sian_paths_prefix)

        my_FIS_hypers = maximal_FIS_Hyperparameters(MAX_K)
        my_FIS_hypers.add_the_explainer(jam_arch)

        FIS_algorithm_start_time = time.time()
        FIS_interactions, FIS_other_stuff = do_the_fis_final(my_FIS_hypers, fis_valX, AGG_K=100)
        FIS_algorithm_time_taken = time.time() - FIS_algorithm_start_time
    else:
        raise Exception(f"FIS_style={FIS_style} not recognized")
    return FIS_interactions, FIS_algorithm_time_taken, FIS_other_stuff, saved_results_path








class Experiment(ABC):
    def __init__(self):
        self.MLP_network_hidden_sizes = [-1, 256, 128, 64, -1]
        self.SIAN_network_hidden_sizes = [-1, 256, 128, 64, -1]
        self.SIAN_network_hidden_small_sizes = [-1, 32, 24, 16, -1]
        
        self.default_batch_size = 32
        self.default_learning_rate = 5.0e-3
        self.default_epochs_steps = (None, 1024*128)


        self.default_tau_values = {
            1: 1.0,
            2: 0.5,
            3: 0.33
        }

        self.default_theta_perc_values = {
            1: 0.5,
            2: 0.5,
            3: 0.5
        }

        self.MLP_saving_settings_results_to_save = ['best_net','final_net']
        # self.MLP_saving_settings_results_to_save = []
        self.SIAN_saving_settings_results_to_save = ['best_net','final_net']
        # self.SIAN_saving_settings_results_to_save = [] 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def run(self, full_experiment_details):
        pass

    def get_basic_training_details(self, full_experiment_details):
        if "BS" in full_experiment_details:
            BS = full_experiment_details["BS"]
        else:
            BS = self.default_batch_size

        if "LR" in full_experiment_details:
            LR = full_experiment_details["LR"]
        else:
            LR = self.default_learning_rate

        if "EP" in full_experiment_details:
            EP = full_experiment_details["EP"]
            STEP = None
        elif "STEP" in full_experiment_details:
            EP = None
            STEP = full_experiment_details["STEP"]
        else:
            EP,STEP = self.default_epochs_steps
            # default value
            EP = None
            STEP = 1024*128

        return BS, LR, EP, STEP

    def collect_the_seeds(self, full_experiment_details):
        if "master_seed" in full_experiment_details:
            master_seed = full_experiment_details["master_seed"]
        else:
            # master_seed = None
            master_seed = 0 # NOTE: seed everything while prepping experiments
        if master_seed is None:
            np.random.seed(None)
            master_seed = int(np.random.randint(1))

        if "trnval_shuffle_seed" in full_experiment_details:
            trnval_shuffle_seed = full_experiment_details["trnval_shuffle_seed"]
        else:
            trnval_shuffle_seed = master_seed
        if "batch_shuffling_seed" in full_experiment_details:
            batch_shuffling_seed = full_experiment_details["batch_shuffling_seed"]
        else:
            batch_shuffling_seed = master_seed
        if "model_init_seed" in full_experiment_details:
            model_init_seed = full_experiment_details["model_init_seed"]
        else:
            model_init_seed = master_seed

        return master_seed, trnval_shuffle_seed, batch_shuffling_seed, model_init_seed


    def get_training_configuration(self, full_experiment_details, exp_folder, type_of_model):
        BS, LR, EP, STEP = self.get_basic_training_details(full_experiment_details)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        
        training_args = TrainingArgs(BS, EP, LR, device, STEP)

        model_init_seed = full_experiment_details["model_init_seed"]
        training_args.model_config.model_init_seed = model_init_seed

        if "lambda1" in full_experiment_details:
            training_args.lambda1 = full_experiment_details["lambda1"]
        if "lambda2" in full_experiment_details:
            training_args.lambda2 = full_experiment_details["lambda2"]

        if type_of_model == "MLP":
            is_masked_mlp = full_experiment_details["is_masked_mlp"] 
            use_mnist_scaling = full_experiment_details["use_mnist_scaling"]

            training_args.model_config.net_name = "MLP"
            training_args.model_config.sizes = self.MLP_network_hidden_sizes
            training_args.model_config.is_masked = is_masked_mlp
            training_args.model_config.use_mnist_scaling = use_mnist_scaling
            training_args.saving_settings.exp_folder = exp_folder + f"model_seed{model_init_seed}_"
            training_args.saving_settings.results_to_save = self.MLP_saving_settings_results_to_save

            if is_masked_mlp: #TODO - 09/21/25 @ 12:00am
                print('STEP',STEP)
                halfstep = STEP // 2
                halfpow = int(np.log(halfstep)/np.log(2))
                masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(100)))
                masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(halfpow+1))) + list(halfstep + np.power(2,np.arange(halfpow+1)))
                
                quarterstep = STEP // 4
                print('quarterstep',quarterstep)
                quarterpow = int(np.log(quarterstep)/np.log(2))
                masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(quarterpow+1))) + list(quarterstep + np.power(2,np.arange(halfpow+2))) + [STEP]
                print('masked_steps_to_evaluate',masked_steps_to_evaluate)
                training_args.return_settings.steps_to_evaluate = masked_steps_to_evaluate

        elif type_of_model == "SIAN":
            is_masked_sian = full_experiment_details["is_masked_sian"] 
            use_mnist_scaling = full_experiment_details["use_mnist_scaling"]

            training_args.model_config.net_name = "SIAN-K"
            training_args.model_config.sizes = self.SIAN_network_hidden_sizes
            training_args.model_config.small_sizes = self.SIAN_network_hidden_small_sizes
            training_args.model_config.is_masked = is_masked_sian
            training_args.model_config.use_mnist_scaling = use_mnist_scaling
            training_args.saving_settings.exp_folder = exp_folder + f"model_seed{model_init_seed}_"
            training_args.saving_settings.results_to_save = self.SIAN_saving_settings_results_to_save
            if is_masked_sian: 
                halfstep = STEP // 2
                halfpow = int(np.log(halfstep)/np.log(2))
                masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(halfpow+1))) + list(halfstep + np.power(2,np.arange(halfpow+1)))
                print('masked_steps_to_evaluate',masked_steps_to_evaluate)
                training_args.return_settings.steps_to_evaluate = masked_steps_to_evaluate
            if "compute_shapeloss" in full_experiment_details:
                training_args.compute_shapeloss = full_experiment_details['compute_shapeloss']

        return training_args


    def get_device_details(self):
        return get_device_details(self.device)

    def add_base_experiment_data(self, experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details):
        experiment_data.update({
            "dataset_str": dataset_str,
            "preproc_owner": preproc_owner,
            "model_init_seed": model_init_seed,
            "data_seed": data_seed,
            "GPU_name": gpu_details["gpu_name"]
        })

    def add_training_params(self, experiment_data, training_args):
        experiment_data.update({
            "batch_size": training_args.batch_size,
            "epochs": training_args.number_of_epochs,
            "steps": training_args.number_of_steps,
            "learning_rate": training_args.learning_rate,
        })

    def add_model_config_params(self, experiment_data, model_config, prefix="mlp"):
        experiment_data.update({
            f"is_masked_{prefix}": model_config.is_masked,
            "use_mnist_scaling": model_config.use_mnist_scaling,
        })

    def add_fis_params(self, experiment_data, FIS_style, MAX_K, number_of_rounds, inters_per_round, tau_thresholds, theta_thresholds, explainer_score_type):
        experiment_data.update({
            "explainer_score_type": explainer_score_type,
            "FIS_style": FIS_style,
            "MAX_K": MAX_K,
            "number_of_rounds": number_of_rounds if FIS_style == "batchwise" else None,
            "inters_per_round": inters_per_round if FIS_style == "batchwise" else None,
            "tau_thresholds": tau_thresholds if FIS_style in ["batchwise", "layerwise"] else None,
            "theta_thresholds": theta_thresholds if FIS_style == "layerwise" else None,
        })

    def save_experiment_data(self, exp_folder, TARGET_TRN_N, experiment_data, suffix="results"):
        datetimestr = gettimestamp()
        saved_results_path = f"{exp_folder}{datetimestr}_N{TARGET_TRN_N}_{suffix}.json"
        with open(saved_results_path, "w") as f:
            json.dump(experiment_data, f, indent=4)
        print("JSON results saved successfully")
        return saved_results_path

    def setup_experiment_folder(self, dataset_str): #TODO: still fix this @rahil we talked about putting the 'single experiment' strings here
        cwd = os.getcwd()
        results_path = cwd + "/" + "results/"
        exp_datetimestr = gettimestamp()
        if dataset_str.startswith("SYNTH_"):
            exp_folder = results_path + exp_datetimestr + '_' + "synthetic_sampleComplexity_sweep/"
        else:
            exp_folder = results_path + exp_datetimestr + '_' + "demo" + '_simple_testing/'
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        return exp_folder





