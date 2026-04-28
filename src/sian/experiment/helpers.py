import os
import copy
import time
import pickle
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





from sian.utils import gettimestamp

from sian.models import evaluate_model_on_test_set
from sian.models import either_normal_or_masked___gradient_descent_training #04/12/2025 @ 1:30am

from sian.models import MLP, SIAN, MaskedMLP, InstaSHAPMasked_SIAN
from sian.fis import layerwise_feature_interaction_selection_algorithm
from sian.fis import batchwise_feature_interaction_selection_algorithm
from sian.fis import maximal_feature_interaction_selection_algorithm

from sian.interpret import JamArchipelago, JamMaskedArchipelago, FakeJamArchipelago
from sian.interpret.basic_wrappers import MixedModelWrapperTorch, Masked_MixedModelWrapperTorch

#04/17/2025
# from sian.models import mnist_SIAN
# from sian.models import Smooth_Or_MNIST_SIAN
#08/16/2025
# from sian.models import SmoothOrMnist_InstaSHAPMasked_SIAN
#09/08/2025
from sian.models import SmoothOrMnist_UnmaskedOrMasked_SIAN
#09/14/2025
from sian.models import SmoothOrMnist_UnmaskedOrMasked_MLP
#09/18/2025
from sian.models import SemiInflated_SmoothOrMnist_UnmaskedOrMasked_SIAN
from sian.models import Mnist_CnnModel




import sklearn
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)






def get_loss_type(task_type):
    if task_type=="regression":
        return "mse"
    elif task_type=="binary_classification":
        return "ce"
    elif task_type=="multiclass_classification":
        return "softmax"
    else:
        raise Exception()


def train_mlp_final(my_dataset, training_args):
    exp_folder = training_args.saving_settings.exp_folder
    datetimestr = gettimestamp()
    model_path = exp_folder+datetimestr+"_"
    net_name = training_args.model_config.net_name
    
    training_args.saving_settings.net_name = net_name
    training_args.saving_settings.results_save_prefix = model_path

    model_init_seed = training_args.model_config.model_init_seed
    if model_init_seed is None:
        model_init_seed = int(np.random.randint(1))
    torch.manual_seed(model_init_seed)

    is_masked_mlp = training_args.model_config.is_masked
    use_mnist_mlp = training_args.model_config.use_mnist_scaling
    D = my_dataset.get_D()
    C = my_dataset.get_C()
    task_type = my_dataset.get_task_type()
    loss_type = get_loss_type(task_type)
    training_args.task_type = task_type
    training_args.loss_type = loss_type

    sizes = training_args.model_config.sizes
    sizes[0] = D; sizes[-1] = C;

    print(f"For MLP Using SEED = {model_init_seed}")
    print('is_masked_mlp',is_masked_mlp)
    print('use_mnist_mlp',use_mnist_mlp)
    if is_masked_mlp:
        mlp_masking_mode = "default_masked"
    elif not is_masked_mlp:
        mlp_masking_mode = "default_unmasked"
    else:
        raise NotImplementedError(f"is_masked_mlp={is_masked_mlp}") 
    if use_mnist_mlp=="mnist":
        mlp_parametrization_mode = "mnist_parametrization"
    elif use_mnist_mlp=="smooth":
        mlp_parametrization_mode = "smooth_parametrization"
    elif use_mnist_mlp=="old":
        mlp_parametrization_mode = "old_mnist_parametrization"
    else:
        raise NotImplementedError(f"use_mnist_mlp={use_mnist_mlp}")
    print('  mlp_masking_mode',mlp_masking_mode)
    print('  mlp_parametrization_mode',mlp_parametrization_mode)

    mlp = SmoothOrMnist_UnmaskedOrMasked_MLP(sizes, masking_mode=mlp_masking_mode, parametrization_mode=mlp_parametrization_mode)
    mlp_results = either_normal_or_masked___gradient_descent_training(my_dataset, mlp, training_args)
    final_test_mse = evaluate_model_on_test_set(my_dataset, mlp, training_args, is_masked_model=is_masked_mlp)
    
    mlp_results["model_init_seed"] = model_init_seed
    return mlp_results


def train_sian_final(my_dataset, training_args):
    exp_folder = training_args.saving_settings.exp_folder
    datetimestr = gettimestamp()
    model_path = exp_folder+datetimestr+"_"
    net_name = training_args.model_config.net_name
    
    training_args.saving_settings.net_name = net_name
    training_args.saving_settings.results_save_prefix = model_path

    is_masked_sian = training_args.model_config.is_masked
    use_mnist_sian = training_args.model_config.use_mnist_scaling
    bias_configuration = training_args.model_config.bias_configuration
    D = my_dataset.get_D()
    C = my_dataset.get_C()
    task_type = my_dataset.get_task_type()
    loss_type = get_loss_type(task_type)
    training_args.task_type = task_type
    training_args.loss_type = loss_type

    training_args.model_config.feature_groups_dict = my_dataset.get_grouped_feature_dict()

    
    sizes = training_args.model_config.sizes
    small_sizes = training_args.model_config.small_sizes
    sizes[0] = D; sizes[-1] = C;
    small_sizes[0] = D; small_sizes[-1] = C;
    FIS_interactions = training_args.model_config.FIS_interactions
    feature_groups_dict = training_args.model_config.feature_groups_dict

    model_init_seed = training_args.model_config.model_init_seed
    if model_init_seed is None:
        model_init_seed = int(np.random.randint(1))
    torch.manual_seed(model_init_seed)

    print(f"For SIAN Using SEED = {model_init_seed}")
    print('is_masked_sian',is_masked_sian)
    print('use_mnist_sian',use_mnist_sian)
    if is_masked_sian:
        sian_masking_mode = "insta_masked"
    elif not is_masked_sian:
        sian_masking_mode = "default_unmasked"
    else:
        raise NotImplementedError(f"is_masked_sian={is_masked_sian}") 
    if use_mnist_sian=="mnist":
        sian_parametrization_mode = "mnist_parametrization"
    elif use_mnist_sian=="smooth":
        sian_parametrization_mode = "smooth_parametrization"
    else:
        raise NotImplementedError(f"use_mnist_sian={use_mnist_sian}")
    print('  sian_masking_mode',sian_masking_mode)
    print('  sian_parametrization_mode',sian_parametrization_mode)

    max_inflation_amount = training_args.model_config.max_inflation_amount
    sian = SemiInflated_SmoothOrMnist_UnmaskedOrMasked_SIAN(sizes, FIS_interactions, small_sizes=small_sizes, feature_groups_dict=feature_groups_dict, masking_mode=sian_masking_mode, parametrization_mode=sian_parametrization_mode, bias_configuration=bias_configuration, max_inflation_amount=max_inflation_amount)

    sian_results = either_normal_or_masked___gradient_descent_training(my_dataset, sian, training_args)    
    final_test_mse = evaluate_model_on_test_set(my_dataset, sian, training_args, is_masked_model=is_masked_sian)
    

    sian_results["model_init_seed"] = model_init_seed
    sian_results["final_test_mse"] = final_test_mse
    return sian_results






def train_cnn_final(my_dataset, training_args):
    exp_folder = training_args.saving_settings.exp_folder
    datetimestr = gettimestamp()
    model_path = exp_folder+datetimestr+"_"
    net_name = training_args.model_config.net_name
    
    training_args.saving_settings.net_name = net_name
    training_args.saving_settings.results_save_prefix = model_path

    model_init_seed = training_args.model_config.model_init_seed
    if model_init_seed is None:
        model_init_seed = int(np.random.randint(1))
    torch.manual_seed(model_init_seed)

    is_masked_mlp = training_args.model_config.is_masked
    use_mnist_mlp = training_args.model_config.use_mnist_scaling
    D = my_dataset.get_D()
    C = my_dataset.get_C()
    task_type = my_dataset.get_task_type()
    loss_type = get_loss_type(task_type)
    training_args.task_type = task_type
    training_args.loss_type = loss_type

    sizes = training_args.model_config.sizes
    sizes[0] = D; sizes[-1] = C;

    print(f"For MLP Using SEED = {model_init_seed}")
    print('is_masked_mlp',is_masked_mlp)
    print('use_mnist_mlp',use_mnist_mlp)
    if is_masked_mlp:
        mlp_masking_mode = "default_masked"
    elif not is_masked_mlp:
        mlp_masking_mode = "default_unmasked"
    else:
        raise NotImplementedError(f"is_masked_mlp={is_masked_mlp}") 
    if use_mnist_mlp=="mnist":
        mlp_parametrization_mode = "mnist_parametrization"
    elif use_mnist_mlp=="smooth":
        mlp_parametrization_mode = "smooth_parametrization"
    else:
        raise NotImplementedError(f"use_mnist_mlp={use_mnist_mlp}")
    print('  mlp_masking_mode',mlp_masking_mode)
    print('  mlp_parametrization_mode',mlp_parametrization_mode)

    mlp = Mnist_CnnModel()
    cnn_results = either_normal_or_masked___gradient_descent_training(my_dataset, mlp, training_args)
    final_test_mse = evaluate_model_on_test_set(my_dataset, mlp, training_args, is_masked_model=is_masked_mlp)


    cnn_results["model_init_seed"] = model_init_seed
    return cnn_results
























def do_the_fis_final(my_FIS_hypers, my_val, AGG_K):
    if my_FIS_hypers.FIS_type=="layerwise":
        FIS_interactions, other_results = layerwise_feature_interaction_selection_algorithm(my_FIS_hypers, my_val, AGG_K)
    elif my_FIS_hypers.FIS_type=="batchwise":
        FIS_interactions, other_results = batchwise_feature_interaction_selection_algorithm(my_FIS_hypers, my_val, AGG_K)
    elif my_FIS_hypers.FIS_type=="maximal":
        FIS_interactions, other_results = maximal_feature_interaction_selection_algorithm(my_FIS_hypers, my_val, AGG_K)
    else:
        raise Exception(f"unrecognised FIS_type={my_FIS_hypers.FIS_type}")
    return FIS_interactions, other_results


def initalize_the_explainer(trained_mlp, my_FID_hypers):
    if hasattr(my_FID_hypers,"is_maximal") and my_FID_hypers.is_maximal:
        jam_arch = FakeJamArchipelago(my_FID_hypers)
    else:
        if my_FID_hypers.is_masked_model:
            jam_arch = JamMaskedArchipelago(trained_mlp, my_FID_hypers)
        else:
            model_wrap_MLP = MixedModelWrapperTorch(trained_mlp, my_FID_hypers.device) 
            jam_arch = JamArchipelago(model_wrap_MLP, my_FID_hypers)
    return jam_arch



def return_score_type_name(is_masked_mlp, explainer_score_type):
    print('Inside function return_score_type_name is_masked_mlp', is_masked_mlp)
    if is_masked_mlp:
        if explainer_score_type=="arch":
            score_type_name = "new_arch_inter_sobol_score"
        elif explainer_score_type=="inc":
            score_type_name = "inc_inter_sobol_score"
        elif explainer_score_type=="rem":
            score_type_name = "rem_inter_sobol_score"
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
            inc_rem_pel_list = ['inc_inter_sobol_score', 'rem_inter_sobol_score', 'new_arch_inter_sobol_score']
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
            inc_rem_pel_list = ['inc_inter_sobol_score', 'rem_inter_sobol_score', 'new_arch_inter_sobol_score']
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







