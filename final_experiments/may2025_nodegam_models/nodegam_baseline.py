#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from sian.fis import layerwise_FIS_Hyperparameters, batchwise_FIS_Hyperparameters
from sian.interpret import unmasked_FID_Hyperparameters, masked_FID_Hyperparameters
from sian import initalize_the_explainer

from sian import train_mlp_final, do_the_fis_final, train_sian_final #steps 1, 2, and 3
from sian.interpret import plot_all_GAM_functions #step 4
from sian.models.nodegam_models import train_node_ga2m, test_node_ga2m

import nodegam
from nodegam.sklearn import NodeGAMRegressor, NodeGAMClassifier

import sklearn
from sklearn.metrics import (
    mean_squared_error, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score
)

import argparse

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


parser = argparse.ArgumentParser(description="Node-GAM Baseline Runner")

parser.add_argument("--dataset_str", type=str, default="UCI_572_taiwanese_bankruptcy_dataset",
                    help="Dataset identifier string")
parser.add_argument("--preproc_owner", type=str, default="FIS2025",
                    help="Preprocessing owner name")
parser.add_argument("--IS_GA2M", type=int, default=0, choices=[0, 1],
                    help="Use GA2M model if set to 1, else use NAM")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility")

args = parser.parse_args()

dataset_str = args.dataset_str
preproc_owner = args.preproc_owner
IS_GA2M = args.IS_GA2M
seed = args.seed


# In[2]:

data_base_path = "../../data/"
load_dataset_path = data_base_path
save_dataset_path = data_base_path+dataset_str+"/"

dataset_obj = Final_TabularDataset(dataset_str, preproc_owner=preproc_owner,
                       load_dataset_path=load_dataset_path, 
                       save_dataset_path=save_dataset_path)


# In[3]:

cwd = os.getcwd()
results_path = cwd + "/" + "results/"
exp_datetimestr = gettimestamp()

exp_folder = results_path+exp_datetimestr +'_'+ "nodegam_baseline_results/"
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)
print(exp_folder)


# In[5]:


train_node_ga2m(dataset_obj, exp_folder, IS_GA2M=IS_GA2M, seed=seed)
test_node_ga2m(dataset_obj, exp_folder, IS_GA2M=IS_GA2M, seed=seed)
