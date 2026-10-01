import time
import os
import copy
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torch.optim as optim
from torch import Tensor

import matplotlib.pyplot as plt
import random

import sklearn
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# https://stackoverflow.com/questions/58608361/string-based-enum-in-python
class TaskType(str, Enum): #Python <=3.10 solution; >=3.11 has "StrEnum()"
    regression = "regression"
    binary = "binary_classification"
    multiclass = "multiclass_classification"
    def __str__(self) -> str:
        return self.value

LOGIT_CLAMPING = False #TODO: move inside

class TrainingArgs():
    pass

    def __init__(self, batch_size=32, number_of_epochs=300, learning_rate=5e-3, device=None, number_of_steps=None, trainval_reduction_percentage=None):

        self.batch_size = batch_size
        if number_of_epochs is not None and number_of_steps is not None:
            raise Exception(f"please define how long to train only using number_of_epochs={number_of_epochs} or number_of_steps={number_of_steps}")
        if number_of_epochs is not None:
            self.number_of_epochs = number_of_epochs
            self.number_of_steps = None
        else:
            self.number_of_epochs = None
            self.number_of_steps = number_of_steps
        self.learning_rate = learning_rate
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.trainval_reduction_percentage = trainval_reduction_percentage

        self.loss_type = None #"mse" or "ce" for regression or classificaiton
        self.opt_type = "Adagrad" #default

        self.trnval_shuffle_seed = 0
        self.batch_shuffling_seed = None
        self.trnval_split_percentage = 0.70
        self.lambda1 = 5e-5
        self.lambda2 = 0.0
        
        #self.lambda1shapeloss = 5e-5
        #self.lambda1shapeloss = 5e-4
        self.lambda1shapeloss = 5e-3
        self.compute_shapeloss = False


        

        self.normalize_XY = (False, False)
        self.normalize_XY = (True, True)

        self.EVALUATE_WITH_TRN_TENSOR = False
        self.EVALUATE_WITH_VAL_TENSOR = False

        self.USE_GRADIENT_CLIPPING = True
        self.LOGIT_CLAMPING = False

        self.early_stopping_patience = None
        # self.early_stopping_patience = 3

        self.PLOTTING_ALL_LOSSES_EVERY_EPOCH = False
        self.EVALUATE_EVERY_K_EPOCHS = None


        class VerbositySettings():
            def __init__(self):
                self.VERBOSE_TRAINING = True
        self.verbosity_settings = VerbositySettings()

        class SavingSettings():
            def __init__(self):
                self.results_save_path_prefix = "results/"
                self.things_to_save = {}

                self.saving_after_evaluation = False
                self.saving_after_evaluation_min_step = 4096 #at least 4096 steps before saving during training
                self.saved_results_path = None
        self.saving_settings = SavingSettings()

        class ReturnSettings():
            def __init__(self):
                self.return_val_tensor = True 
                self.return_val_output = True
                self.return_final_net = True
                self.return_best_net = True

                self.epochs_to_evaluate =  []
                self.steps_to_evaluate =  [0] + list(np.power(2,np.arange(100))) #exponential time scale

        self.return_settings = ReturnSettings()

        class ModelConfig():
            def __init__(self):
                self.model_init_seed = None
                self.is_masked = None
                self.use_mnist_scaling = None

                self.bias_configuration = {
                    "biases_on" : False,
                    "bias_on" : True,
                    "biases_scaling" : 1.0,
                    "bias_scaling" : 1.0,
                    "initialize_bias" : False,
                 }
                # self.bias_configuration = {
                #     "biases_on" : True,
                #     "bias_on" : True,
                #     "biases_scaling" : 100.0,
                #     "bias_scaling" : 100.0,
                #     "initialize_bias" : False,
                #  }
                # self.max_inflation_amount = 16
                self.max_inflation_amount = 32
                # self.max_inflation_amount = 64
        self.model_config = ModelConfig()

    def to_string(self):
        raise NotImplementedError("sorry, not implemented yet")


# def generate_mask_helper(x_batch, mask_distribution_type):
def generate_mask_helper(x_shape, x_device, mask_distribution_type):
    if mask_distribution_type == "unmasked":
        # s_batch = torch.ones_like(x_batch).long()
        s_batch = torch.ones(x_shape, device=x_device, dtype=torch.long)
    elif mask_distribution_type == "ShapKern":
        D = x_shape[1]
        interior_D = list(range(1,D))
        kernel_D = [1/i/(D-i) for i in interior_D]
        kernel_D = np.array(kernel_D) / np.sum(kernel_D)
        sizes = np.random.choice(interior_D, size=x_shape[0], p=kernel_D)
        nprng = np.random.default_rng()
        s_batch = (nprng.permuted( np.tile(np.arange(D)[None,:],(x_shape[0],1)), axis=1) >= sizes[:,None])
        s_batch=torch.LongTensor(s_batch)
    elif mask_distribution_type == "ShapProb":
        # mask_p_s = float(np.random.rand(x_shape[0]))  #TODO: 09/14/26 - WAS THIS BUGGED OR DIFFERENT FORMAT EXPECTED?
        mask_p_s = torch.Tensor( np.random.rand(x_shape[0]) )
        s_batch = (torch.rand(size=(x_shape)) >= mask_p_s[:,None]).long().to(x_device)
    elif mask_distribution_type == "ShapProb_fixedP":
        mask_p_s = float(np.random.rand(1))  
        s_batch = (torch.rand(size=(x_shape)) >= mask_p_s).long().to(x_device)
    elif mask_distribution_type in ["BanzKern", "BanzProb", "bernoulli"]:
        raise NotImplementedError(f"not yet mask_distribution={mask_distribution_type}") #TODO
    else:
        raise NotImplementedError(f"mask_distribution={mask_distribution_type}")
    return s_batch


#TODO: this is broken because of D0 v. D for onehot features
# def generate_mask_mixture(x_batch, masking_details):
def generate_mask_mixture(x_shape, x_device, masking_details):
    mask_prob = masking_details["mask_prob"]
    mask_mix_type = masking_details["mask_mix_type"]
    mask_distribution_type = masking_details["mask_distribution_type"]

    if mask_mix_type=="heterogeneous_mixing":
        if np.random.rand(1) > mask_prob:
            s_batch = generate_mask_helper(x_shape, x_device, "unmasked") 
        else:
            s_batch = generate_mask_helper(x_shape, x_device, mask_distribution_type)
        pass
    elif mask_mix_type=="homogeneous_mixing":
        s_batch = generate_mask_helper(x_shape, x_device, mask_distribution_type)
        unmasking_indices = (np.random.rand(x_shape[0]) >= mask_prob)
        s_batch[unmasking_indices] = 1
    else:
        raise NotImplementedError(f"mask_mix_type={mask_mix_type}")
    return s_batch


class MaskScheduler():
    def __init__(self, grouped_feat_dict):
        self.STEP = None
        self.step = None
        self.masking_start_step = None
        #self.mask_prob_scheduler = lambda s : float(np.clip(-1.0 + (s / self.STEP * 3), 0.0, 1.0)) #[off,grow,on]
        #self.mask_prob_scheduler = lambda s : float(np.clip(-2.0 + (s / self.STEP * 4), 0.0, 1.0)) #[off,off,grow,on]
        # self.mask_prob_scheduler = lambda mystep : float(np.clip(  ((mystep - self.masking_start_step) / self.STEP * 4), 0.0, 1.0)) #[off,off,grow,on]
        #09/22/25 @ 7:00pm
        # self.mask_prob_scheduler = lambda mystep : float(np.clip(  ((mystep - self.masking_start_step) / self.STEP * 4), 0.0, 0.5)) #[off,off,grow,on]
        # self.mask_prob_scheduler = lambda mystep : 0.5 #10/03/25 @ 11:15pm
        # self.mask_prob_scheduler = lambda mystep : 0.0 #10/04/25 @ 12:15am
        # self.mask_prob_scheduler = lambda mystep : 0.5 * float(mystep >= self.masking_start_step) #10/04/25 @ 12:45am
        # self.mask_prob_scheduler = lambda mystep : 1.0 * float(mystep >= self.masking_start_step) #10/04/25 @9:45pm
        # self.mask_prob_scheduler = lambda mystep : float(np.clip(  ((mystep - self.masking_start_step) / self.STEP * 4), 0.0, 0.5)) #10/06/25 @ 2:00pm -- back on
        # self.mask_prob_scheduler = lambda mystep : 0.5 * float(mystep >= self.STEP // 4) +  0.5 * float(mystep >= self.STEP // 2) #10/07/25 @ 1:00am -- keep trying
        # self.mask_prob_scheduler = lambda mystep : 1.0 * float(mystep >= self.masking_start_step) #10/07/25 @2:15am
        # self.mask_prob_scheduler = lambda mystep : float(np.clip(  ((mystep - self.masking_start_step) / self.STEP * 4), 0.0, 1.0)) #10/07/25 @ 11:50pm
        self.mask_prob_scheduler = lambda mystep : 0.5 #9/25/26 @ 7:45pm


        # self.mask_mix_type = 'heterogeneous_mixing'
        self.mask_mix_type = 'homogeneous_mixing'
        self.mask_distribution_type = "ShapProb_fixedP"

        self.grouped_feat_dict = grouped_feat_dict
        D0 = grouped_feat_dict["D0"]
        D  = grouped_feat_dict["D"]
        # self.degroup_tensor = torch.zeros((D0,D),dtype=torch.long) #RuntimeError: "addmm_cuda" not implemented for 'Long'
        self.degroup_tensor = torch.zeros((D0,D))
        for i in range(D0):
            for ii in grouped_feat_dict[i]:
                self.degroup_tensor[i,ii] = 1
        if False: #for showing reindexing matrix
            plt.imshow(self.degroup_tensor.detach().cpu().numpy(), cmap='magma');
            plt.colorbar()
            plt.show()


    def initialize_max_step(self, STEP):
        self.STEP = STEP
        # ##self.masking_start_step = STEP // 2
        # ##self.masking_start_step = 0 #mask from beginning
        # # self.masking_start_step = STEP // 2
        # # self.masking_start_step = STEP // 4 #10/04/25 @ 1:45pm
        # # self.masking_start_step = 0  #10/04/25 @ 3:45pm
        # # self.masking_start_step = STEP // 2
        # self.masking_start_step = STEP // 4
        # # self.masking_start_step = 0
        if True:  #09/25/26 @ 7:45pm
            self.masking_start_step = 0
        self.step = 0
        
    def get_min_masking_step(self):
        return self.masking_start_step 
        
    def get_masking_details(self, step):
        self.step = step
        masking_details = {
            "mask_prob" : self.mask_prob_scheduler(step),
            "mask_mix_type" : self.mask_mix_type,
            "mask_distribution_type" : self.mask_distribution_type,
        }
        return masking_details

    def get_mask(self, x_batch, step=None):
        if step is None:
            step = self.step
        masking_details = self.get_masking_details(step)

        N0 = x_batch.shape[0] # batch size (probably)
        D0 = self.grouped_feat_dict["D0"]
        new_x_shape = (N0, D0)
        # print('new_x_shape',new_x_shape)
        x_device = x_batch.device
        new_s_batch = generate_mask_mixture(new_x_shape, x_device, masking_details)
        s_batch = torch.matmul( new_s_batch.float(), self.degroup_tensor.to(x_device) ).long()
        return s_batch






TRACK_VAL_SAMPLING = True #TODO: pull inside, better helper functions so not does nothing inside





def either_normal_or_masked___gradient_descent_training(dataset_object, net, training_args):
        
    BS = training_args.batch_size
    EP = training_args.number_of_epochs
    LR = training_args.learning_rate
    STEP = training_args.number_of_steps
    
    lambda1 = training_args.lambda1
    lambda2 = training_args.lambda2

    lambda1shapeloss = training_args.lambda1shapeloss
    COMPUTE_SHAPELOSS = training_args.compute_shapeloss
    if COMPUTE_SHAPELOSS:
        net.gam.compute_shapeloss_while_training = True



    USE_GRADIENT_CLIPPING = training_args.USE_GRADIENT_CLIPPING
    # LOGIT_CLAMPING = training_args.LOGIT_CLAMPING
    early_stopping_patience = training_args.early_stopping_patience    
    VERBOSE_TRAINING = training_args.verbosity_settings.VERBOSE_TRAINING
    MASKING_MODE = training_args.model_config.is_masked
    TASK_TYPE = training_args.task_type 
    print("MASKING_MODE",MASKING_MODE)


    EVALUATE_WITH_TRN_TENSOR = training_args.EVALUATE_WITH_TRN_TENSOR
    EVALUATE_WITH_VAL_TENSOR = training_args.EVALUATE_WITH_VAL_TENSOR
    PLOTTING_ALL_LOSSES_EVERY_EPOCH = training_args.PLOTTING_ALL_LOSSES_EVERY_EPOCH
    # There is a bug in the plotting for non-regression datasets
    PLOTTING_ALL_LOSSES_EVERY_EPOCH = True #DEBUG
    PLOTTING_ALL_LOSSES_EVERY_STEP  = True #DEBUG
    PLOTTING_MASKING_LOSS_EVERY_STEP = True #DEBUG
    PLOTTING_ALL_LOSSES_EVERY_EPOCH = False #no more DEBUG
    PLOTTING_ALL_LOSSES_EVERY_STEP  = False #no more DEBUG
    PLOTTING_MASKING_LOSS_EVERY_STEP = False #no more DEBUG
    
    # SAVING_MASK_EVALUATION = True #DEBUG -- very memory intensive for saving (especially depending on size-based or individual-mask-based)
    SAVING_MASK_EVALUATION = False
    
    # DOUBLE_EARLY_STOPPING = True
    DOUBLE_EARLY_STOPPING = False

    

    DOUBLE_EARLY_STOPPING = (DOUBLE_EARLY_STOPPING and MASKING_MODE)
    PLOTTING_MASKING_LOSS_EVERY_STEP = (PLOTTING_MASKING_LOSS_EVERY_STEP and MASKING_MODE)
    if PLOTTING_ALL_LOSSES_EVERY_EPOCH or PLOTTING_ALL_LOSSES_EVERY_STEP:
        loss_list = []
        val_loss_list = []


    EVALUTE_TEST_DURING_RUN = True #08/17/2025 -- turning on for synthetic sweeps, but might be fine to keep


    steps_to_evaluate = training_args.return_settings.steps_to_evaluate
    epochs_to_evaluate = training_args.return_settings.epochs_to_evaluate

    results_save_prefix = training_args.saving_settings.results_save_prefix  
    results_to_save = training_args.saving_settings.results_to_save 
    net_name = training_args.saving_settings.net_name 
    saving_after_evaluation = training_args.saving_settings.saving_after_evaluation
    saving_after_evaluation_min_step = training_args.saving_settings.saving_after_evaluation_min_step
    

    device = training_args.device
    net = net.to(device)



    normalize_XY = training_args.normalize_XY
    dataset_object.shuffle_and_split_trnval(trnval_shuffle_seed=training_args.trnval_shuffle_seed, trnval_split_percentage=training_args.trnval_split_percentage, trnval_reduc_percentage=training_args.trainval_reduction_percentage)
    trn_loader, val_loader = dataset_object.pull_trnval_loaders(device, BS, training_args.batch_shuffling_seed, normalize_XY[0], normalize_XY[1])
    if EVALUTE_TEST_DURING_RUN:
        tst_loader = dataset_object.pull_tst_loaders(device, BS, training_args.batch_shuffling_seed, normalize_XY[0], normalize_XY[1])

    if False: #TODO: can no longer do this
        if hasattr(net,'gam'): 
            if training_args.loss_type=="softmax":
                net.gam.bias = torch.nn.Parameter(  torch.Tensor(np.log(np.mean(trnY,axis=0)))   )


    print(f"Expected steps per epoch: {len(trn_loader)} (Dataset size: {len(trn_loader.dataset)}, Batch size: {BS})")
    print(f"also expected val steps per epoch: {len(val_loader)} (Dataset size: {len(val_loader.dataset)}, Batch size: {BS})")

    # Adjust batch size if dataset is too small
    if len(trn_loader.dataset) < BS:
        BS = len(trn_loader.dataset)
        print(f"Adjusted batch size to {BS} to match training dataset size.")
        trn_loader, val_loader = dataset_object.pull_trnval_loaders(device, BS, training_args.batch_shuffling_seed, normalize_XY[0], normalize_XY[1])
        if EVALUTE_TEST_DURING_RUN:
            tst_loader = dataset_object.pull_tst_loaders(device, BS, training_args.batch_shuffling_seed, normalize_XY[0], normalize_XY[1])
        print(f"Updated steps per epoch: {len(trn_loader)} (Dataset size: {len(trn_loader.dataset)}, Batch size: {BS})")

    trn_tensor = None
    val_tensor = None
    if EVALUATE_WITH_TRN_TENSOR:	
        trn_tensor = dataset_object.pull_trn_tensor(device)
    if EVALUATE_WITH_VAL_TENSOR:	
        val_tensor = dataset_object.pull_val_tensor(device)
    assert not EVALUATE_WITH_TRN_TENSOR, "only trn_loader implemented right now"
    assert not EVALUATE_WITH_VAL_TENSOR, "only val_loader implemented right now"

    if training_args.opt_type=="Adagrad":
        opt = torch.optim.Adagrad(net.parameters(), lr=LR)
    else:
        raise Exception(f"training_args.opt_type={training_args.opt_type} not recognized")
    
    if EP is None:
        EP = (STEP // len(trn_loader)) + 2
    EVALUATE_EVERY_K_EPOCHS = training_args.EVALUATE_EVERY_K_EPOCHS
    if EVALUATE_EVERY_K_EPOCHS is None:
        EVALUATE_EVERY_K_EPOCHS = EP
    if STEP is None:
        STEP = EP * len(trn_loader)
    
    D = dataset_object.get_D()
    n_classes = dataset_object.get_C()
    all_trn_accs = np.zeros(EP)
    all_val_accs = np.zeros(EP)
    all_losses = np.zeros((EP,len(trn_loader),7))
    all_trn_losses = np.zeros((EP,len(trn_loader)))
    all_val_losses = np.zeros(EP)
    
    gradient_training_time_taken = 0.0
    metric_evaluation_time_taken = 0.0
    

    if MASKING_MODE:
        masking_phase = False
        mask_scheduler = None   #TODO: move outside
        if mask_scheduler is None:
            mask_scheduler = MaskScheduler(dataset_object.get_grouped_feature_dict())
        mask_scheduler2 = MaskScheduler(dataset_object.get_grouped_feature_dict())
        mask_scheduler2.mask_prob_scheduler = lambda mystep : 1.0 #full prob of random mask for evaluation (otherwise cant early stop)

        if STEP is not None:
            assert (STEP is not None), f'EP{EP},STEP{STEP}'
            mask_scheduler.initialize_max_step(STEP)
            masking_start_step = mask_scheduler.get_min_masking_step() #defaults to the correct STEP/2, future could do more complex mixings
        else:
            assert (EP==0), f'EP{EP},STEP{STEP}'

        all_subset_losses = torch.zeros((EP, D+1, 2)).to(device)
        subset_indexer = torch.ones(D).float().to(device) #NOTE: "addmv_impl_cuda" not implemented for 'Long'


        FULL_SUBSET_INDEXING_TRACKING = False
        # FULL_SUBSET_INDEXING_TRACKING = True
        if FULL_SUBSET_INDEXING_TRACKING: #getting freaky with full indexing on 2^d
            # all_subset_losses = torch.zeros((1, 2**D, 2)).to(device)
            # subset_indexer = (2**torch.arange(D).float()).to(device) #NOTE: "addmv_impl_cuda" not implemented for 'Long'
            # print('subset_indexer',subset_indexer) #actually this even breaks down for 2^25 because of the float accuracy issues
            if True:
                grouped_feat_dict = dataset_object.get_grouped_feature_dict()
                D0 = grouped_feat_dict['D0']
                all_subset_losses = torch.zeros((1, 2**D0, 2)).to(device)
                subset_indexer = torch.zeros(D).float().to(device)
                cum_ii = 0
                for i in range(D0):
                    subset_indexer[cum_ii] = 2**i
                    cum_ii += len(grouped_feat_dict[i])
                print('subset_indexer',subset_indexer)
        else:
            grouped_feat_dict = dataset_object.get_grouped_feature_dict()
    
    best_val_score = -float('inf')
    best_net = None
    best_step = 0


    if early_stopping_patience is None:
        # early_stopping_patience = EP
        early_stopping_patience = float('inf')

    epochs_without_improvement = 0
    
    trn_metric = 0.0 #enables passing "EP = 0"
    val_metric = 0.0 #enables passing "EP = 0"

    PRINT_LOSS_PER_STEP = True
    PRINT_LOSS_PER_STEP = False #09/20/25 @ 2:55pm -- too verbose jupyter memory
    step = 0
    step_data  = {}
    epoch_data = {}



    def evaluate_during_training(data_loader, mask_sampler=None): #NOTE: be aware that "net" and "masking_mode" are defined implicitly here 
        if TASK_TYPE == TaskType.regression:
            if mask_sampler is None:
                __metric = evaluate_during_training_from_dataloader_REG(net, data_loader, MASKING_MODE, mask_sampler=mask_sampler)
                # print('__metric',__metric)
            if mask_sampler is not None:
                eval_all_subset_losses = evaluate_during_training_from_dataloader_REG(net, data_loader, MASKING_MODE, mask_sampler=mask_sampler, grouped_feat_dict=grouped_feat_dict)
                # print('eval_all_subset_losses',eval_all_subset_losses)
                __metric = [float(x) for x in ((eval_all_subset_losses[:,0]) / (eval_all_subset_losses[:,1])).cpu().numpy() ]
        elif TASK_TYPE == TaskType.binary:
            __metric = evaluate_during_training_from_dataloader_BINARY_auc(net, data_loader, MASKING_MODE, mask_sampler=mask_sampler)
        elif TASK_TYPE == TaskType.multiclass:
            C = dataset_object.get_C()
            __metric = evaluate_during_training_from_dataloader_MULTI_auc(net, data_loader, MASKING_MODE, n_classes=C, mask_sampler=mask_sampler)
        else:
            raise Exception("not implemented task")
            __metric = 0.0
        return __metric
    
    def get_evaluation(): #NOTE: be aware that "net" and "masking_mode" are defined implicitly here 
        net.eval() 
        trn_metric = evaluate_during_training(trn_loader)
        val_metric = evaluate_during_training(val_loader)
        evaluation = {"trn_metric":trn_metric, "val_metric":val_metric}
        if EVALUTE_TEST_DURING_RUN:
            tst_metric = evaluate_during_training(tst_loader)
            evaluation["tst_metric"] = tst_metric
        net.train()
        return evaluation
    
    def get_mask_evaluation(mask_sampler): #NOTE: be aware that "net" and "masking_mode" are defined implicitly here 
        net.eval() 
        trn_metric = evaluate_during_training(trn_loader, mask_sampler)
        val_metric = evaluate_during_training(val_loader, mask_sampler)
        evaluation = {"trn_metric":trn_metric, "val_metric":val_metric}
        if EVALUTE_TEST_DURING_RUN:
            tst_metric = evaluate_during_training(tst_loader, mask_sampler)
            evaluation["tst_metric"] = tst_metric
        net.train()
        return evaluation






    if 0 in steps_to_evaluate or 0 in epochs_to_evaluate:
        evaluation = get_evaluation()
        if PLOTTING_MASKING_LOSS_EVERY_STEP:
            # evaluation['masking_trn_mse'] = np.zeros(D+1) + float('inf')
            evaluation['masking_trn_mse'] = [float('inf') for d in range(D+1)]
            evaluation['masking_trn_mse2'] = [float('inf') for d in range(D+1)]
            if FULL_SUBSET_INDEXING_TRACKING:
                evaluation['masking_trn_mse'] = [float('inf') for d in range(2**D0)]
                evaluation['masking_trn_mse2'] = [float('inf') for d in range(2**D0)]

        if DOUBLE_EARLY_STOPPING:
            mask_evaluation = get_mask_evaluation(mask_scheduler2)
            if SAVING_MASK_EVALUATION:
                evaluation['mask_evaluation'] = mask_evaluation
        if 0 in steps_to_evaluate:
            step_data[0] = evaluation
        if 0 in epochs_to_evaluate:
            epoch_data[0] = evaluation


    full_training_start_time = time.time()
    for k in range(EP):
        if step >= STEP:
            k = EP-1
            break
        if VERBOSE_TRAINING:
            print('Epoch', k)
        epoch_start_time = time.time()

        net.train()
        for j, (x_batch, y_batch) in enumerate(trn_loader):
            # print('x_batch',x_batch.shape)
            # print('y_batch',y_batch.shape)
            # if j>0:
            #     break #09/20/25 @ 4:45pm -- THIS IS HOW I USED TO DO STEPS I THINK????
            if MASKING_MODE:
                if step < masking_start_step and masking_phase:
                    print(f'we early stopped the vanilla part step={step}')
                    continue
                # if step == masking_start_step:
                if step >= masking_start_step and (not masking_phase):
                    masking_phase = True #not actually useful right now
                    opt = torch.optim.Adagrad(net.parameters(), lr=LR) #10/04/25 @ 11:00pm
                    print('Adagrad()')
                    opt = torch.optim.Adagrad(net.parameters(), lr=10*LR) #10/06/25 @ 4:00am
                    # if best_net is not None:
                    if False: #10/07/25 @ 2:30am
                        net = copy.deepcopy(best_net) #09/22/25 @ 12:10am -- I had forgotten to put this back in 
                    best_val_score = -float('inf')
                    best_step = masking_start_step
                    print("Starting to mask ^_^")
            grad_start_time = time.time()
            if torch.any(torch.isnan(x_batch)) or torch.any(torch.isinf(x_batch)):
                print(f"Warning: NaNs or infinities in train batch {j}")
                continue

            if not MASKING_MODE:
                dnn_logits, gam_logits, shape_loss = net(x_batch)
                logits = dnn_logits + gam_logits
            else:
                s_batch = mask_scheduler.get_mask(x_batch, step)
                # print('s_batch mean',torch.mean(s_batch.float()))
                dnn_logits, gam_logits, shape_loss = net((x_batch, s_batch))
                logits = dnn_logits + gam_logits

            l1_reg = torch.zeros(1).to(device)
            l2_reg = torch.zeros(1).to(device)
            if lambda1 > 0 or lambda2 > 0:
                all_linear_params = net.collectParameters()
                l1_reg = lambda1 * torch.norm(all_linear_params, 1)
                l2_reg = lambda2 * torch.norm(all_linear_params, 2)


            if training_args.loss_type=="mse":
                mseloss_ = (y_batch.narrow(1, 0, 1) - logits.narrow(1, 0, 1)) ** 2
                mseloss = torch.mean(mseloss_)
            elif training_args.loss_type=="ce":
                logprobs = torch.nn.LogSigmoid()(logits)
                lognotprobs = torch.nn.LogSigmoid()(-logits)
                celoss_ = -y_batch*logprobs-(1-y_batch)*lognotprobs
                celoss = torch.mean(celoss_)
                mseloss = celoss # misnomer
            elif training_args.loss_type=="softmax":
                logprobs = torch.nn.LogSoftmax(dim=-1)(logits)
                smloss_ = -y_batch*logprobs
                smloss = torch.mean( torch.sum(smloss_,dim=-1))
                mseloss = smloss # misnomer
            else:
                raise NotImplementedError(f"loss_type={training_args.loss_type}")
            
            if PLOTTING_ALL_LOSSES_EVERY_EPOCH:
                loss_list.append(mseloss.item())

            # loss = mseloss + l1_reg
            loss = mseloss + l1_reg + l2_reg
            if COMPUTE_SHAPELOSS:
                actual_loss_shapeloss = lambda1shapeloss * torch.sum( shape_loss )
                loss += actual_loss_shapeloss
            if VERBOSE_TRAINING and PRINT_LOSS_PER_STEP:
                print(f"Epoch {k}, Step {j}, Loss: {loss.item():.4f}")
            loss.backward()

            if USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            opt.step()
            opt.zero_grad()
            step += 1

            all_trn_losses[k, j] = mseloss.item()
            all_losses[k, j, 0] = loss.item()
            all_losses[k, j, 1] = mseloss.item()
            all_losses[k, j, 2] = l1_reg.item()
            all_losses[k, j, 3] = l2_reg.item()

            
            if PLOTTING_MASKING_LOSS_EVERY_STEP and MASKING_MODE and training_args.loss_type == "mse": #TODO: look into costliness of this
                s_index = torch.matmul(s_batch.float(), subset_indexer).long()
                if not FULL_SUBSET_INDEXING_TRACKING:
                    all_subset_losses[k, :, 0].index_put_((s_index,), mseloss_[:, 0].detach(), accumulate=True)
                    all_subset_losses[k, :, 1].index_put_((s_index,), torch.ones_like(mseloss_.detach())[:, 0], accumulate=True)
                else:
                    all_subset_losses[0, :, 0].index_put_((s_index,), mseloss_[:, 0].detach(), accumulate=True)
                    all_subset_losses[0, :, 1].index_put_((s_index,), torch.ones_like(mseloss_.detach())[:, 0], accumulate=True)
                
            gradient_training_time_taken += (time.time() - grad_start_time)            

            eval_start_time = time.time()
            if step in steps_to_evaluate:
                evaluation = get_evaluation()
                print(step,'evaluation',evaluation) #TODO: remove, only for debugging
                if True: #09/13/2025 - TODO: need this?
                    evaluation["L1"] = float(l1_reg.item())
                    evaluation["L2"] = float(l2_reg.item())
                    if COMPUTE_SHAPELOSS:
                        evaluation["shapeloss"] = float(actual_loss_shapeloss.item())
                    partial_training_time = time.time() - full_training_start_time
                    evaluation["partial_training_time"] = partial_training_time
                if PLOTTING_MASKING_LOSS_EVERY_STEP: #09/21/25 -- also doing looking at masking
                    evaluation["masking_trn_mse"] = [float(x) for x in (torch.sum(all_subset_losses[:,:,0],dim=0) / torch.sum(all_subset_losses[:,:,1],dim=0)).cpu().numpy() ]

                    if True:
                        if not FULL_SUBSET_INDEXING_TRACKING:
                            evaluation["masking_trn_mse2"] = [float(x) for x in (all_subset_losses[k,:,0] / all_subset_losses[k,:,1]).cpu().numpy() ]
                        else:
                            evaluation["masking_trn_mse2"] = [float(x) for x in (all_subset_losses[0,:,0] / all_subset_losses[0,:,1]).cpu().numpy() ]

                # print(step,'evaluation',evaluation) #TODO: remove, only for debugging

                if DOUBLE_EARLY_STOPPING:
                    mask_evaluation = get_mask_evaluation(mask_scheduler2)
                    if SAVING_MASK_EVALUATION:
                        evaluation['mask_evaluation'] = mask_evaluation
                # print(step,'evaluation',evaluation) #TODO: remove, only for debugging
                step_data[step] = evaluation

                if True: #checking evaluation for best_net
                    val_metric = evaluation['val_metric']
                    # if DOUBLE_EARLY_STOPPING:
                    if DOUBLE_EARLY_STOPPING and masking_phase: # TODO TODO: check importance
                        val_metric = mask_evaluation['val_metric']
                        if TRACK_VAL_SAMPLING:
                            val_metric = np.mean(val_metric)
                    val_score = -val_metric if TASK_TYPE == TaskType.regression else val_metric

                    if val_score > best_val_score:
                        # print(f"Val score: {val_score}, Best val score: {best_val_score}")
                        best_val_score = val_score
                        best_step = step
                        best_net = copy.deepcopy(net) #NOTE: slight concern with keeping two model copies on the GPU
                        epochs_without_improvement = 0
                        print(f"Steps without improvement = {epochs_without_improvement}")
                    else:
                        epochs_without_improvement += 1
                        print(f"Steps without improvement = {epochs_without_improvement}")

                    if VERBOSE_TRAINING:
                        trn_metric = evaluation['val_metric']
                        metric_name = "MSE" if TASK_TYPE == TaskType.regression else "ROC-AUC"
                        print(f'{metric_name} for train and val: {trn_metric:.4f}, {val_metric:.4f}')

                        
                    if PLOTTING_ALL_LOSSES_EVERY_STEP:
                        plt.plot(1+np.arange(len(loss_list)),loss_list)
                        if True:
                            substeps = list(step_data.keys())
                            trn_steps = [step_data[substep]['trn_metric'] for substep in substeps]
                            val_steps = [step_data[substep]['val_metric'] for substep in substeps]
                            plt.plot(1+np.array(substeps),val_steps)
                            plt.plot(1+np.array(substeps),trn_steps)
                        plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,2].reshape(-1)[:len(loss_list)])
                        plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,3].reshape(-1)[:len(loss_list)])
                        plt.xscale('log')
                        plt.ylim(0.0,1.3)
                        # plt.ylim(-.01,3)
                        plt.show()
                        
                    if PLOTTING_MASKING_LOSS_EVERY_STEP and PLOTTING_ALL_LOSSES_EVERY_STEP:
                        if TASK_TYPE in [TaskType.binary, TaskType.multiclass]:
                            print("WARNING - CLASSIFICATION MASKED PLOTS NOT SHOWN")
                        else:
                            if not FULL_SUBSET_INDEXING_TRACKING:
                                substeps = list(step_data.keys())
                                trn_mask_steps = np.stack([np.array(step_data[substep]['masking_trn_mse']) for substep in substeps])
                                print('trn_mask_steps',trn_mask_steps.shape)
                                print('substeps',len(substeps))
                                for d in range(D+1):
                                    mix_d = d / D
                                    # color_d = (1-mix_d)*np.array([0,1,0]) + (mix_d) * np.array([1,1,1])
                                    color_d = (mix_d)*np.array([0,1,0]) + (1-mix_d) * np.array([0,0,0])
                                    plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                # plt.plot(1+np.array(substeps),trn_steps)
                                # plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,2].reshape(-1)[:len(loss_list)])
                                # plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,3].reshape(-1)[:len(loss_list)])
                                plt.xscale('log')
                                plt.xlim(masking_start_step*2/3,None) #09/22/25
                                # plt.ylim(0.0,1.3)
                                plt.ylim(-.01,3)
                                plt.show()
                            if not FULL_SUBSET_INDEXING_TRACKING:
                                substeps = list(step_data.keys())
                                trn_mask_steps = np.stack([np.array(step_data[substep]['masking_trn_mse2']) for substep in substeps])
                                print('trn_mask_steps',trn_mask_steps.shape)
                                print('substeps',len(substeps))
                                for d in range(D+1):
                                    mix_d = d / D
                                    color_d = (mix_d)*np.array([.3,1,.3]) + (1-mix_d) * np.array([.2,.2,.2])
                                    plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                plt.xscale('log')
                                plt.xlim(masking_start_step*2/3,None)
                                plt.ylim(-.01,3)
                                plt.show()
                            if FULL_SUBSET_INDEXING_TRACKING:
                                substeps = list(step_data.keys())
                                trn_mask_steps = np.stack([np.array(step_data[substep]['masking_trn_mse']) for substep in substeps])
                                print('trn_mask_steps',trn_mask_steps.shape)
                                print('substeps',len(substeps))
                                for d in range(2**D0):
                                    # mix_d = d / D
                                    # color_d = (1-mix_d)*np.array([0,1,0]) + (mix_d) * np.array([1,1,1])
                                    # color_d = (mix_d)*np.array([0,1,0]) + (1-mix_d) * np.array([0,0,0])
                                    color_d = "C"+str(d)
                                    plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                plt.xscale('log')
                                # plt.xlim(masking_start_step*2/3,None)
                                plt.ylim(-.01,3)
                                plt.show()
                            # if DOUBLE_EARLY_STOPPING:
                            if DOUBLE_EARLY_STOPPING and not TRACK_VAL_SAMPLING:
                                plt.plot([1],[0])
                                if True:
                                    substeps = list(step_data.keys())
                                    trn_mask_steps = [step_data[substep]['mask_evaluation']['trn_metric'] for substep in substeps]
                                    val_mask_steps = [step_data[substep]['mask_evaluation']['val_metric'] for substep in substeps]
                                    plt.plot(1+np.array(substeps),val_mask_steps)
                                    plt.plot(1+np.array(substeps),trn_mask_steps)
                                plt.xscale('log')
                                plt.ylim(0.0,1.3)
                                # plt.ylim(-.01,3)
                                plt.show()
                            if DOUBLE_EARLY_STOPPING and TRACK_VAL_SAMPLING:
                                if FULL_SUBSET_INDEXING_TRACKING:
                                    substeps = list(step_data.keys())
                                    trn_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['trn_metric']) for substep in substeps])
                                    val_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['val_metric']) for substep in substeps])
                                    # substeps = list(step_data.keys())
                                    # trn_mask_steps = np.stack([np.array(step_data[substep]['masking_trn_mse']) for substep in substeps])
                                    print('trn_mask_steps',trn_mask_steps.shape)
                                    print('substeps',len(substeps))
                                    for d in range(2**D0):
                                        # mix_d = d / D
                                        # color_d = (1-mix_d)*np.array([0,1,0]) + (mix_d) * np.array([1,1,1])
                                        # color_d = (mix_d)*np.array([0,1,0]) + (1-mix_d) * np.array([0,0,0])
                                        color_d = "C"+str(d)
                                        plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                        plt.plot(1+np.array(substeps),val_mask_steps[:,d], c=color_d, linestyle='--')
                                    plt.xscale('log')
                                    # plt.xlim(masking_start_step*2/3,None)
                                    plt.ylim(-.01,3)
                                    plt.show()
                                    '''
                                    division
                                    '''
                                    substeps = list(step_data.keys())
                                    trn_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['trn_metric']) for substep in substeps])
                                    val_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['val_metric']) for substep in substeps])
                                    print('trn_mask_steps',trn_mask_steps.shape)
                                    print('substeps',len(substeps))
                                    for d in range(2**D0):
                                        color_d = "C"+str(d)
                                        plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                        plt.plot(1+np.array(substeps),val_mask_steps[:,d], c=color_d, linestyle='--')
                                    plt.xscale('log')
                                    plt.xlim(1000,None)
                                    plt.ylim(-.01,1.4)
                                    plt.show()

                                    
                                plt.plot([1],[0])
                                if True:
                                    substeps = list(step_data.keys())
                                    # trn_mask_steps = [step_data[substep]['mask_evaluation']['trn_metric'] for substep in substeps]
                                    # val_mask_steps = [step_data[substep]['mask_evaluation']['val_metric'] for substep in substeps]
                                    # trn_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['trn_metric']) for substep in substeps])
                                    # val_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['val_metric']) for substep in substeps])
                                    trn_mask_steps = [np.mean(step_data[substep]['mask_evaluation']['trn_metric']) for substep in substeps]
                                    val_mask_steps = [np.mean(step_data[substep]['mask_evaluation']['val_metric']) for substep in substeps]
                                    plt.plot(1+np.array(substeps),val_mask_steps)
                                    plt.plot(1+np.array(substeps),trn_mask_steps)
                                plt.xscale('log')
                                plt.ylim(0.0,1.3)
                                plt.ylim(-.01,3)
                                plt.show()

                    if (epochs_without_improvement >= early_stopping_patience):
                        print('early_stopping_patience',early_stopping_patience)
                        if VERBOSE_TRAINING:
                            print(f"Early stopping triggered after {k + 1} epochs with no improvement in validation {metric_name}")

                        if not MASKING_MODE: #unmasked can break
                            break
                        else: #masked must update training and go for that long
                            raise Exception('not currently using early stopping with double stopping')
                            masking_phase = True
                            opt = torch.optim.Adagrad(net.parameters(), lr=LR) #10/04/25 @ 11:00pm
                            print('Adagrad()')
                            opt = torch.optim.Adagrad(net.parameters(), lr=10*LR) #10/06/25 @ 4:00am
                            masking_end_step = masking_start_step + best_step
                            print("unmasked cant break") #10/06/25 @ 12:00pm
                            print('best_step',best_step)
                            print('masking_start_step',masking_start_step)
                            print('masking_end_step',masking_end_step)
                            mask_scheduler.STEP = masking_end_step #TODO: cleaner way of doing this
                            net = copy.deepcopy(best_net)
                            best_val_score = -float('inf')
                            best_step = masking_start_step
                            STEP = masking_end_step
                            print("Starting to mask ^_^ (early)")




                if saving_after_evaluation and step>=saving_after_evaluation_min_step:
                    all_training_results = {
                        "step_data": step_data,
                        "epoch_data": epoch_data,
                    }
                    experiment_data = training_args.saving_settings.experiment_data
                    saved_results_path = training_args.saving_settings.saved_results_path
                    experiment_data["sian_results.step_data"] = all_training_results["step_data"]
                    experiment_data["sian_results.epoch_data"] = all_training_results["epoch_data"]
                    experiment_data["SIAN_partial_training_time"] = partial_training_time
                    with open(saved_results_path, "w") as f:
                        json.dump(experiment_data, f, indent=4)
                    pass

                    
            metric_evaluation_time_taken += (time.time() - eval_start_time)

            if step >= STEP:
                k = EP-1
                break

        if TASK_TYPE == "regression":
            avg_epoch_loss = np.mean(all_trn_losses[k])
            if VERBOSE_TRAINING:
                print(f"Epoch {k} Average Training Loss: {avg_epoch_loss:.4f}")

        if PLOTTING_ALL_LOSSES_EVERY_EPOCH:
            plt.plot(1+np.arange(len(loss_list)),loss_list)
            if True:
                substeps = list(step_data.keys())
                trn_steps = [step_data[substep]['trn_metric'] for substep in substeps]
                val_steps = [step_data[substep]['val_metric'] for substep in substeps]
                plt.plot(1+np.array(substeps),val_steps)
                plt.plot(1+np.array(substeps),trn_steps)
            plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,2].reshape(-1)[:len(loss_list)])
            plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,3].reshape(-1)[:len(loss_list)])
            plt.xscale('log')
            plt.ylim(0.0,1.3)
            # plt.ylim(-.01,3)
            plt.show()

            if False:
                plt.figure(figsize=(8,5.5))
                plt.plot(np.mean(all_losses[:(k+1),:,0],axis=1),c='C2',alpha=.4)
                plt.plot(all_val_losses[:k],c='C1')
                plt.plot(np.mean(all_losses[:(k+1),:,1],axis=1),c='C0',alpha=.4)
                plt.plot(np.mean(all_losses[:(k+1),:,2],axis=1),c='C3')
                plt.plot(all_trn_accs[:k],c='C0',linestyle='--')
                plt.plot(all_val_accs[:k],c='C1',linestyle='--')

                
                ###plt.plot(all_val_losses1[:k],c='C4')
                ###plt.plot(all_val_losses2[:k],c='C6')
                plt.plot([-50,EP+50],[0,0],c='k')
                #plt.legend(['total','val CE','trn CE','L1'])
                plt.legend(['total','val MSE','trn MSE','L1','','',])#'XOR_3','XOR_5'])
                
                #plt.ylim(-.01,2)
                plt.ylim(-.01,3)
                plt.xscale('log')
                plt.title("{ReLU MLP} Loss While Training")
                plt.show()
        
        if VERBOSE_TRAINING:
            print('epoch', str(k) + "'")
        net.eval()

        # if (k+1)%EVALUATE_EVERY_K_EPOCHS==0 or (k+1)==EP:
        if (k+1) in epochs_to_evaluate or (k+1)%EVALUATE_EVERY_K_EPOCHS==0 or (k+1)==EP:
            pass
            # eval_start_time = time.time()

            # evaluation = get_evaluation()
            # epoch_data[k+1] = evaluation


            # val_metric = evaluation['val_metric']
            # val_score = -val_metric if TASK_TYPE == TaskType.regression else val_metric

            # if val_score > best_val_score:
            #     # print(f"Val score: {val_score}, Best val score: {best_val_score}")
            #     best_val_score = val_score
            #     best_net = copy.deepcopy(net) #NOTE: slight concern with keeping two model copies on the GPU
            #     epochs_without_improvement = 0
            #     print(f"Epochs without improvement = {epochs_without_improvement}")
            # else:
            #     epochs_without_improvement += 1
            #     print(f"Epochs without improvement = {epochs_without_improvement}")

            # if VERBOSE_TRAINING:
            #     metric_name = "MSE" if TASK_TYPE == TaskType.regression else "ROC AUC"
            #     print(f'{metric_name} for train and val: {trn_metric:.4f}, {val_metric:.4f}')

            # if epochs_without_improvement >= early_stopping_patience:
            #     if VERBOSE_TRAINING:
            #         print(f"Early stopping triggered after {k + 1} epochs with no improvement in validation {metric_name}")
            #     break
            # metric_evaluation_time_taken += (time.time() - eval_start_time)
                
        epoch_time = time.time() - epoch_start_time
        if VERBOSE_TRAINING:
            print(f"--- {epoch_time:.3f} seconds in epoch ---")


        if PLOTTING_ALL_LOSSES_EVERY_EPOCH:
            val_loss_list.append(val_metric)

            # #JAM -- changing to this version for now
            pass
            # k=k+1
            # plt.figure(figsize=(10,7))
            # plt.plot(np.mean(all_losses[:k,:,0],axis=1))
            # plt.plot(val_loss_list)
            # plt.plot(np.mean(all_losses[:k,:,1],axis=1))
            # plt.plot(np.mean(all_losses[:k,:,2],axis=1))
            # plt.plot([-50,EP+50],[0,0],c='k')
            # # plt.legend(['trn loss','val','trn mse','trn l1'])
            # plt.legend(['trn loss','val mse','trn mse','trn l1'])
            # plt.xlim(-2,k+2)
            # plt.title("DNN/MLP Loss While Training")
            # plt.show()


    total_training_time = time.time() - full_training_start_time
    if VERBOSE_TRAINING:
        print('FULLY TRAINED USING', total_training_time, 'seconds')

    if MASKING_MODE: #early stopping from masking phase so 'best_net' is 'final_net'
        best_net = copy.deepcopy(net)

    if 'final_net' in results_to_save: 
        file_prefix = results_save_prefix
        if 'sian' in net_name:
            net.compress()
        final_model_path = file_prefix + net_name + '_final_net.pt'
        ### final_model_path = file_prefix + '_final_net.pt' #05/22/26 - windows file name sizes again
        print('final_model_path',final_model_path)
        torch.save(net,  final_model_path)
        # torch.save(net.state_dict(),  file_prefix + net_name + '_final_net.pt')
        if 'sian' in net_name:
            net.blocksparse()

    if 'best_net' in results_to_save: #NOTE: slight concern with keeping two model copies on the GPU
        file_prefix = results_save_prefix
        if 'sian' in net_name:
            best_net.compress()
        best_model_path = file_prefix + net_name + '_best_net.pt'
        torch.save(best_net, best_model_path)
        # torch.save(best_net.state_dict(),  file_prefix + net_name + '_best_net.pt')
        if 'sian' in net_name:
            best_net.blocksparse()

    all_training_results = {
        "step_data": step_data,
        "epoch_data": epoch_data,
        "total_training_time": total_training_time,

        "final_model_path" : final_model_path,
        "best_model_path" : best_model_path,

        "gradient_training_time_taken": gradient_training_time_taken,
        "metric_evaluation_time_taken": metric_evaluation_time_taken,
    }
    if training_args.return_settings.return_val_tensor:
        if val_tensor is None:
            val_tensor = dataset_object.pull_val_tensor(device)
        all_training_results["val_tensor"] = val_tensor
    if training_args.return_settings.return_val_output:
        val_output = dataset_object.pull_val_output(device)
        all_training_results["val_output"] = val_output
    if training_args.return_settings.return_final_net:
        all_training_results["final_net"] = net
    if training_args.return_settings.return_best_net:
        all_training_results["best_net"] = best_net
    return all_training_results










def surrogate_based___gradient_descent_training(dataset_object, net, target_net, training_args, level_of_surrogate=1):
        
    BS = training_args.batch_size
    EP = training_args.number_of_epochs
    LR = training_args.learning_rate
    STEP = training_args.number_of_steps
    
    lambda1 = training_args.lambda1
    lambda2 = training_args.lambda2

    lambda1shapeloss = training_args.lambda1shapeloss
    COMPUTE_SHAPELOSS = training_args.compute_shapeloss
    if COMPUTE_SHAPELOSS:
        net.gam.compute_shapeloss_while_training = True



    USE_GRADIENT_CLIPPING = training_args.USE_GRADIENT_CLIPPING
    # LOGIT_CLAMPING = training_args.LOGIT_CLAMPING
    early_stopping_patience = training_args.early_stopping_patience    
    VERBOSE_TRAINING = training_args.verbosity_settings.VERBOSE_TRAINING
    MASKING_MODE = training_args.model_config.is_masked
    TASK_TYPE = training_args.task_type 
    print("MASKING_MODE",MASKING_MODE)


    EVALUATE_WITH_TRN_TENSOR = training_args.EVALUATE_WITH_TRN_TENSOR
    EVALUATE_WITH_VAL_TENSOR = training_args.EVALUATE_WITH_VAL_TENSOR
    PLOTTING_ALL_LOSSES_EVERY_EPOCH = training_args.PLOTTING_ALL_LOSSES_EVERY_EPOCH
    # There is a bug in the plotting for non-regression datasets
    PLOTTING_ALL_LOSSES_EVERY_EPOCH = True #DEBUG
    PLOTTING_ALL_LOSSES_EVERY_STEP  = True #DEBUG
    PLOTTING_MASKING_LOSS_EVERY_STEP = True #DEBUG
    PLOTTING_ALL_LOSSES_EVERY_EPOCH = False #no more DEBUG
    PLOTTING_ALL_LOSSES_EVERY_STEP  = False #no more DEBUG
    PLOTTING_MASKING_LOSS_EVERY_STEP = False #no more DEBUG
    
    # SAVING_MASK_EVALUATION = True #DEBUG -- very memory intensive for saving (especially depending on size-based or individual-mask-based)
    SAVING_MASK_EVALUATION = False
    
    # DOUBLE_EARLY_STOPPING = True
    DOUBLE_EARLY_STOPPING = False

    

    DOUBLE_EARLY_STOPPING = (DOUBLE_EARLY_STOPPING and MASKING_MODE)
    PLOTTING_MASKING_LOSS_EVERY_STEP = (PLOTTING_MASKING_LOSS_EVERY_STEP and MASKING_MODE)
    if PLOTTING_ALL_LOSSES_EVERY_EPOCH or PLOTTING_ALL_LOSSES_EVERY_STEP:
        loss_list = []
        val_loss_list = []


    EVALUTE_TEST_DURING_RUN = True #08/17/2025 -- turning on for synthetic sweeps, but might be fine to keep


    steps_to_evaluate = training_args.return_settings.steps_to_evaluate
    epochs_to_evaluate = training_args.return_settings.epochs_to_evaluate

    results_save_prefix = training_args.saving_settings.results_save_prefix  
    results_to_save = training_args.saving_settings.results_to_save 
    net_name = training_args.saving_settings.net_name 
    saving_after_evaluation = training_args.saving_settings.saving_after_evaluation
    saving_after_evaluation_min_step = training_args.saving_settings.saving_after_evaluation_min_step
    

    device = training_args.device
    net = net.to(device)



    normalize_XY = training_args.normalize_XY
    dataset_object.shuffle_and_split_trnval(trnval_shuffle_seed=training_args.trnval_shuffle_seed, trnval_split_percentage=training_args.trnval_split_percentage, trnval_reduc_percentage=training_args.trainval_reduction_percentage)
    trn_loader, val_loader = dataset_object.pull_trnval_loaders(device, BS, training_args.batch_shuffling_seed, normalize_XY[0], normalize_XY[1])
    if EVALUTE_TEST_DURING_RUN:
        tst_loader = dataset_object.pull_tst_loaders(device, BS, training_args.batch_shuffling_seed, normalize_XY[0], normalize_XY[1])

    if False: #TODO: can no longer do this
        if hasattr(net,'gam'): 
            if training_args.loss_type=="softmax":
                net.gam.bias = torch.nn.Parameter(  torch.Tensor(np.log(np.mean(trnY,axis=0)))   )


    print(f"Expected steps per epoch: {len(trn_loader)} (Dataset size: {len(trn_loader.dataset)}, Batch size: {BS})")
    print(f"also expected val steps per epoch: {len(val_loader)} (Dataset size: {len(val_loader.dataset)}, Batch size: {BS})")

    # Adjust batch size if dataset is too small
    if len(trn_loader.dataset) < BS:
        BS = len(trn_loader.dataset)
        print(f"Adjusted batch size to {BS} to match training dataset size.")
        trn_loader, val_loader = dataset_object.pull_trnval_loaders(device, BS, training_args.batch_shuffling_seed, normalize_XY[0], normalize_XY[1])
        if EVALUTE_TEST_DURING_RUN:
            tst_loader = dataset_object.pull_tst_loaders(device, BS, training_args.batch_shuffling_seed, normalize_XY[0], normalize_XY[1])
        print(f"Updated steps per epoch: {len(trn_loader)} (Dataset size: {len(trn_loader.dataset)}, Batch size: {BS})")

    trn_tensor = None
    val_tensor = None
    if EVALUATE_WITH_TRN_TENSOR:	
        trn_tensor = dataset_object.pull_trn_tensor(device)
    if EVALUATE_WITH_VAL_TENSOR:	
        val_tensor = dataset_object.pull_val_tensor(device)
    assert not EVALUATE_WITH_TRN_TENSOR, "only trn_loader implemented right now"
    assert not EVALUATE_WITH_VAL_TENSOR, "only val_loader implemented right now"

    if training_args.opt_type=="Adagrad":
        opt = torch.optim.Adagrad(net.parameters(), lr=LR)
    else:
        raise Exception(f"training_args.opt_type={training_args.opt_type} not recognized")
    
    if EP is None:
        EP = (STEP // len(trn_loader)) + 2
    EVALUATE_EVERY_K_EPOCHS = training_args.EVALUATE_EVERY_K_EPOCHS
    if EVALUATE_EVERY_K_EPOCHS is None:
        EVALUATE_EVERY_K_EPOCHS = EP
    if STEP is None:
        STEP = EP * len(trn_loader)
    
    D = dataset_object.get_D()
    n_classes = dataset_object.get_C()
    all_trn_accs = np.zeros(EP)
    all_val_accs = np.zeros(EP)
    all_losses = np.zeros((EP,len(trn_loader),7))
    all_trn_losses = np.zeros((EP,len(trn_loader)))
    all_val_losses = np.zeros(EP)
    
    gradient_training_time_taken = 0.0
    metric_evaluation_time_taken = 0.0
    

    if MASKING_MODE:
        masking_phase = False
        mask_scheduler = None   #TODO: move outside
        if mask_scheduler is None:
            mask_scheduler = MaskScheduler(dataset_object.get_grouped_feature_dict())
        mask_scheduler2 = MaskScheduler(dataset_object.get_grouped_feature_dict())
        mask_scheduler2.mask_prob_scheduler = lambda mystep : 1.0 #full prob of random mask for evaluation (otherwise cant early stop)

        if STEP is not None:
            assert (STEP is not None), f'EP{EP},STEP{STEP}'
            mask_scheduler.initialize_max_step(STEP)
            masking_start_step = mask_scheduler.get_min_masking_step() #defaults to the correct STEP/2, future could do more complex mixings
        else:
            assert (EP==0), f'EP{EP},STEP{STEP}'

        all_subset_losses = torch.zeros((EP, D+1, 2)).to(device)
        subset_indexer = torch.ones(D).float().to(device) #NOTE: "addmv_impl_cuda" not implemented for 'Long'


        FULL_SUBSET_INDEXING_TRACKING = False
        # FULL_SUBSET_INDEXING_TRACKING = True
        if FULL_SUBSET_INDEXING_TRACKING: #getting freaky with full indexing on 2^d
            # all_subset_losses = torch.zeros((1, 2**D, 2)).to(device)
            # subset_indexer = (2**torch.arange(D).float()).to(device) #NOTE: "addmv_impl_cuda" not implemented for 'Long'
            # print('subset_indexer',subset_indexer) #actually this even breaks down for 2^25 because of the float accuracy issues
            if True:
                grouped_feat_dict = dataset_object.get_grouped_feature_dict()
                D0 = grouped_feat_dict['D0']
                all_subset_losses = torch.zeros((1, 2**D0, 2)).to(device)
                subset_indexer = torch.zeros(D).float().to(device)
                cum_ii = 0
                for i in range(D0):
                    subset_indexer[cum_ii] = 2**i
                    cum_ii += len(grouped_feat_dict[i])
                print('subset_indexer',subset_indexer)
        else:
            grouped_feat_dict = dataset_object.get_grouped_feature_dict()
    
    best_val_score = -float('inf')
    best_net = None
    best_step = 0


    if early_stopping_patience is None:
        # early_stopping_patience = EP
        early_stopping_patience = float('inf')

    epochs_without_improvement = 0
    
    trn_metric = 0.0 #enables passing "EP = 0"
    val_metric = 0.0 #enables passing "EP = 0"

    PRINT_LOSS_PER_STEP = True
    PRINT_LOSS_PER_STEP = False #09/20/25 @ 2:55pm -- too verbose jupyter memory
    step = 0
    step_data  = {}
    epoch_data = {}



    def evaluate_during_training(data_loader, mask_sampler=None): #NOTE: be aware that "net" and "masking_mode" are defined implicitly here  (and now "level_of_surrogate" is too)
        if TASK_TYPE == TaskType.regression:
            if mask_sampler is None:
                if level_of_surrogate==1:
                    __metric = evaluate_during_training_from_dataloader_REG(net, data_loader, MASKING_MODE, mask_sampler=mask_sampler)
                    # print('__metric',__metric)
                elif level_of_surrogate==2:
                    pass
                    __metric = 0.0 #TODO: it seems like I would need to tear out "MASKING_MODE" and turn it into an integer to fix this, so let's ignore it for now.
                else:
                    raise NotImplementedError("too meta for now")
            if mask_sampler is not None:
                if level_of_surrogate==1:
                    eval_all_subset_losses = evaluate_during_training_from_dataloader_REG(net, data_loader, MASKING_MODE, mask_sampler=mask_sampler, grouped_feat_dict=grouped_feat_dict)
                    # print('eval_all_subset_losses',eval_all_subset_losses)
                    __metric = [float(x) for x in ((eval_all_subset_losses[:,0]) / (eval_all_subset_losses[:,1])).cpu().numpy() ]
                elif level_of_surrogate==2:
                    pass
                    __metric = 0.0 #TODO: it seems like I would need to tear out "MASKING_MODE" and turn it into an integer to fix this, so let's ignore it for now.
                else:
                    raise NotImplementedError("too meta for now")
        elif TASK_TYPE == TaskType.binary:
            __metric = evaluate_during_training_from_dataloader_BINARY_auc(net, data_loader, MASKING_MODE, mask_sampler=mask_sampler)
        elif TASK_TYPE == TaskType.multiclass:
            C = dataset_object.get_C()
            __metric = evaluate_during_training_from_dataloader_MULTI_auc(net, data_loader, MASKING_MODE, n_classes=C, mask_sampler=mask_sampler)
        else:
            raise Exception("not implemented task")
            __metric = 0.0
        return __metric
    
    def get_evaluation(): #NOTE: be aware that "net" and "masking_mode" are defined implicitly here 
        net.eval() 
        trn_metric = evaluate_during_training(trn_loader)
        val_metric = evaluate_during_training(val_loader)
        evaluation = {"trn_metric":trn_metric, "val_metric":val_metric}
        if EVALUTE_TEST_DURING_RUN:
            tst_metric = evaluate_during_training(tst_loader)
            evaluation["tst_metric"] = tst_metric
        net.train()
        return evaluation
    
    def get_mask_evaluation(mask_sampler): #NOTE: be aware that "net" and "masking_mode" are defined implicitly here 
        net.eval() 
        trn_metric = evaluate_during_training(trn_loader, mask_sampler)
        val_metric = evaluate_during_training(val_loader, mask_sampler)
        evaluation = {"trn_metric":trn_metric, "val_metric":val_metric}
        if EVALUTE_TEST_DURING_RUN:
            tst_metric = evaluate_during_training(tst_loader, mask_sampler)
            evaluation["tst_metric"] = tst_metric
        net.train()
        return evaluation






    if 0 in steps_to_evaluate or 0 in epochs_to_evaluate:
        evaluation = get_evaluation()
        if PLOTTING_MASKING_LOSS_EVERY_STEP:
            # evaluation['masking_trn_mse'] = np.zeros(D+1) + float('inf')
            evaluation['masking_trn_mse'] = [float('inf') for d in range(D+1)]
            evaluation['masking_trn_mse2'] = [float('inf') for d in range(D+1)]
            if FULL_SUBSET_INDEXING_TRACKING:
                evaluation['masking_trn_mse'] = [float('inf') for d in range(2**D0)]
                evaluation['masking_trn_mse2'] = [float('inf') for d in range(2**D0)]

        if DOUBLE_EARLY_STOPPING:
            mask_evaluation = get_mask_evaluation(mask_scheduler2)
            if SAVING_MASK_EVALUATION:
                evaluation['mask_evaluation'] = mask_evaluation
        if 0 in steps_to_evaluate:
            step_data[0] = evaluation
        if 0 in epochs_to_evaluate:
            epoch_data[0] = evaluation


    full_training_start_time = time.time()
    for k in range(EP):
        if step >= STEP:
            k = EP-1
            break
        if VERBOSE_TRAINING:
            print('Epoch', k)
        epoch_start_time = time.time()

        net.train()
        for j, (x_batch, y_batch) in enumerate(trn_loader):
            # print('x_batch',x_batch.shape)
            # print('y_batch',y_batch.shape)
            # if j>0:
            #     break #09/20/25 @ 4:45pm -- THIS IS HOW I USED TO DO STEPS I THINK????
            if MASKING_MODE:
                if step < masking_start_step and masking_phase:
                    print(f'we early stopped the vanilla part step={step}')
                    continue
                # if step == masking_start_step:
                if step >= masking_start_step and (not masking_phase):
                    masking_phase = True #not actually useful right now
                    opt = torch.optim.Adagrad(net.parameters(), lr=LR) #10/04/25 @ 11:00pm
                    print('Adagrad()')
                    opt = torch.optim.Adagrad(net.parameters(), lr=10*LR) #10/06/25 @ 4:00am
                    # if best_net is not None:
                    if False: #10/07/25 @ 2:30am
                        net = copy.deepcopy(best_net) #09/22/25 @ 12:10am -- I had forgotten to put this back in 
                    best_val_score = -float('inf')
                    best_step = masking_start_step
                    print("Starting to mask ^_^")
            grad_start_time = time.time()
            if torch.any(torch.isnan(x_batch)) or torch.any(torch.isinf(x_batch)):
                print(f"Warning: NaNs or infinities in train batch {j}")
                continue

            if not MASKING_MODE:
                dnn_logits, gam_logits, shape_loss = net(x_batch)
                logits = dnn_logits + gam_logits
            else:
                if level_of_surrogate==1:
                    s_batch = mask_scheduler.get_mask(x_batch, step)
                    dnn_logits, gam_logits, shape_loss = net((x_batch, s_batch))
                    logits = dnn_logits + gam_logits
                    f_batch = target_net(x_batch)
                    f_batch = (f_batch[0]+f_batch[1]).detach()
                elif level_of_surrogate==2:
                    s_batch = mask_scheduler.get_mask(x_batch, step)
                    t_batch = mask_scheduler.get_mask(x_batch, step)
                    dnn_logits, gam_logits, shape_loss = net((x_batch, s_batch, t_batch))
                    logits = dnn_logits + gam_logits
                    f_batch = target_net((x_batch, s_batch))
                    f_batch = (f_batch[0]+f_batch[1]).detach()
                else:
                    raise NotImplementedError(f"that's too meta for now, bro")
                    
                    
            l1_reg = torch.zeros(1).to(device)
            l2_reg = torch.zeros(1).to(device)
            if lambda1 > 0 or lambda2 > 0:
                all_linear_params = net.collectParameters()
                l1_reg = lambda1 * torch.norm(all_linear_params, 1)
                l2_reg = lambda2 * torch.norm(all_linear_params, 2)


            if training_args.loss_type=="mse":
                if False:
                    pass
                    #mseloss_ = (y_batch.narrow(1, 0, 1) - logits.narrow(1, 0, 1)) ** 2
                else:
                    mseloss_ = (f_batch.narrow(1, 0, 1) - logits.narrow(1, 0, 1)) ** 2
                mseloss = torch.mean(mseloss_)
            elif training_args.loss_type=="ce":
                raise NotImplementedError(f"TODO cross-ent surrogates")
                logprobs = torch.nn.LogSigmoid()(logits)
                lognotprobs = torch.nn.LogSigmoid()(-logits)
                celoss_ = -y_batch*logprobs-(1-y_batch)*lognotprobs
                celoss = torch.mean(celoss_)
                mseloss = celoss # misnomer
            elif training_args.loss_type=="softmax":
                raise NotImplementedError(f"TODO softmax surrogates")
                logprobs = torch.nn.LogSoftmax(dim=-1)(logits)
                smloss_ = -y_batch*logprobs
                smloss = torch.mean( torch.sum(smloss_,dim=-1))
                mseloss = smloss # misnomer
            else:
                raise NotImplementedError(f"loss_type={training_args.loss_type}")
            
            if PLOTTING_ALL_LOSSES_EVERY_EPOCH:
                loss_list.append(mseloss.item())

            # loss = mseloss + l1_reg
            loss = mseloss + l1_reg + l2_reg
            if COMPUTE_SHAPELOSS:
                actual_loss_shapeloss = lambda1shapeloss * torch.sum( shape_loss )
                loss += actual_loss_shapeloss
            if VERBOSE_TRAINING and PRINT_LOSS_PER_STEP:
                print(f"Epoch {k}, Step {j}, Loss: {loss.item():.4f}")
            loss.backward()

            if USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            opt.step()
            opt.zero_grad()
            step += 1

            all_trn_losses[k, j] = mseloss.item()
            all_losses[k, j, 0] = loss.item()
            all_losses[k, j, 1] = mseloss.item()
            all_losses[k, j, 2] = l1_reg.item()
            all_losses[k, j, 3] = l2_reg.item()

            
            if PLOTTING_MASKING_LOSS_EVERY_STEP and MASKING_MODE and training_args.loss_type == "mse": #TODO: look into costliness of this
                s_index = torch.matmul(s_batch.float(), subset_indexer).long()
                if not FULL_SUBSET_INDEXING_TRACKING:
                    all_subset_losses[k, :, 0].index_put_((s_index,), mseloss_[:, 0].detach(), accumulate=True)
                    all_subset_losses[k, :, 1].index_put_((s_index,), torch.ones_like(mseloss_.detach())[:, 0], accumulate=True)
                else:
                    all_subset_losses[0, :, 0].index_put_((s_index,), mseloss_[:, 0].detach(), accumulate=True)
                    all_subset_losses[0, :, 1].index_put_((s_index,), torch.ones_like(mseloss_.detach())[:, 0], accumulate=True)
                
            gradient_training_time_taken += (time.time() - grad_start_time)            

            eval_start_time = time.time()
            if step in steps_to_evaluate:
                evaluation = get_evaluation()
                print(step,'evaluation',evaluation) #TODO: remove, only for debugging
                if True: #09/13/2025 - TODO: need this?
                    evaluation["L1"] = float(l1_reg.item())
                    evaluation["L2"] = float(l2_reg.item())
                    if COMPUTE_SHAPELOSS:
                        evaluation["shapeloss"] = float(actual_loss_shapeloss.item())
                    partial_training_time = time.time() - full_training_start_time
                    evaluation["partial_training_time"] = partial_training_time
                if PLOTTING_MASKING_LOSS_EVERY_STEP: #09/21/25 -- also doing looking at masking
                    evaluation["masking_trn_mse"] = [float(x) for x in (torch.sum(all_subset_losses[:,:,0],dim=0) / torch.sum(all_subset_losses[:,:,1],dim=0)).cpu().numpy() ]

                    if True:
                        if not FULL_SUBSET_INDEXING_TRACKING:
                            evaluation["masking_trn_mse2"] = [float(x) for x in (all_subset_losses[k,:,0] / all_subset_losses[k,:,1]).cpu().numpy() ]
                        else:
                            evaluation["masking_trn_mse2"] = [float(x) for x in (all_subset_losses[0,:,0] / all_subset_losses[0,:,1]).cpu().numpy() ]

                # print(step,'evaluation',evaluation) #TODO: remove, only for debugging

                if DOUBLE_EARLY_STOPPING:
                    mask_evaluation = get_mask_evaluation(mask_scheduler2)
                    if SAVING_MASK_EVALUATION:
                        evaluation['mask_evaluation'] = mask_evaluation
                # print(step,'evaluation',evaluation) #TODO: remove, only for debugging
                step_data[step] = evaluation

                if True: #checking evaluation for best_net
                    val_metric = evaluation['val_metric']
                    # if DOUBLE_EARLY_STOPPING:
                    if DOUBLE_EARLY_STOPPING and masking_phase: # TODO TODO: check importance
                        val_metric = mask_evaluation['val_metric']
                        if TRACK_VAL_SAMPLING:
                            val_metric = np.mean(val_metric)
                    val_score = -val_metric if TASK_TYPE == TaskType.regression else val_metric

                    if val_score > best_val_score:
                        # print(f"Val score: {val_score}, Best val score: {best_val_score}")
                        best_val_score = val_score
                        best_step = step
                        best_net = copy.deepcopy(net) #NOTE: slight concern with keeping two model copies on the GPU
                        epochs_without_improvement = 0
                        print(f"Steps without improvement = {epochs_without_improvement}")
                    else:
                        epochs_without_improvement += 1
                        print(f"Steps without improvement = {epochs_without_improvement}")

                    if VERBOSE_TRAINING:
                        trn_metric = evaluation['val_metric']
                        metric_name = "MSE" if TASK_TYPE == TaskType.regression else "ROC-AUC"
                        print(f'{metric_name} for train and val: {trn_metric:.4f}, {val_metric:.4f}')

                        
                    if PLOTTING_ALL_LOSSES_EVERY_STEP:
                        plt.plot(1+np.arange(len(loss_list)),loss_list)
                        if True:
                            substeps = list(step_data.keys())
                            trn_steps = [step_data[substep]['trn_metric'] for substep in substeps]
                            val_steps = [step_data[substep]['val_metric'] for substep in substeps]
                            plt.plot(1+np.array(substeps),val_steps)
                            plt.plot(1+np.array(substeps),trn_steps)
                        plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,2].reshape(-1)[:len(loss_list)])
                        plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,3].reshape(-1)[:len(loss_list)])
                        plt.xscale('log')
                        plt.ylim(0.0,1.3)
                        # plt.ylim(-.01,3)
                        plt.show()
                        
                    if PLOTTING_MASKING_LOSS_EVERY_STEP and PLOTTING_ALL_LOSSES_EVERY_STEP:
                        if TASK_TYPE in [TaskType.binary, TaskType.multiclass]:
                            print("WARNING - CLASSIFICATION MASKED PLOTS NOT SHOWN")
                        else:
                            if not FULL_SUBSET_INDEXING_TRACKING:
                                substeps = list(step_data.keys())
                                trn_mask_steps = np.stack([np.array(step_data[substep]['masking_trn_mse']) for substep in substeps])
                                print('trn_mask_steps',trn_mask_steps.shape)
                                print('substeps',len(substeps))
                                for d in range(D+1):
                                    mix_d = d / D
                                    # color_d = (1-mix_d)*np.array([0,1,0]) + (mix_d) * np.array([1,1,1])
                                    color_d = (mix_d)*np.array([0,1,0]) + (1-mix_d) * np.array([0,0,0])
                                    plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                # plt.plot(1+np.array(substeps),trn_steps)
                                # plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,2].reshape(-1)[:len(loss_list)])
                                # plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,3].reshape(-1)[:len(loss_list)])
                                plt.xscale('log')
                                plt.xlim(masking_start_step*2/3,None) #09/22/25
                                # plt.ylim(0.0,1.3)
                                plt.ylim(-.01,3)
                                plt.show()
                            if not FULL_SUBSET_INDEXING_TRACKING:
                                substeps = list(step_data.keys())
                                trn_mask_steps = np.stack([np.array(step_data[substep]['masking_trn_mse2']) for substep in substeps])
                                print('trn_mask_steps',trn_mask_steps.shape)
                                print('substeps',len(substeps))
                                for d in range(D+1):
                                    mix_d = d / D
                                    color_d = (mix_d)*np.array([.3,1,.3]) + (1-mix_d) * np.array([.2,.2,.2])
                                    plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                plt.xscale('log')
                                plt.xlim(masking_start_step*2/3,None)
                                plt.ylim(-.01,3)
                                plt.show()
                            if FULL_SUBSET_INDEXING_TRACKING:
                                substeps = list(step_data.keys())
                                trn_mask_steps = np.stack([np.array(step_data[substep]['masking_trn_mse']) for substep in substeps])
                                print('trn_mask_steps',trn_mask_steps.shape)
                                print('substeps',len(substeps))
                                for d in range(2**D0):
                                    # mix_d = d / D
                                    # color_d = (1-mix_d)*np.array([0,1,0]) + (mix_d) * np.array([1,1,1])
                                    # color_d = (mix_d)*np.array([0,1,0]) + (1-mix_d) * np.array([0,0,0])
                                    color_d = "C"+str(d)
                                    plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                plt.xscale('log')
                                # plt.xlim(masking_start_step*2/3,None)
                                plt.ylim(-.01,3)
                                plt.show()
                            # if DOUBLE_EARLY_STOPPING:
                            if DOUBLE_EARLY_STOPPING and not TRACK_VAL_SAMPLING:
                                plt.plot([1],[0])
                                if True:
                                    substeps = list(step_data.keys())
                                    trn_mask_steps = [step_data[substep]['mask_evaluation']['trn_metric'] for substep in substeps]
                                    val_mask_steps = [step_data[substep]['mask_evaluation']['val_metric'] for substep in substeps]
                                    plt.plot(1+np.array(substeps),val_mask_steps)
                                    plt.plot(1+np.array(substeps),trn_mask_steps)
                                plt.xscale('log')
                                plt.ylim(0.0,1.3)
                                # plt.ylim(-.01,3)
                                plt.show()
                            if DOUBLE_EARLY_STOPPING and TRACK_VAL_SAMPLING:
                                if FULL_SUBSET_INDEXING_TRACKING:
                                    substeps = list(step_data.keys())
                                    trn_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['trn_metric']) for substep in substeps])
                                    val_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['val_metric']) for substep in substeps])
                                    # substeps = list(step_data.keys())
                                    # trn_mask_steps = np.stack([np.array(step_data[substep]['masking_trn_mse']) for substep in substeps])
                                    print('trn_mask_steps',trn_mask_steps.shape)
                                    print('substeps',len(substeps))
                                    for d in range(2**D0):
                                        # mix_d = d / D
                                        # color_d = (1-mix_d)*np.array([0,1,0]) + (mix_d) * np.array([1,1,1])
                                        # color_d = (mix_d)*np.array([0,1,0]) + (1-mix_d) * np.array([0,0,0])
                                        color_d = "C"+str(d)
                                        plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                        plt.plot(1+np.array(substeps),val_mask_steps[:,d], c=color_d, linestyle='--')
                                    plt.xscale('log')
                                    # plt.xlim(masking_start_step*2/3,None)
                                    plt.ylim(-.01,3)
                                    plt.show()
                                    '''
                                    division
                                    '''
                                    substeps = list(step_data.keys())
                                    trn_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['trn_metric']) for substep in substeps])
                                    val_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['val_metric']) for substep in substeps])
                                    print('trn_mask_steps',trn_mask_steps.shape)
                                    print('substeps',len(substeps))
                                    for d in range(2**D0):
                                        color_d = "C"+str(d)
                                        plt.plot(1+np.array(substeps),trn_mask_steps[:,d], c=color_d)
                                        plt.plot(1+np.array(substeps),val_mask_steps[:,d], c=color_d, linestyle='--')
                                    plt.xscale('log')
                                    plt.xlim(1000,None)
                                    plt.ylim(-.01,1.4)
                                    plt.show()

                                    
                                plt.plot([1],[0])
                                if True:
                                    substeps = list(step_data.keys())
                                    # trn_mask_steps = [step_data[substep]['mask_evaluation']['trn_metric'] for substep in substeps]
                                    # val_mask_steps = [step_data[substep]['mask_evaluation']['val_metric'] for substep in substeps]
                                    # trn_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['trn_metric']) for substep in substeps])
                                    # val_mask_steps = np.stack([np.array(step_data[substep]['mask_evaluation']['val_metric']) for substep in substeps])
                                    trn_mask_steps = [np.mean(step_data[substep]['mask_evaluation']['trn_metric']) for substep in substeps]
                                    val_mask_steps = [np.mean(step_data[substep]['mask_evaluation']['val_metric']) for substep in substeps]
                                    plt.plot(1+np.array(substeps),val_mask_steps)
                                    plt.plot(1+np.array(substeps),trn_mask_steps)
                                plt.xscale('log')
                                plt.ylim(0.0,1.3)
                                plt.ylim(-.01,3)
                                plt.show()

                    if (epochs_without_improvement >= early_stopping_patience):
                        print('early_stopping_patience',early_stopping_patience)
                        if VERBOSE_TRAINING:
                            print(f"Early stopping triggered after {k + 1} epochs with no improvement in validation {metric_name}")

                        if not MASKING_MODE: #unmasked can break
                            break
                        else: #masked must update training and go for that long
                            raise Exception('not currently using early stopping with double stopping')
                            masking_phase = True
                            opt = torch.optim.Adagrad(net.parameters(), lr=LR) #10/04/25 @ 11:00pm
                            print('Adagrad()')
                            opt = torch.optim.Adagrad(net.parameters(), lr=10*LR) #10/06/25 @ 4:00am
                            masking_end_step = masking_start_step + best_step
                            print("unmasked cant break") #10/06/25 @ 12:00pm
                            print('best_step',best_step)
                            print('masking_start_step',masking_start_step)
                            print('masking_end_step',masking_end_step)
                            mask_scheduler.STEP = masking_end_step #TODO: cleaner way of doing this
                            net = copy.deepcopy(best_net)
                            best_val_score = -float('inf')
                            best_step = masking_start_step
                            STEP = masking_end_step
                            print("Starting to mask ^_^ (early)")




                if saving_after_evaluation and step>=saving_after_evaluation_min_step:
                    all_training_results = {
                        "step_data": step_data,
                        "epoch_data": epoch_data,
                    }
                    experiment_data = training_args.saving_settings.experiment_data
                    saved_results_path = training_args.saving_settings.saved_results_path
                    experiment_data["sian_results.step_data"] = all_training_results["step_data"]
                    experiment_data["sian_results.epoch_data"] = all_training_results["epoch_data"]
                    experiment_data["SIAN_partial_training_time"] = partial_training_time
                    with open(saved_results_path, "w") as f:
                        json.dump(experiment_data, f, indent=4)
                    pass

                    
            metric_evaluation_time_taken += (time.time() - eval_start_time)

            if step >= STEP:
                k = EP-1
                break

        if TASK_TYPE == "regression":
            avg_epoch_loss = np.mean(all_trn_losses[k])
            if VERBOSE_TRAINING:
                print(f"Epoch {k} Average Training Loss: {avg_epoch_loss:.4f}")

        if PLOTTING_ALL_LOSSES_EVERY_EPOCH:
            plt.plot(1+np.arange(len(loss_list)),loss_list)
            if True:
                substeps = list(step_data.keys())
                trn_steps = [step_data[substep]['trn_metric'] for substep in substeps]
                val_steps = [step_data[substep]['val_metric'] for substep in substeps]
                plt.plot(1+np.array(substeps),val_steps)
                plt.plot(1+np.array(substeps),trn_steps)
            plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,2].reshape(-1)[:len(loss_list)])
            plt.plot(1+np.arange(len(loss_list)),all_losses[:(k+1),:,3].reshape(-1)[:len(loss_list)])
            plt.xscale('log')
            plt.ylim(0.0,1.3)
            # plt.ylim(-.01,3)
            plt.show()

            if False:
                plt.figure(figsize=(8,5.5))
                plt.plot(np.mean(all_losses[:(k+1),:,0],axis=1),c='C2',alpha=.4)
                plt.plot(all_val_losses[:k],c='C1')
                plt.plot(np.mean(all_losses[:(k+1),:,1],axis=1),c='C0',alpha=.4)
                plt.plot(np.mean(all_losses[:(k+1),:,2],axis=1),c='C3')
                plt.plot(all_trn_accs[:k],c='C0',linestyle='--')
                plt.plot(all_val_accs[:k],c='C1',linestyle='--')

                
                ###plt.plot(all_val_losses1[:k],c='C4')
                ###plt.plot(all_val_losses2[:k],c='C6')
                plt.plot([-50,EP+50],[0,0],c='k')
                #plt.legend(['total','val CE','trn CE','L1'])
                plt.legend(['total','val MSE','trn MSE','L1','','',])#'XOR_3','XOR_5'])
                
                #plt.ylim(-.01,2)
                plt.ylim(-.01,3)
                plt.xscale('log')
                plt.title("{ReLU MLP} Loss While Training")
                plt.show()
        
        if VERBOSE_TRAINING:
            print('epoch', str(k) + "'")
        net.eval()

        # if (k+1)%EVALUATE_EVERY_K_EPOCHS==0 or (k+1)==EP:
        if (k+1) in epochs_to_evaluate or (k+1)%EVALUATE_EVERY_K_EPOCHS==0 or (k+1)==EP:
            pass
            # eval_start_time = time.time()

            # evaluation = get_evaluation()
            # epoch_data[k+1] = evaluation


            # val_metric = evaluation['val_metric']
            # val_score = -val_metric if TASK_TYPE == TaskType.regression else val_metric

            # if val_score > best_val_score:
            #     # print(f"Val score: {val_score}, Best val score: {best_val_score}")
            #     best_val_score = val_score
            #     best_net = copy.deepcopy(net) #NOTE: slight concern with keeping two model copies on the GPU
            #     epochs_without_improvement = 0
            #     print(f"Epochs without improvement = {epochs_without_improvement}")
            # else:
            #     epochs_without_improvement += 1
            #     print(f"Epochs without improvement = {epochs_without_improvement}")

            # if VERBOSE_TRAINING:
            #     metric_name = "MSE" if TASK_TYPE == TaskType.regression else "ROC AUC"
            #     print(f'{metric_name} for train and val: {trn_metric:.4f}, {val_metric:.4f}')

            # if epochs_without_improvement >= early_stopping_patience:
            #     if VERBOSE_TRAINING:
            #         print(f"Early stopping triggered after {k + 1} epochs with no improvement in validation {metric_name}")
            #     break
            # metric_evaluation_time_taken += (time.time() - eval_start_time)
                
        epoch_time = time.time() - epoch_start_time
        if VERBOSE_TRAINING:
            print(f"--- {epoch_time:.3f} seconds in epoch ---")


        if PLOTTING_ALL_LOSSES_EVERY_EPOCH:
            val_loss_list.append(val_metric)

            # #JAM -- changing to this version for now
            pass
            # k=k+1
            # plt.figure(figsize=(10,7))
            # plt.plot(np.mean(all_losses[:k,:,0],axis=1))
            # plt.plot(val_loss_list)
            # plt.plot(np.mean(all_losses[:k,:,1],axis=1))
            # plt.plot(np.mean(all_losses[:k,:,2],axis=1))
            # plt.plot([-50,EP+50],[0,0],c='k')
            # # plt.legend(['trn loss','val','trn mse','trn l1'])
            # plt.legend(['trn loss','val mse','trn mse','trn l1'])
            # plt.xlim(-2,k+2)
            # plt.title("DNN/MLP Loss While Training")
            # plt.show()


    total_training_time = time.time() - full_training_start_time
    if VERBOSE_TRAINING:
        print('FULLY TRAINED USING', total_training_time, 'seconds')

    if MASKING_MODE: #early stopping from masking phase so 'best_net' is 'final_net'
        best_net = copy.deepcopy(net)

    if 'final_net' in results_to_save: 
        file_prefix = results_save_prefix
        if 'sian' in net_name:
            net.compress()
        final_model_path = file_prefix + net_name + '_final_net.pt'
        ### final_model_path = file_prefix + '_final_net.pt' #05/22/26 - windows file name sizes again
        print('final_model_path',final_model_path)
        torch.save(net,  final_model_path)
        # torch.save(net.state_dict(),  file_prefix + net_name + '_final_net.pt')
        if 'sian' in net_name:
            net.blocksparse()

    if 'best_net' in results_to_save: #NOTE: slight concern with keeping two model copies on the GPU
        file_prefix = results_save_prefix
        if 'sian' in net_name:
            best_net.compress()
        best_model_path = file_prefix + net_name + '_best_net.pt'
        torch.save(best_net, best_model_path)
        # torch.save(best_net.state_dict(),  file_prefix + net_name + '_best_net.pt')
        if 'sian' in net_name:
            best_net.blocksparse()

    all_training_results = {
        "step_data": step_data,
        "epoch_data": epoch_data,
        "total_training_time": total_training_time,

        "final_model_path" : final_model_path,
        "best_model_path" : best_model_path,

        "gradient_training_time_taken": gradient_training_time_taken,
        "metric_evaluation_time_taken": metric_evaluation_time_taken,
    }
    if training_args.return_settings.return_val_tensor:
        if val_tensor is None:
            val_tensor = dataset_object.pull_val_tensor(device)
        all_training_results["val_tensor"] = val_tensor
    if training_args.return_settings.return_val_output:
        val_output = dataset_object.pull_val_output(device)
        all_training_results["val_output"] = val_output
    if training_args.return_settings.return_final_net:
        all_training_results["final_net"] = net
    if training_args.return_settings.return_best_net:
        all_training_results["best_net"] = best_net
    return all_training_results











#used to discount the really large datasets (higgs, songyear, treecover) which have too many samples to evaluate each time
MAXIMUM_EVALUATION_STEPS = None
MAXIMUM_EVALUATION_STEPS = 1000

def evaluate_during_training_from_dataloader_REG(net, data_loader, masking_mode, mask_sampler=None, grouped_feat_dict=None):
    total_count = 0
    trn_metric = 0.0
    trn_y_true = []
    trn_y_score = []

    TOTAL_ROUNDS = 10
    if not masking_mode or mask_sampler is None:
        TOTAL_ROUNDS = 1
    
    if (masking_mode) and (mask_sampler is not None) and TRACK_VAL_SAMPLING:
        # print('ayaya')
        # grouped_feat_dict = dataset_object.get_grouped_feature_dict()
        # grouped_feat_dict = None
        
        D  = grouped_feat_dict['D']
        D0 = grouped_feat_dict['D0']
        device = data_loader.dataset.tensors[0].device
        all_subset_losses = torch.zeros((D+1, 2)).to(device)
        subset_indexer = torch.ones(D).float().to(device) #NOTE: "addmv_impl_cuda" not implemented for 'Long'

        FULL_OOM_SUBSET_INDEXING_TRACKING = False
        if FULL_OOM_SUBSET_INDEXING_TRACKING: #JAM @ 10:20pm on 12/01/2025
            eval_all_subset_losses = torch.zeros((2**D0,2)).to(device)

            all_subset_losses = torch.zeros((1, 2**D0, 2)).to(device)
            subset_indexer = torch.zeros(D).float().to(device)
            cum_ii = 0
            for i in range(D0):
                subset_indexer[cum_ii] = 2**i
                cum_ii += len(grouped_feat_dict[i])
            # print('subset_indexer',subset_indexer)
    

    with torch.no_grad():
        for _ in range(TOTAL_ROUNDS):
            for j, (x_batch, y_batch) in enumerate(data_loader):
                if j > MAXIMUM_EVALUATION_STEPS:
                    break
                new_count = x_batch.shape[0]
                if torch.any(torch.isnan(x_batch)) or torch.any(torch.isinf(x_batch)):
                    print(f"Warning: NaNs or infinities in train batch {j}")
                    continue
                if not masking_mode:
                    logits_trn = net(x_batch)
                else:
                    if mask_sampler is None:
                        logits_trn = net( (x_batch,torch.ones_like(x_batch)) )
                    else:
                        s_batch = mask_sampler.get_mask(x_batch)
                        logits_trn = net( (x_batch, s_batch) )
                logits_trn = logits_trn[0] + logits_trn[1]
                if LOGIT_CLAMPING:
                    logits_trn = torch.clamp(logits_trn, min=-100, max=100)

                new_trn_metric_ = ((y_batch[:, 0] - logits_trn[:, 0]) ** 2)
                new_trn_metric = torch.mean(new_trn_metric_).item()
                trn_metric = trn_metric + (new_trn_metric - trn_metric) * new_count / (total_count + new_count)
                total_count += new_count
                
                if (masking_mode) and (mask_sampler is not None) and TRACK_VAL_SAMPLING and FULL_OOM_SUBSET_INDEXING_TRACKING:
                    s_index = torch.matmul(s_batch.float(), subset_indexer).long()
                    # print('s_index',s_index)
                    eval_all_subset_losses[:, 0].index_put_((s_index,), new_trn_metric_.detach(), accumulate=True)
                    eval_all_subset_losses[:, 1].index_put_((s_index,), torch.ones_like(new_trn_metric_.detach()), accumulate=True)
                    trn_metric = eval_all_subset_losses

    return trn_metric
    # return eval_all_subset_losses

def evaluate_during_training_from_dataloader_BINARY_auc(net, data_loader, masking_mode, mask_sampler=None):
    total_count = 0
    trn_metric = 0.0
    trn_y_true = []
    trn_y_score = []
    with torch.no_grad():
        for j, (x_batch, y_batch) in enumerate(data_loader):
            if j > MAXIMUM_EVALUATION_STEPS:
                break
            new_count = x_batch.shape[0]
            if torch.any(torch.isnan(x_batch)) or torch.any(torch.isinf(x_batch)):
                print(f"Warning: NaNs or infinities in train batch {j}")
                continue
            if not masking_mode:
                logits_trn = net(x_batch)
            else:
                if mask_sampler is None:
                    logits_trn = net( (x_batch,torch.ones_like(x_batch)) )
                else:
                    s_batch = mask_sampler.get_mask(x_batch)
                    logits_trn = net( (x_batch, s_batch) )
            logits_trn = logits_trn[0] + logits_trn[1]
            if LOGIT_CLAMPING:
                logits_trn = torch.clamp(logits_trn, min=-100, max=100)

            probs = torch.sigmoid(logits_trn[:, 0])
            trn_y_true.extend(y_batch[:, 0].cpu().numpy())
            trn_y_score.extend(probs.cpu().numpy())
            total_count += new_count

    trn_y_true = np.array(trn_y_true)
    trn_y_score = np.array(trn_y_score)
    try:
        trn_metric = roc_auc_score(trn_y_true, trn_y_score) if len(np.unique(trn_y_true)) > 1 else 0.0
    except ValueError as e:
        print(f"ROC AUC calculation failed in train: {e}. Setting metric to 0.0.")
        trn_metric = 0.0
    return trn_metric

def evaluate_during_training_from_dataloader_MULTI_auc(net, data_loader, masking_mode, n_classes=None, mask_sampler=None):
    total_count = 0
    trn_metric = 0.0
    trn_y_true = []
    trn_y_score = []
    with torch.no_grad():
        for j, (x_batch, y_batch) in enumerate(data_loader):
            if j > MAXIMUM_EVALUATION_STEPS:
                break
            new_count = x_batch.shape[0]
            if torch.any(torch.isnan(x_batch)) or torch.any(torch.isinf(x_batch)):
                print(f"Warning: NaNs or infinities in train batch {j}")
                continue
            if not masking_mode:
                logits_trn = net(x_batch)
            else:
                if mask_sampler is None:
                    logits_trn = net( (x_batch,torch.ones_like(x_batch)) )
                else:
                    s_batch = mask_sampler.get_mask(x_batch)
                    logits_trn = net( (x_batch, s_batch) )
            logits_trn = logits_trn[0] + logits_trn[1]
            if LOGIT_CLAMPING:
                logits_trn = torch.clamp(logits_trn, min=-100, max=100)

            logprobs = torch.nn.LogSoftmax(dim=-1)(logits_trn)
            probs = torch.exp(logprobs)
            trn_y_true.extend(y_batch.cpu().numpy())
            trn_y_score.extend(probs.cpu().numpy())
            total_count += new_count

    trn_y_true = np.array(trn_y_true)
    trn_y_score = np.array(trn_y_score)
    try:
        all_classes = np.arange(n_classes)
        if trn_y_score.shape[1] != n_classes:
            print(f"Warning: trn_y_score has {trn_y_score.shape[1]} columns, expected {n_classes}. Adjusting...")
            trn_y_score = np.pad(trn_y_score, ((0, 0), (0, n_classes - trn_y_score.shape[1])), mode='constant', constant_values=1e-10)
            trn_y_score = trn_y_score / trn_y_score.sum(axis=1, keepdims=True)
        trn_metric = roc_auc_score(trn_y_true, trn_y_score, multi_class='ovr') if np.sum(trn_y_true, axis=0).min() > 0 else 0.0
    except ValueError as e:
        print(f"ROC AUC calculation failed in train: {e}. Setting metric to 0.0.")
        trn_metric = 0.0
    return trn_metric











def evaluate_model_on_test_set(dataset_object, net, training_args, is_masked_model=False):
    normalize_XY = training_args.normalize_XY
    _, _, tstX, tstY = dataset_object.pull_data(normalize_XY[0], normalize_XY[1])
    device = training_args.device
    n_classes = dataset_object.get_C()
    task_type = dataset_object.get_task_type()

    test_tensor = torch.from_numpy(tstX).float().to(device)

    if is_masked_model:
        identity_mask = torch.ones_like(test_tensor).to(device)
        test_input = (test_tensor, identity_mask)
    else:
        test_input = test_tensor

    net = net.to(device)
    net.eval()

    with torch.no_grad():
        logits_test = net(test_input)
        logits_test = logits_test[0] + logits_test[1]
        LOGIT_CLAMPING = training_args.LOGIT_CLAMPING
        if LOGIT_CLAMPING:
            logits_test = torch.clamp(logits_test, min=-100, max=100)

    if task_type == TaskType.regression:
        preds = logits_test[:, 0].cpu().numpy() if logits_test.ndim > 1 else logits_test.cpu().numpy()
        y_true = tstY[:, 0] if tstY.ndim > 1 else tstY
        test_mse = np.mean((y_true - preds) ** 2)
        print(f"Test MSE: {test_mse:.4f}")
        return {"mse": test_mse}
    else:
        if task_type == TaskType.binary:
            y_score = torch.sigmoid(logits_test[:, 0]).cpu().numpy()
            y_true = tstY[:, 0] if tstY.ndim > 1 else tstY
            y_pred = (y_score >= 0.5).astype(int)
        elif task_type == TaskType.multiclass:
            logprobs = torch.nn.LogSoftmax(dim=-1)(logits_test)
            probs = torch.exp(logprobs)
            y_score = probs.cpu().numpy()
            y_true = tstY  # One-hot encoded: (N, n_classes)
            y_pred = np.argmax(y_score, axis=1)

        test_results = {}
        # Accuracy, Precision, Recall, F1
        y_true_labels = np.argmax(y_true, axis=1) if task_type == TaskType.multiclass else y_true
        test_results["accuracy"] = accuracy_score(y_true_labels, y_pred)
        test_results["precision"] = precision_score(y_true_labels, y_pred, average='macro', zero_division=0)
        test_results["recall"] = recall_score(y_true_labels, y_pred, average='macro', zero_division=0)
        test_results["f1"] = f1_score(y_true_labels, y_pred, average='macro', zero_division=0)

        try: # ROC-AUC and AUPRC
            if task_type == TaskType.binary:
                test_results["roc_auc"] = roc_auc_score(y_true, y_score)
                test_results["auprc"] = average_precision_score(y_true, y_score)
            elif task_type == TaskType.multiclass:
                if y_score.shape[1] != n_classes:
                    print(f"Warning: y_score has {y_score.shape[1]} columns, expected {n_classes}. Adjusting...")
                    y_score = np.pad(y_score, ((0, 0), (0, n_classes - y_score.shape[1])), mode='constant', constant_values=1e-10)
                    y_score = y_score / y_score.sum(axis=1, keepdims=True)
                test_results["roc_auc"] = roc_auc_score(y_true, y_score, multi_class='ovr') if np.sum(y_true, axis=0).min() > 0 else 0.0
                test_results["auprc"] = average_precision_score(y_true, y_score, average="macro")
        except ValueError as e:
            print(f"Warning: AUC metrics could not be computed. Reason: {e}")
            test_results["roc_auc"] = 0.0
            test_results["auprc"] = 0.0

        print(f"Test Accuracy:  {test_results['accuracy']:.4f}")
        print(f"Test Precision: {test_results['precision']:.4f}")
        print(f"Test Recall:    {test_results['recall']:.4f}")
        print(f"Test F1 Score:  {test_results['f1']:.4f}")
        print(f"Test ROC AUC:   {test_results['roc_auc']:.4f}")
        print(f"Test AUPRC:     {test_results['auprc']:.4f}")

        return test_results


















