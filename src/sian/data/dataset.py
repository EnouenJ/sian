import numpy as np
import torch



import os




from torch.utils.data import TensorDataset, DataLoader



from sian.data.data_loader import loadDataset #NOTE: NEED TO DEPRICATE
# from sian.data.data_loader import loadLabels
from sian.data.data_loader import loadHeader


from sian.data.data_loader import preprocess_dataset
from sian.data.data_loader import load_from_preprocessed
from sian.data.data_loader import load_gen_from_preprocessed

from sklearn.preprocessing import label_binarize

from .data_loader import get_readables_from_full_readables #NOTE: needed only for generative for some reason right now




default_preproc_owner_dict = {
    "UCI_275_bike_sharing_dataset" : "SIAN2022",
    "UCI_186_wine_quality" : "SIAN2022",
    "UCI_374_appliances_energy_prediction" : "SIAN2022",
    "otherSource_cal_housing" : "SIAN2022",
    "UCI_203_yearpredictionMSD" : "SIAN2022",
    "UCI_280_higgs_boson_dataset" : "SIAN2022",
    
    "UCI_2_adults_dataset" : "InstaSHAP2025",
    "UCI_31_tree_cover_type_dataset" : "InstaSHAP2025",
    
    "UCI_1_abalone_dataset" : "FIS2025",
    "Kaggle_blastchar_telco_customer_churn" : "FIS2025",
    "UCI_332_online_news_popularity" : "FIS2025",
    "Kaggle_mlgulb_credit_card_fraud_dataset" : "FIS2025",
    "Kaggle_ishadss_eucalyptus_dataset" : "FIS2025",
    "otherSource_Microsoft_search_queries" : "FIS2025",
    "UCI_171_madelon_dataset" : "FIS2025",
    "UCI_572_taiwanese_bankruptcy_dataset" : "FIS2025",
    
    "otherSource_MNIST" : "FIS2025",
}

class npBasicScaler():
    def __init__(self, trnvalX=None):
        if trnvalX is not None:
            self.fit(trnvalX)

    def fit(self, trnvalX):
        self.mean = np.mean(trnvalX,axis=0)
        self.std  = np.sqrt(np.var(trnvalX,axis=0))

    def protect(self, protected_dimensions):
        self.mean[protected_dimensions] = 0.0
        self.std[protected_dimensions]  = 1.0

    def normalize(self, X):
        return (X-self.mean[None])/self.std[None]
        
    def denormalize(self, X):
        return X*self.std[None]+self.mean[None]

    def __call__(self, X):
        return self.normalize(X)

class npIdentityScaler(): #NOTE: probably worth considering how to do unbalanced classification with similar techniques (beyond identity)
    def __init__(self, trnvalX=None):
        if trnvalX is not None:
            self.fit(trnvalX)

    def fit(self, trnvalX):
        pass

    def normalize(self, X):
        return X
        
    def denormalize(self, X):
        return X
        
    def __call__(self, X):
        return self.normalize(X)
        
        
def get_scale_protected_dimensions_helper(full_readable_labels):
    D0 = full_readable_labels["D0"]
    protected_dimensions = []
    for d in range(D0):
        label_info_d = full_readable_labels[d]
        encoding_d = label_info_d['encoding']
        if "cts." in encoding_d or encoding_d=="disc.ordinal":
            protected_d = False
        elif  encoding_d=="disc.onehot":
            protected_d = True
        else:
            raise Exception(f"RENORMALIZING NOT SUPPORTED, encoding={encoding_d} is not supported")
        if protected_d:
            startdim = label_info_d['startdim']
            numdims = label_info_d['numdims']
            protected_dimensions.extend(list(range(startdim,startdim+numdims)))
    return protected_dimensions





def seed_worker(worker_id): #this is needed for multiGPU I think
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Final_TabularDataset:
    def __init__(self, dataset_str, preproc_owner=None, type="realworld.tabular",
                 load_dataset_path=None, save_dataset_path=None):
        SAVING_PREPROCESSED_VERSION = True  # TODO: move as potential kwargs
        OVERWRITE_EXISTING_PROCESSED = False

        if preproc_owner is None:
            #NOTE: rahil, I think this was only working because you have manually been passing this instead of None? (I think your script builder was doing this)
            if dataset_str in default_preproc_owner_dict:
                preproc_owner = default_preproc_owner_dict[dataset_str]
            print(f"preproc_owner=None, so using default preproc_owner={preproc_owner}")
        # print('Final_TabularDataset().save_dataset_path',save_dataset_path)

        already_preprocessed = False
        header_dict = loadHeader(save_dataset_path)
        print('header_dict',header_dict)
        if header_dict is not None:
            if header_dict["dataset_id"]==dataset_str and header_dict["preproc_owner"]==preproc_owner:
                already_preprocessed = True
                if OVERWRITE_EXISTING_PROCESSED:
                    already_preprocessed = False #specifically overwriting even though existing
            else:
                if OVERWRITE_EXISTING_PROCESSED:
                    pass #just overwrite implicitly
                else:
                    raise Exception(f"existing preprocessed dataset at save location  &&  OVERWRITE_EXISTING_PROCESSED={OVERWRITE_EXISTING_PROCESSED}")

        if not already_preprocessed:
            print(f"Creating new dataset at {save_dataset_path}")
            #maybe have this part tell us if it is an "X->Y" dataset or an "X only" dataset
            header_dict, XY_tuple, label_stuff_dict = \
                preprocess_dataset(dataset_str, preproc_owner=preproc_owner,
                                   load_dataset_path=load_dataset_path, save_dataset_path=save_dataset_path,
                                   SAVING_PREPROCESSED_VERSION=SAVING_PREPROCESSED_VERSION)
        else:
            print(f"Loading existing dataset from {save_dataset_path}")
            header_dict, XY_tuple, label_stuff_dict = load_from_preprocessed(save_dataset_path)
        
        self.header_dict = header_dict
        self.label_stuff_dict = label_stuff_dict

        # Load and preprocess data
        trnvalX, trnvalY, tstX, tstY = XY_tuple
        task_type = self.get_task_type()

        # Handle label shapes based on task type
        if task_type == "multiclass_classification" and len(trnvalY.shape) == 1:
            # Convert integer labels to one-hot for multiclass classification
            num_classes = len(np.unique(trnvalY))
            trnvalY = np.eye(num_classes)[trnvalY.astype(int)]
            tstY = np.eye(num_classes)[tstY.astype(int)]
        elif len(trnvalY.shape) == 1:
            # For regression or binary classification, add a dimension if needed
            trnvalY = trnvalY[:, None]
            tstY = tstY[:, None]

        self.trnvalX = trnvalX
        self.trnvalY = trnvalY
        self.tstX = tstX
        self.tstY = tstY

        self.trnX, self.valX, self.trnY, self.valY = None, None, None, None

        del XY_tuple
        del trnvalX, trnvalY, tstX, tstY

        # Prepare for renormalization
        full_readable_labels = self.get_full_readable_labels()
        protected_dimensions = get_scale_protected_dimensions_helper(full_readable_labels)
        self.x_scaler = npBasicScaler(self.trnvalX)
        self.x_scaler.protect(protected_dimensions)
        
        if task_type == "regression":
            self.y_scaler = npBasicScaler(self.trnvalY)
        elif task_type in ["binary_classification", "multiclass_classification"]:
            self.y_scaler = npIdentityScaler()
        else:
            raise Exception(f"RENORMALIZING NOT SUPPORTED, task_type={task_type} is not supported")

    def get_D(self):
        return self.trnvalX.shape[1]
        
    def get_C(self):
        task_type = self.get_task_type()
        if task_type == "multiclass_classification":
            # Return the number of unique classes based on the unique values in trnvalY
            if len(self.trnvalY.shape) == 2 and self.trnvalY.shape[1] > 1:
                return self.trnvalY.shape[1]  # One-hot encoded
            else:
                return len(np.unique(self.trnvalY))  # Integer labels
        else:
            # For regression or binary classification, use the second dimension
            return self.trnvalY.shape[1]

    def get_N(self):
        return self.trnvalX.shape[0]
        
    def get_base_D(self):
        full_readable_labels = self.label_stuff_dict["full_readable_labels"]
        return full_readable_labels["D0"]
        
    
    def get_dataset_id(self):
        return self.header_dict["dataset_id"]
        
    def get_readable_labels(self):
        return self.label_stuff_dict["readable_labels"]
        
    def get_full_readable_labels(self):
        return self.label_stuff_dict["full_readable_labels"]

    def get_grouped_feature_dict(self):
        grouped_features_dict = {}
        full_readable_labels = self.get_full_readable_labels()
        D0 = full_readable_labels["D0"]
        D  = self.get_D()
        grouped_features_dict["D0"] = D0
        grouped_features_dict["D"]  = D
        C  = self.get_C()
        grouped_features_dict["C"]  = C
        for d in range(D0):
            feat_d_info = full_readable_labels[d]
            startdim = feat_d_info['startdim']
            numdims  = feat_d_info['numdims']
            grouped_features_dict[d] = list(range(startdim,startdim+numdims))
        return grouped_features_dict
    
    def get_task_type(self):
        full_readable_labels = self.get_full_readable_labels()
        return full_readable_labels["task_type"]

    def shuffle_and_split_trnval(self, trnval_shuffle_seed=None, trnval_split_percentage=0.7, trnval_reduc_percentage=1.0):
        if trnval_shuffle_seed is None:
            np.random.seed(None)
            self.trnval_shuffle_seed = np.random.randint()
        else:
            self.trnval_shuffle_seed = trnval_shuffle_seed
        np.random.seed(self.trnval_shuffle_seed)
        print('trnval_shuffle_seed',trnval_shuffle_seed)
        print('self.trnval_shuffle_seed',self.trnval_shuffle_seed)

        M_NUM = int(self.trnvalX.shape[0])
        M_TRNVAL_N = int(M_NUM*trnval_reduc_percentage)
        M_TRN_N = int(M_TRNVAL_N * trnval_split_percentage)
        rand_indices = np.random.permutation(M_NUM)
        print('M_NUM',M_NUM,'M_TRNVAL_N',M_TRNVAL_N,'M_TRN_N',M_TRN_N)
        self.trnX, self.valX = self.trnvalX[rand_indices[:M_TRN_N]], self.trnvalX[rand_indices[M_TRN_N:M_TRNVAL_N]]
        self.trnY, self.valY = self.trnvalY[rand_indices[:M_TRN_N]], self.trnvalY[rand_indices[M_TRN_N:M_TRNVAL_N]]
        pass

    def pull_data(self, renormalizeX=True, renormalizeY=True):
        # print('pull_data()', renormalizeX,renormalizeY)
        x_map = self.x_scaler if renormalizeX else lambda x : x
        y_map = self.y_scaler if renormalizeY else lambda y : y
        return (x_map(self.trnvalX),y_map(self.trnvalY),x_map(self.tstX),y_map(self.tstY))
    
    def pull_trnval_data(self, renormalizeX=True, renormalizeY=True):
        # print('pull_trnval_data()', renormalizeX,renormalizeY)
        x_map = self.x_scaler if renormalizeX else lambda x : x
        y_map = self.y_scaler if renormalizeY else lambda y : y
        return (x_map(self.trnX),y_map(self.trnY),x_map(self.valX),y_map(self.valY))

    def pull_trnval_loaders(self, device, BS, batch_shuffling_seed=None, renormalizeX=True, renormalizeY=True):
        trnX, trnY, valX, valY = self.pull_trnval_data(renormalizeX, renormalizeY)
        print('trnX', 'trnY', trnX.shape, trnY.shape)
        if np.any(np.isnan(trnX)) or np.any(np.isinf(trnX)):
            print("Warning: NaNs or infinities detected in trnX")
        if np.any(np.isnan(valX)) or np.any(np.isinf(valX)):
            print("Warning: NaNs or infinities detected in valX")
            
        trn_data = TensorDataset(torch.from_numpy(trnX).float().to(device),
                                torch.from_numpy(trnY).float().to(device))
        val_data = TensorDataset(torch.from_numpy(valX).float().to(device),
                                torch.from_numpy(valY).float().to(device))

        datashuffle_rng = torch.Generator()
        if batch_shuffling_seed is None:
            batch_shuffling_seed = int(np.random.randint(1))
        datashuffle_rng.manual_seed(batch_shuffling_seed)
        usually_drop_last = lambda data_tensor : (len(data_tensor) > BS) # Updated from usually_drop_last = lambda data_tensor : (data_tensor.shape[0] > BS) which causes an AttributeError - Rahil
        trn_loader = DataLoader(dataset=trn_data, batch_size=BS, shuffle=True, drop_last=usually_drop_last(trn_data), generator=datashuffle_rng, worker_init_fn=seed_worker)
        val_loader = DataLoader(dataset=val_data, batch_size=BS, shuffle=True, drop_last=usually_drop_last(val_data), generator=datashuffle_rng, worker_init_fn=seed_worker)
        return trn_loader, val_loader

    def pull_tst_loaders(self, device, BS, batch_shuffling_seed=None, renormalizeX=True, renormalizeY=True):
        _, _, tstX, tstY = self.pull_data(renormalizeX, renormalizeY)
        
        tst_data = TensorDataset(torch.from_numpy(tstX).float().to(device),
                                 torch.from_numpy(tstY).float().to(device))

        datashuffle_rng = torch.Generator()
        if batch_shuffling_seed is None:
            batch_shuffling_seed = int(np.random.randint(1))
        datashuffle_rng.manual_seed(batch_shuffling_seed)
        usually_drop_last = lambda data_tensor : (len(data_tensor) > BS) # Updated from usually_drop_last = lambda data_tensor : (data_tensor.shape[0] > BS) which causes an AttributeError - Rahil
        tst_loader = DataLoader(dataset=tst_data, batch_size=BS, shuffle=True, drop_last=usually_drop_last(tst_data), generator=datashuffle_rng, worker_init_fn=seed_worker)
        return tst_loader

    def pull_trn_tensor(self, device):
        return torch.from_numpy(self.trnX).float().to(device)
    def pull_val_tensor(self, device):
        return torch.from_numpy(self.valX).float().to(device)
    def pull_tst_tensor(self, device):
        return torch.from_numpy(self.tstX).float().to(device)
    def pull_trn_output(self, device):
        return torch.from_numpy(self.trnY).float().to(device)
    def pull_val_output(self, device):
        return torch.from_numpy(self.valY).float().to(device)
    def pull_tst_output(self, device):
        return torch.from_numpy(self.tstY).float().to(device)

        








class Final_TabularGenerativeDataset():
    def __init__(self, dataset_str, preproc_owner=None, type="realworld.tabular",
                 load_dataset_path=None, save_dataset_path=None):
        SAVING_PREPROCESSED_VERSION = True  # TODO: move as potential kwargs
        OVERWRITE_EXISTING_PROCESSED = False

        if preproc_owner is None:
            if dataset_str in default_preproc_owner_dict:
                preproc_owner = default_preproc_owner_dict[dataset_str]
            print(f"preproc_owner=None, so using default preproc_owner={preproc_owner}")

        # already_preprocessed = False
        # header_dict = loadHeader(save_dataset_path)
        # print('header_dict',header_dict)
        # if header_dict is not None:
        #     if header_dict["dataset_id"]==dataset_str and header_dict["preproc_owner"]==preproc_owner:
        #         already_preprocessed = True
        #         if OVERWRITE_EXISTING_PROCESSED:
        #             already_preprocessed = False #SPECIFICALLY OVERWRITING
        #     else:
        #         if OVERWRITE_EXISTING_PROCESSED:
        #             pass #just overwrite implicitly
        #         else:
        #             raise Exception(f"existing preprocessed dataset at save location  &&  OVERWRITE_EXISTING_PROCESSED={OVERWRITE_EXISTING_PROCESSED}")
        already_preprocessed = True #TODO: fix gen later

        if not already_preprocessed:
            print("PROCESSING AND LOADING")
            raise Exception("NO GEN DATASETS LIKE THIS YET")
            #maybe have this part tell us if it is an "X->Y" dataset or an "X only" dataset
            # header_dict, XY_tuple, label_stuff_dict = \
            # preprocess_dataset(dataset_str, preproc_owner=preproc_owner,
            #                    load_dataset_path=load_dataset_path, save_dataset_path=save_dataset_path,
            #                    SAVING_PREPROCESSED_VERSION=SAVING_PREPROCESSED_VERSION)
        else:
            print("LOADING FROM EXISTING")
            # header_dict, XY_tuple, label_stuff_dict = load_from_preprocessed(save_dataset_path)
            header_dict, XY_tuple, label_stuff_dict = load_gen_from_preprocessed(save_dataset_path)
        
        self.base_header_dict = header_dict
        self.base_label_stuff_dict = label_stuff_dict
        self.label_stuff_dict = None

        # Load and preprocess data -- NOTE: need to probably add "task_type" for generative
        trnvalX, tstX = XY_tuple
        self._trnvalX = trnvalX
        self._tstX = tstX

        self.trnvalX, self.trnvalY, self.tstX, self.tstY = None, None, None, None
        self.trnX, self.valX, self.trnY, self.valY = None, None, None, None

        del XY_tuple
        del trnvalX, tstX

        # TODO TODO -- need to consider renormalization for generative data

        # # Prepare for renormalization
        # full_readable_labels = self.get_full_readable_labels()
        # protected_dimensions = get_scale_protected_dimensions_helper(full_readable_labels)
        # self.x_scaler = npBasicScaler(self.trnvalX)
        # self.x_scaler.protect(protected_dimensions)
        
        # if task_type == "regression":
        #     self.y_scaler = npBasicScaler(self.trnvalY)
        # elif task_type in ["binary_classification", "multiclass_classification"]:
        #     self.y_scaler = npIdentityScaler()
        # else:
        #     raise Exception(f"RENORMALIZING NOT SUPPORTED, task_type={task_type} is not supported")

    def get_D(self):
        if self.trnvalX is not None:
            return self.trnvalX.shape[1]
        else:
            raise Exception("This is a generative dataset, there is no specific Y target yet")
        
    def get_base_D(self):
        # return self._trnvalX.shape[1]
        base_full_readable_labels = self.base_label_stuff_dict["full_readable_labels"]
        return base_full_readable_labels["D0"]
        
    def get_C(self):
        if self.trnvalX is not None:
            return self.trnvalY.shape[1]
        else:
            raise Exception("This is a generative dataset, there is no specific Y target yet")



    #TODO TODO TODO TODO TODO TODO TODO TODO
    def prep_directional_prediction(self, target_feature_index=0):
        D = self.get_base_D()
        base_full_readable_labels = self.base_label_stuff_dict["full_readable_labels"]
        new_full_readable_labels = {}
        new_full_readable_labels["D0"] = D
        new_full_readable_labels["task_type"] = None
        currdim = 0
        dd=0
        trnvalX_list = []
        tstX_list = []
        self.trnvalX = np.copy(self._trnvalX)
        self.tstX = np.copy(self._tstX)
        for d in range(D):
            full_info_d = base_full_readable_labels[d]
            startdim_d = full_info_d['startdim']
            dim_d = full_info_d['numdims']
            # print(full_info_d)

            if d==target_feature_index:
                new_full_readable_labels[-1] = full_info_d
                if full_info_d['encoding'] == 'cts.raw':
                    new_full_readable_labels["task_type"] = "regression"
                elif full_info_d['encoding'] == 'disc.onehot':
                    new_full_readable_labels["task_type"] = "multiclass_classification" #NOTE: update this later
                else:
                    raise NotImplementedError(f"\'task_type\' not found for output variable with \'encoding\'={full_info_d['encoding']}")
                self.trnvalY = (  self._trnvalX[:,startdim_d:startdim_d+dim_d]  )
                self.tstY = (  self._tstX[:,startdim_d:startdim_d+dim_d]  )
                # print(np.mean(self.trnvalY,axis=0)) #debug for potential shallow copy issues
                self.trnvalX[:,startdim_d:startdim_d+dim_d] = 0 
                self.tstX[:,startdim_d:startdim_d+dim_d] = 0
                # print(np.mean(self.trnvalY,axis=0)) #debug for potential shallow copy issues

            #else: #TODO TODO DOING THIS ALWAYS FOR NOW TO FOLLOW MY PREVIOUS CODE, BUT MAYBE LONG TERM IT IS A BIT ODD  (sepcifically about this zeroing out thing)
            if True:
                new_full_readable_labels[dd] = full_info_d
                dd += 1
                currdim += dim_d

        self.label_stuff_dict = {
            "readable_labels" : get_readables_from_full_readables(new_full_readable_labels),
            "full_readable_labels" : new_full_readable_labels,
        }



    def get_dataset_id(self):
        return self.header_dict["dataset_id"]
        
    def get_readable_labels(self):
        return self.label_stuff_dict["readable_labels"]
        
    def get_full_readable_labels(self):
        return self.label_stuff_dict["full_readable_labels"]
    
    def get_base_full_readable_labels(self):
        return self.base_label_stuff_dict["full_readable_labels"]

    def get_grouped_feature_dict(self):
        grouped_features_dict = {}
        full_readable_labels = self.get_full_readable_labels()
        D0 = full_readable_labels["D0"]
        D  = self.get_D()
        grouped_features_dict["D0"] = D0
        grouped_features_dict["D"]  = D
        C  = self.get_C()
        grouped_features_dict["C"]  = C
        for d in range(D0):
            feat_d_info = full_readable_labels[d]
            startdim = feat_d_info['startdim']
            numdims  = feat_d_info['numdims']
            grouped_features_dict[d] = list(range(startdim,startdim+numdims))
        return grouped_features_dict
    
    def get_task_type(self):
        full_readable_labels = self.get_full_readable_labels()
        return full_readable_labels["task_type"]

    def shuffle_and_split_trnval(self, trnval_shuffle_seed=None, trnval_split_percentage=0.7):
        if trnval_shuffle_seed is None:
            np.random.seed(None)
            self.trnval_shuffle_seed = np.random.randint()
        else:
            self.trnval_shuffle_seed = trnval_shuffle_seed
        np.random.seed(self.trnval_shuffle_seed)
        print('trnval_shuffle_seed',trnval_shuffle_seed)
        print('self.trnval_shuffle_seed',self.trnval_shuffle_seed)

        M_NUM = self.trnvalX.shape[0]
        rand_indices = np.random.permutation(M_NUM)
        M_TRN_NUM = int(M_NUM * trnval_split_percentage)
        self.trnX, self.valX = self.trnvalX[rand_indices[:M_TRN_NUM]], self.trnvalX[rand_indices[M_TRN_NUM:]]
        self.trnY, self.valY = self.trnvalY[rand_indices[:M_TRN_NUM]], self.trnvalY[rand_indices[M_TRN_NUM:]]
        pass

    def pull_gen_data(self): #NOTE: fix this later
        return (self._trnvalX,self._tstX) #TODO: adding renormalize to generative X as well


    def pull_data(self, renormalizeXandY=True):
        return (self.trnvalX,self.trnvalY,self.tstX,self.tstY)
        # if not renormalizeXandY:
        #     return (self.trnvalX,self.trnvalY,self.tstX,self.tstY)
        # else:
        #     return (self.x_scaler(self.trnvalX),self.y_scaler(self.trnvalY),self.x_scaler(self.tstX),self.y_scaler(self.tstY))
    
    def pull_trnval_data(self, renormalizeXandY=True):
        return (self.trnX,self.trnY,self.valX,self.valY)
        # if not renormalizeXandY:
        #     return (self.trnX,self.trnY,self.valX,self.valY)
        # else:
        #     return (self.x_scaler(self.trnX),self.y_scaler(self.trnY),self.x_scaler(self.valX),self.y_scaler(self.valY))

    #TODO:
    '''
    directly implement the trnval loader right here, no?

    '''
