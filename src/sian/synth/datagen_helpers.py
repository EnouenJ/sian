
import os
import numpy as np

from sian.utils import gettimestamp
from sian.data import final_save_header, final_save_gen_dataset, final_save_labels
from sian.data import get_readables_from_full_readables




def save_synthetic_as_official_dataset(save_dataset_path, synth_dataset_id, X, dgp_x):
    
    datetimestr = gettimestamp()
    if not os.path.exists(save_dataset_path):
        os.mkdir(save_dataset_path)

    fake_header_dict = {
        "Preprocessed Datetime" : datetimestr, 
        "dataset_id" : synth_dataset_id,
        "preproc_owner" : None,

        "load_dataset_path" : None,
        "save_dataset_path" : save_dataset_path,
    }
    
    full_readable_labels = dgp_x.return_full_readable_labels()
    readable_labels = get_readables_from_full_readables(full_readable_labels)
    label_dict = {
        "readable_labels" : readable_labels,
        "full_readable_labels" : full_readable_labels,
    }
    
    trainval_portion, is_test_split_shuffled, shuffle_test_split_seed = 0.5, False, None
    fullX=np.concatenate([X,np.zeros_like(X)])
    final_save_header(save_dataset_path, fake_header_dict)
    trnvalX,tstX = final_save_gen_dataset(save_dataset_path, fullX, trainval_portion, is_test_split_shuffled, shuffle_test_split_seed)
    final_save_labels(save_dataset_path, label_dict)






