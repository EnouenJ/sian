import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import csv
import json

from sian.utils import gettimestamp
import warnings


# def final_save_dataset(path, X, Y, split=None)
def save_full_dataset(path, X, Y):
    np.save(path+'fullX.npy', X)
    np.save(path+'fullY.npy', Y)

def load_full_dataset(path):
    fullX = np.load(path+'fullX.npy')
    fullY = np.load(path+'fullY.npy')
    return (fullX,fullY)

# splits and saves
def final_save_dataset(path, X, Y, trainval_portion, is_shuffled, shuffle_seed, SAVING=True):
    if shuffle_seed is None and is_shuffled is True:
        warnings.warn("potentially unreproducible shuffling behavior")
    
    NUM = X.shape[0]
    TRN_NUM = int(trainval_portion*NUM)
    TST_NUM = NUM - TRN_NUM

    if not is_shuffled:
        trnvalX = (X[:TRN_NUM])
        trnvalY = (Y[:TRN_NUM])
        tstX = (X[-TST_NUM:])
        tstY = (Y[-TST_NUM:])
    else:
        np.random.seed(shuffle_seed)
        perm = np.random.permutation(NUM)
        trnvalX = (X[perm[:TRN_NUM]])
        trnvalY = (Y[perm[:TRN_NUM]])
        tstX = (X[perm[-TST_NUM:]])
        tstY = (Y[perm[-TST_NUM:]])
        
    if not os.path.exists(path):
        os.mkdir(path)
    if SAVING:
        np.save(path+'trnvalX.npy',trnvalX)
        np.save(path+'trnvalY.npy',trnvalY)
        np.save(path+'tstX.npy',tstX)
        np.save(path+'tstY.npy',tstY)
    return trnvalX,trnvalY,tstX,tstY

def final_load_dataset(path):
    trnvalX = np.load(path+'trnvalX.npy')
    trnvalY = np.load(path+'trnvalY.npy')
    tstX = np.load(path+'tstX.npy')
    tstY = np.load(path+'tstY.npy')
    return trnvalX,trnvalY,tstX,tstY


def final_save_labels(path, data_labels_dict):
    with open(path+'data_labels.json', 'w', encoding='utf-8') as f:
        json.dump(data_labels_dict, f, ensure_ascii=False, indent=4)

def final_load_labels(path):
    labels_path = path+'data_labels.json'
    if not os.path.exists(labels_path):
        return None,None
    else:
        with open(labels_path,'r') as f:
            labels_dict = json.load(f)
    readable_labels = convert_dict_keys_to_int(labels_dict["readable_labels"])
    full_readable_labels = convert_dict_keys_to_int(labels_dict["full_readable_labels"])

    if True:
        data_labels_dict = {
            "readable_labels" : readable_labels,
            "full_readable_labels" : full_readable_labels,
        }
        return data_labels_dict


def final_save_header(path, header_dict):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(path, exist_ok=True)
    with open(path+'data_header.json', 'w', encoding='utf-8') as f:
        json.dump(header_dict, f, ensure_ascii=False, indent=4)

def final_load_header(path):
    header_path = path+'data_header.json'
    if not os.path.exists(header_path):
        return None
    else:
        with open(header_path,'r') as f:
            header_dict = json.load(f)
    return header_dict






def get_readables_from_full_readables(full_readable_labels):
    D0 = full_readable_labels["D0"]
    readable_labels = {}
    if -1 in full_readable_labels:
        readable_labels[-1] = full_readable_labels[-1]['label']
    for d in range(D0):
        readable_labels[d] = full_readable_labels[d]['label']
    return readable_labels










def saveDataset(path,X,Y,split=None):
    if split is None:
        split = (.8,.2)
    NUM = X.shape[0]
    TRN_NUM = int(split[0]*NUM)
    TST_NUM = NUM - TRN_NUM
    
    trnX = (X[:TRN_NUM])
    trnY = (Y[:TRN_NUM])
    tstX = (X[-TST_NUM:])
    tstY = (Y[-TST_NUM:])
        
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(path+'trnvalX.npy',trnX)
    np.save(path+'trnvalY.npy',trnY)
    np.save(path+'tstX.npy',tstX)
    np.save(path+'tstY.npy',tstY)
    
def loadDataset(path):
    trnX = np.load(path+'trnvalX.npy')
    trnY = np.load(path+'trnvalY.npy')
    tstX = np.load(path+'tstX.npy')
    tstY = np.load(path+'tstY.npy')
    return trnX,trnY,tstX,tstY


#TODO: currently has no shuffling (needs a seed); also need to integrate with v1 differences
def shuffleAndSaveDataset_v2(path,X,Y):
    # split = (.7,.1,.1,.1)
    split = (.6,.1,.1,.2)
    NUM = X.shape[0]
    TRN_NUM = int(split[0]*NUM)
    VAL_NUM = int(split[1]*NUM)
    VXL_NUM = int(split[2]*NUM)
    TST_NUM = NUM - TRN_NUM - VAL_NUM - VXL_NUM
    print(TRN_NUM,VAL_NUM,VXL_NUM,TST_NUM)

    trnX = (X[:TRN_NUM])
    trnY = (Y[:TRN_NUM])
    valX = (X[TRN_NUM:TRN_NUM+VAL_NUM])
    valY = (Y[TRN_NUM:TRN_NUM+VAL_NUM])
    vxlX = (X[TRN_NUM+VAL_NUM:TRN_NUM+VAL_NUM+VXL_NUM])
    vxlY = (Y[TRN_NUM+VAL_NUM:TRN_NUM+VAL_NUM+VXL_NUM])
    tstX = (X[-TST_NUM:])
    tstY = (Y[-TST_NUM:])

    if not os.path.exists(path):
        os.mkdir(path)
    np.save(path+'trnX.npy',trnX)
    np.save(path+'trnY.npy',trnY)
    np.save(path+'valX.npy',valX)
    np.save(path+'valY.npy',valY)
    np.save(path+'vxlX.npy',vxlX)
    np.save(path+'vxlY.npy',vxlY)
    np.save(path+'tstX.npy',tstX)
    np.save(path+'tstY.npy',tstY)

def loadPreshuffledDataset(path):
    trnX = np.load(path+'trnX.npy')
    trnY = np.load(path+'trnY.npy')
    valX = np.load(path+'valX.npy')
    valY = np.load(path+'valY.npy')
    vxlX = np.load(path+'vxlX.npy')
    vxlY = np.load(path+'vxlY.npy')
    tstX = np.load(path+'tstX.npy')
    tstY = np.load(path+'tstY.npy')
    return trnX,trnY,valX,valY,vxlX,vxlY,tstX,tstY



def convert_dict_keys_to_int(label_dict):
    new_dict = {}
    for key_str in label_dict:
        # print("key_str",key_str)
        try: #NOTE: care on this because of floats and maybe .isdigit() adapted for negatives is better
            new_dict[int(key_str)] = label_dict[key_str]
        except ValueError:
            new_dict[key_str] = label_dict[key_str]
    return new_dict


# def saveLabels(path, readable_labels, datatype_labels):
#     data_labels_dict = {
#         "readable_labels" : readable_labels,
#         "datatype_labels" : datatype_labels,
#     }
#     # with open(path+'readable_labels.json', 'w', encoding='utf-8') as f:
#     #     json.dump(readable_labels, f, ensure_ascii=False, indent=4)
#     with open(path+'data_labels.json', 'w', encoding='utf-8') as f:
#         json.dump(data_labels_dict, f, ensure_ascii=False, indent=4)

# def loadLabels(path):
#     # with open(path) as f:
#     #     labels = json.load(f)
#     # return labels
#     # with open(path) as f:
#     with open(path) as f: #TODO: add like loadheader (saftey and suffix)
#         labels_dict = json.load(f)
#     readable_labels = convert_dict_keys_to_int(labels_dict["readable_labels"])
#     datatype_labels = convert_dict_keys_to_int(labels_dict["datatype_labels"])
#     return readable_labels,datatype_labels

def saveHeader(path, header_dict):
    with open(path+'data_header.json', 'w', encoding='utf-8') as f:
        json.dump(header_dict, f, ensure_ascii=False, indent=4)

def loadHeader(path):
    header_path = path+'data_header.json'
    print('header_path',header_path)
    if not os.path.exists(header_path):
        return None
    else:
        # with open(path) as f:
        with open(header_path,'r') as f:
            header_dict = json.load(f)
    return header_dict








# def preprocess_dataset(dataset_id, load_dataset_path="data/", save_dataset_path="data/", preproc_owner=None):
def preprocess_dataset(dataset_id, preproc_owner=None,    load_dataset_path="data/", save_dataset_path="data/", SAVING_PREPROCESSED_VERSION=True):
    if preproc_owner is None:
        preproc_owner = "FIS2025"


    datetimestr = gettimestamp()
    header_dict = {
        "Preprocessed Datetime" : datetimestr, 
        "dataset_id" : dataset_id,
        "preproc_owner" : preproc_owner,
        "load_dataset_path" : load_dataset_path,
        "save_dataset_path" : save_dataset_path,
    }


    preprocessing_function = None
    if dataset_id == "UCI_275_bike_sharing_dataset":
        preprocessing_function = preprocess_bike_sharing_dataset
    elif dataset_id == "UCI_186_wine_quality":
        preprocessing_function = preprocess_wine_quality_dataset
    elif dataset_id == "UCI_374_appliances_energy_prediction":
        preprocessing_function = preprocess_energy_dataset
    elif dataset_id == "UCI_2_adults_dataset":
        preprocessing_function = preprocess_adults_income_dataset
    elif dataset_id == "UCI_31_tree_cover_type_dataset":
        preprocessing_function = preprocess_tree_cover_dataset
    elif dataset_id == "otherSource_cal_housing": 
        preprocessing_function = preprocess_california_housing_dataset
    elif dataset_id == "UCI_203_yearpredictionMSD":
        preprocessing_function = preprocess_song_year_prediction_dataset
    elif dataset_id == "UCI_280_higgs_boson_dataset":
        preprocessing_function = preprocess_higgs_boson_dataset
    elif dataset_id == "UCI_1_abalone_dataset":
        preprocessing_function = process_abalone_dataset
    elif dataset_id == "Kaggle_blastchar_telco_customer_churn":
        preprocessing_function = preprocess_telco_customer_churn_dataset
    elif dataset_id == "UCI_332_online_news_popularity":
        preprocessing_function = preprocess_online_news_popularity_dataset
    elif dataset_id == "Kaggle_mlgulb_credit_card_fraud_dataset":
        preprocessing_function = preprocess_credit_card_fraud_dataset
    elif dataset_id == "Kaggle_ishadss_eucalyptus_dataset":
        preprocessing_function = preprocess_eucalyptus_dataset
    elif dataset_id == "otherSource_Microsoft_search_queries":
        preprocessing_function = preprocess_Microsoft_dataset
    elif dataset_id == "UCI_171_madelon_dataset":
        preprocessing_function = preprocess_madelon_dataset
    elif dataset_id == "UCI_572_taiwanese_bankruptcy_dataset":
        preprocessing_function = preprocess_taiwanese_bankruptcy_dataset
    elif dataset_id == "otherSource_MNIST":
        preprocessing_function = preprocess_mnist_dataset
    else:
        raise NotImplementedError(f'preprocessing for dataset_id=\"{dataset_id}\" not supported or dataset not found at \"{load_dataset_path}\"')
    if True:
        pass
    else:
        raise Exception(f"dataset_id not found\"{dataset_id}\"")




    XY_stuff, label_stuff = preprocessing_function(load_dataset_path,save_dataset_path,preproc_owner)
    label_dict = {
        "readable_labels" : label_stuff[0],
        "full_readable_labels" : label_stuff[1],
    }
    fullX, fullY, trainval_portion, is_test_split_shuffled, shuffle_test_split_seed = XY_stuff #NOTE: dictionary probably makes this easier
    if trainval_portion is None:
        trainval_portion = 0.80 # default to 80/20 split
    header_dict["is_test_split_shuffled"]  = is_test_split_shuffled
    header_dict["shuffle_test_split_seed"] = shuffle_test_split_seed
    header_dict["trainval_portion"]        = trainval_portion
    if SAVING_PREPROCESSED_VERSION:
        final_save_header(save_dataset_path, header_dict) 
        ### ### saveDataset(save_dataset_path, *XY_stuff) #MAYBE A DICTIONARY IS BETTER?? #TODO: yeah, it is, I convert to a dict later anyways
        trnvalX,trnvalY,tstX,tstY = final_save_dataset(save_dataset_path, fullX, fullY, 
                                                        trainval_portion, is_test_split_shuffled, shuffle_test_split_seed)
        # saveLabels(save_dataset_path, *label_stuff)        
        # final_save_labels(save_dataset_path, *label_stuff)
        final_save_labels(save_dataset_path, label_dict)
    else:
        trnvalX,trnvalY,tstX,tstY = final_save_dataset(save_dataset_path, fullX, fullY, 
                                                        trainval_portion, is_test_split_shuffled, shuffle_test_split_seed, SAVING=False)
    return header_dict, (trnvalX,trnvalY,tstX,tstY), label_dict

def load_from_preprocessed(save_dataset_path):
    header_dict = final_load_header(save_dataset_path) 
    trnvalX,trnvalY,tstX,tstY = final_load_dataset(save_dataset_path)
    label_stuff = final_load_labels(save_dataset_path)
    return header_dict, (trnvalX,trnvalY,tstX,tstY), label_stuff












# NOTE: these three may still have slight bugs for the generative datasets

def load_gen_from_preprocessed(save_dataset_path):
    header_dict = final_load_header(save_dataset_path) 
    trnvalX,tstX = final_load_gen_dataset(save_dataset_path)
    label_stuff = final_load_labels(save_dataset_path)
    return header_dict, (trnvalX,tstX), label_stuff


def final_load_gen_dataset(path):
    trnvalX = np.load(path+'trnvalX.npy')
    tstX = np.load(path+'tstX.npy')
    return trnvalX,tstX

def final_save_gen_dataset(path, X, trainval_portion, is_shuffled, shuffle_seed, SAVING=True):
    if shuffle_seed is None and is_shuffled is True:
        warnings.warn("potentially unreproducible shuffling behavior")
    
    NUM = X.shape[0]
    TRN_NUM = int(trainval_portion*NUM)
    TST_NUM = NUM - TRN_NUM

    if not is_shuffled:
        trnvalX = (X[:TRN_NUM])
        tstX = (X[-TST_NUM:])
    else:
        np.random.seed(shuffle_seed)
        perm = np.random.permutation(NUM)
        trnvalX = (X[perm[:TRN_NUM]])
        tstX = (X[perm[-TST_NUM:]])
        
    if not os.path.exists(path):
        os.mkdir(path)
    if SAVING:
        np.save(path+'trnvalX.npy',trnvalX)
        np.save(path+'tstX.npy',tstX)
    return trnvalX,tstX



























































#NOTE: should eventually change to os.join() to be more generic

def preprocess_bike_sharing_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):
    if preproc_owner=="SIAN2022" or preproc_owner=="Chan":
        file = open(load_dataset_path+'hour.csv','r')
        N = 17379 #total number of samples
        bike_share_data = np.zeros((N,17))

        file.readline()
        i=0
        for line in file:
            values = line.replace('\n','').split(',')
            day = int(values[1][8:10])
            values[1] = day

            bike_share_data[i] = [float(val) for val in values]
            i+=1
            
        bike_share_data = bike_share_data[:,1:]    
        bike_share_data[:,13:16] = np.log(1+bike_share_data[:,13:16])
        # saveDataset(save_dataset_path, bike_share_data[:,:13],bike_share_data[:,15])
        XY_stuff = (bike_share_data[:,:13], bike_share_data[:,15], None, False, None)

        
        readable_labels = {
            0  : "day",
            1  : "season",
            2  : "year",
            3  : "month",
            4  : "hour",

            5  : "holiday",
            6  : "day of week",
            7  : "workday",

            8  : "weather",
            9  : "temperature",
            10 : "feels_like_temp",
            11 : "humidity",
            12 : "wind speed",
        }
        datatype_labels = {
            0  : ['semidiscrete', 1, 31], #'ordinal'
            1  : ['semidiscrete', 1, 4],
            2  : ['semidiscrete', 0, 2],
            3  : ['semidiscrete', 1, 12],
            4  : ['semidiscrete', 0, 24],

            5  : ['semidiscrete', 0, 2],
            6  : ['semidiscrete', 0, 7],
            7  : ['semidiscrete', 0, 2],

            8  : ['semidiscrete', 1, 4],   ###		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
            9  : ['continuous', ],
            10 : ['continuous', ],
            11 : ['semicontinuous', 0.0, 1.0],
            12 : ['semicontinuous', 0.0, None],
        }
        #TODO: subreadable labels
        sub_readable_labels = {
            0  : "day",
            1  : ["winter","spring","summer","fall"],   #guessing based on months
            2  : "year",
            3  : "month",
            4  : "hour",

            5  : ["no holiday", "special holiday"],
            6  : ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],  #day of week (pretty sure starts at sunday -- ought to double check)
            7  : ["weekend/holiday", "workday"],

            8  : ['Clear','Mist','Light Rain/Snow', 'Heavy Rain/Snow'],   #Weather
            9  : "temperature",
            10 : "feels_like_temp",
            11 : "humidity",
            12 : "wind speed",
        }
        full_readable_labels = {
            -1 : {}, #TODO: add the output
            'task_type' : "regression",
            "D0" : 13,

            0 : {"label" : "day", 
                "startdim" : 0, "numdims" : 1, 
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 1, "count" : 31},
            1 : {"label" : "season",
                "startdim" : 1, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 1, "count" : 4, #NOTE: should I label these "disc.ordinal.min" and "disc.ordinal.count"?
                "sublabels" : ["winter","spring","summer","fall"],},
            2 : {"label" : "year",
                "startdim" : 2, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 2},
            3 : {"label" : "month",
                "startdim" : 3, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 1, "count" : 12},
            4 : {"label" : "hour",
                "startdim" : 4, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 24},

            5 : {"label" : "holiday",
                "startdim" : 5, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 2},
            6 : {"label" : "day of week",
                "startdim" : 6, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 7},
            7 : {"label" : "workday",
                "startdim" : 7, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinal.timeseries-ish",
                "min" : 0, "count" : 2},
            8 : {"label" : "weather",
                "startdim" : 8, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "discrete.ordinalish.categoricalish",
                "min" : 1, "count" : 4,
                "sublabels" : ['Clear','Mist','Light Rain/Snow', 'Heavy Rain/Snow'],},
                
            9 : {"label" : "temperature",
                "startdim" : 9, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            10: {"label" : "feels_like_temp",
                "startdim" : 10, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            11: {"label" : "humidity",
                "startdim" : 11, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : 1.0},
            12: {"label" : "wind speed",
                "startdim" : 12, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None},
        }


        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")


def preprocess_energy_dataset(load_dataset_path, save_dataset_path, preproc_owner):
    
    if preproc_owner=="SIAN2022":

        file = open(load_dataset_path+'energydata_complete.csv', 'r')
        # Read header and remaining lines
        header = file.readline()
        lines = file.readlines()
        N = len(lines)

        energydatas = np.zeros( (N,31) )
        # (31 features) like original paper
        xd=0
        file_to_open = os.path.join(load_dataset_path, "energydata_complete.csv")
        with open(file_to_open, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            for row in spamreader:
                #print(row)
                if xd!=0:
                    datestr = row.pop(0)#row[0]
                    year=int(datestr[:4])
                    mon=int(datestr[5:7])
                    day=int(datestr[8:10])
                    #print(year,mon,day)
                    hour=int(datestr[11:13])
                    minu=int(datestr[14:16])
                    sec=int(datestr[17:19])
                    #print(hour,minu,sec)
                    #print(len(row))
                    newrow = [float(thing) for thing in row]
                    xdd = datetime(year,mon,day)
                    weekday_num = (xdd.weekday())
                    #print(weekday_num)
                    weekend_status = 1
                    if weekday_num==5 or weekday_num==6:
                        weekend_status = 0

                    energydatas[xd-1,:28] = newrow
                    energydatas[xd-1,28]  = hour*60*60+minu*60+sec
                    energydatas[xd-1,29]  = weekend_status
                    energydatas[xd-1,30]  = weekday_num
                    
                if xd==0:
                    pass
                    #print(len(row))
                    for thing in row:
                        pass
                        #print(thing)

                xd+=1
                #if xd>3:
                #if xd>1 and len(datestr)!=len('2016-01-11 18:30:00'):
                #if xd>1 and xdd.weekday()!=0:
                    #quit()
        print(xd-1)

        #NOTE: MOVING ALL MEAN/VARI SCALING OUTSIDE OF THIS PART

        # # PHASE TWO of ADJUSTING --- TODO: double check these renormaliziation values and do preloading of hardcoded scaling in a more clean fashion
        # means = [97.6949581960983, 3.8018748416518875, 21.686571386748575, 40.259739279784384, 20.341219463849324, 40.42042041370675, 22.267610984880132, 39.24250007720708, 20.855334722410657, 39.02690378814554, 19.59210632801864, 50.94928262962212, 7.910939332403554, 54.609083387583084, 20.267106470136902, 35.38820021508418, 22.02910672298018, 42.93616537238887, 19.485828160608985, 41.552400753376595, 7.4116645553585565, 755.5226019761848, 79.75041803901698, 4.0397517101596145, 38.33083354446415, 3.7607068659741527, 24.988033485049435, 24.988033485049435, 42907.12946541677, 0.7227261211046364, 2.977248543197365]
        # varis = [102.52229296483686, 7.9357865338828555, 1.606024953504309, 3.9791980108449656, 2.1929179723029724, 4.069709427602022, 2.006059709288938, 3.2544940346086375, 2.0428327184256, 4.341210661356689, 1.8445765380903627, 9.021805724091333, 6.090192304747093, 31.14901666343978, 2.109939865220912, 5.114078456225064, 1.9561121604537024, 5.224228315096331, 2.0146613412833356, 4.151392141435619, 5.317274083683381, 7.399253187484764, 14.900710023289937, 2.451158501562397, 11.79441992615117, 4.194541559356488, 14.49626657342557, 14.49626657342557, 24939.38895001474, 0.4476528509657154, 1.9855671861501942]

        # means = [100, 3.8, 22, 40, 20, 40, 20, 40, 20, 40, 20, 50, 8, 50, 20, 35, 20, 40, 20, 40, 7, 750, 80, 4.0, 40, 4, 25, 25, 40000]#, 0.75, 3.0]
        # varis = [100, 8,  1.6,  4,  2,  4,  2,  4,  2,  4,  2,  9, 6, 30,  2,  5,  2,  5,  2,  4, 5, 7.5, 15, 2.5, 12, 4, 15, 15, 25000]#, 0.45, 2.0]

        # # N=energydatas.shape[0]
        # for i in range(29):
        #     #print(means[i],varis[i])
        #     energydatas[:,i] = (energydatas[:,i]-means[i])/varis[i]
        all_energies_adjusted = np.zeros( (N,29+2+7) ) #38
        all_energies_adjusted[:,:29] = energydatas[:,:29]

        # putting back in the Y scaling at least to become more managable sizes
        all_energies_adjusted[0] = (all_energies_adjusted[0] - 100. ) / 100.

        all_energies_adjusted[np.arange(N),(29+energydatas[:,29]).astype(int)] = 1
        all_energies_adjusted[np.arange(N)[:,None],(31+energydatas[:,30]).astype(int)[:,None]] = 1

        XY_stuff = (all_energies_adjusted[:, 1:], all_energies_adjusted[:, :1], None, False, None)

        readable_labels = {
            0: "lights",
            1: "T1",
            2: "RH_1",
            3: "T2",
            4: "RH_2",
            5: "T3",
            6: "RH_3",
            7: "T4",
            8: "RH_4",
            9: "T5",
            10: "RH_5",
            11: "T6",
            12: "RH_6",
            13: "T7",
            14: "RH_7",
            15: "T8",
            16: "RH_8",
            17: "T9",
            18: "RH_9",
            19: "T_out",
            20: "Press_mm_hg",
            21: "RH_out",
            22: "Windspeed",
            23: "Visibility",
            24: "Tdewpoint",
            25: "rv1",
            26: "rv2",
            27: "time_of_day",
            28: "weekend_status",
            29: "weekday"
        }

        full_readable_labels = {
            -1: {"label": "Appliances", "task_type": "regression", "units" : "Wh * 100"},
            "task_type": "regression",
            "D0": 30,
            "D" : 37,

            0: {"label": "lights", "startdim": 0, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": None, "units" : "Wh"},
            1: {"label": "T1",     "startdim": 1, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None, "units" : "deg C"},
            2: {"label": "RH_1",   "startdim": 2, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0, "units" : "%"},
            3: {"label": "T2",     "startdim": 3, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            4: {"label": "RH_2",   "startdim": 4, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0},
            5: {"label": "T3",     "startdim": 5, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            6: {"label": "RH_3",   "startdim": 6, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0},
            7: {"label": "T4",     "startdim": 7, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            8: {"label": "RH_4",   "startdim": 8, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0},
            9: {"label": "T5",     "startdim": 9, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            10: {"label": "RH_5",  "startdim": 10, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0},
            11: {"label": "T6",    "startdim": 11, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            12: {"label": "RH_6",  "startdim": 12, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0},
            13: {"label": "T7",    "startdim": 13, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            14: {"label": "RH_7",  "startdim": 14, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0},
            15: {"label": "T8",    "startdim": 15, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            16: {"label": "RH_8",  "startdim": 16, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0},
            17: {"label": "T9",    "startdim": 17, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            18: {"label": "RH_9",  "startdim": 18, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 100.0},
            19: {"label": "T_out", "startdim": 19, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            20: {"label": "Press_mm_hg", "startdim": 20, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            21: {"label": "RH_out",      "startdim": 21, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "_lb": 0.0, "ub": 100.0},
            22: {"label": "Windspeed",   "startdim": 22, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": None, "units" : "m/s"},
            23: {"label": "Visibility",  "startdim": 23, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": None, "units" : "km"},
            24: {"label": "Tdewpoint",   "startdim": 24, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            25: {"label": "rv1",         "startdim": 25, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            26: {"label": "rv2",         "startdim": 26, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": None, "ub": None},
            27: {"label": "time_of_day",    "startdim": 27, "numdims": 1, "encoding": "cts.raw", "type": "cts.", "lb": 0.0, "ub": 24.0*60*60}, #86400
            28: {"label": "weekend_status", "startdim": 28, "numdims": 2, "encoding": "disc.onehot", "type": "discrete.categorical", "sublabels": ["weekend", "workday"]},
            29: {"label": "weekday",        "startdim": 30, "numdims": 7, "encoding": "disc.onehot", "type": "discrete.categorical", "sublabels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]} #NOTE: are we sure?
        }

        label_stuff = (readable_labels, full_readable_labels)

        saveDataset(save_dataset_path, all_energies_adjusted[:, 1:], all_energies_adjusted[:, :1])  

        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")





def preprocess_wine_quality_dataset(load_dataset_path, save_dataset_path, preproc_owner):

    if preproc_owner=="SIAN2022":

        white_wines = loadWines(load_dataset_path+'winequality-white.csv',4898)
        red_wines = loadWines(load_dataset_path+'winequality-red.csv',1599)
        white_wines[:,11]=1
        red_wines[:,12]=1

        # print('white_wines',white_wines.shape)
        # print('red_wines',red_wines.shape)
        wines = np.concatenate([white_wines,red_wines],axis=0)
        # print('wines',wines.shape)

        XY_stuff = (wines[:, :-1], wines[:, -1], None, True, 0)


        
        # readable_labels = ['fixed acidity', 'volatile acidity', 'citric acid',
        #            'residual sugar', 'chlorides', 'free sulfur dioxide',
        #            'total sulfur dioxide', 'density', 'pH', 'sulphates',
        #            'alcohol', 'quality']
        readable_labels_list = ['fixed acidity', 'volatile acidity', 'citric acid',
                   'residual sugar', 'chlorides', 'free sulfur dioxide',
                   'total sulfur dioxide', 'density', 'pH', 'sulphates',
                   'alcohol', 'color', 'quality']
        readable_labels = {}
        # datatype_labels = {}
        for d,label in enumerate(readable_labels_list[:-1]):
            readable_labels[d] = label
            # datatype_labels[d] = ['continuous']
        # label_stuff = (readable_labels, datatype_labels)
        full_readable_labels = {
            -1 : {"label" : "wine quality",
                "startdim" : 0, "numdims" : 1,
                "encoding" : "cts.disc", "type" : "disc.ordinal",
                "count" : 10,
            },
            "task_type" : "regression",
            "D0" : 12,

            0 : {"label" : "fixed acidity", 
                "startdim" : 0, "numdims" : 1, 
                "encoding" : "cts.raw", "type" : "cts.",
                },
            1 : {"label" : "volatile acidity",
                "startdim" : 1, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            2 : {"label" : "citric acid",
                "startdim" : 2, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            3 : {"label" : "residual sugar",
                "startdim" : 3, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            4 : {"label" : "chlorides",
                "startdim" : 4, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            5 : {"label" : "free sulfur dioxide",
                "startdim" : 5, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            6 : {"label" : "total sulfur dioxide",
                "startdim" : 6, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            7 : {"label" : "density",
                "startdim" : 7, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                },
            8 : {"label" : "pH",
                "startdim" : 8, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            9 : {"label" : "sulphates",
                "startdim" : 9, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            10: {"label" : "alcohol",
                "startdim" : 10, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            11: {"label" : "color",
                "startdim" : 11, "numdims" : 2,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 2,
                "sublabels" : ["white", "red"],},
        }


        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff

    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")


# @James
def loadWines(CSV_FILE_NAME,N):
    wines = np.zeros( (N,14) )
    xd=0
    
    with open(CSV_FILE_NAME, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='\"')
        for row in spamreader:
            if xd!=0:
                newrow = [float(thing) for thing in row]
                quality = newrow.pop(11)
                wines[xd-1,:11] = newrow
                wines[xd-1,13]  = quality
                
            if xd==0:
                pass
                #print(len(row))
                for thing in row:
                    pass
                    #print(thing)

            xd+=1
    return wines





def preprocess_adults_income_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):
    if preproc_owner=="InstaSHAP2025":
        N = 32561
        D = 15
        CTS_VARS = [1,3, 11,12,13]
        CSV_FILENAME = load_dataset_path+'adult.data'
        #TODO: add test to this (it is a separate file)
        

        def get_cumulative_index_sizes(I_ks):
            D=len(I_ks)
            cum_n = 0
            cum_I_ks = [0,]
            for d in range(D):
                cum_n += I_ks[d]
                cum_I_ks.append(cum_n)
            return cum_I_ks

        event_dictionary = {
            0 : ['<=50K', '>50K'],
            1 : ['<20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-79', '80-89', '90+'],
            2 : ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked', None],
            3 : ['<50K', '50K-100K', '100K-150K', '150K-200K', '200K-250K', '250K-300K', '300K-350K', '350K-400K', '400K+'],
            4 : ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'],

            #DUPLICATED FEATURE AS A NUMERICAL VALUE
            5 : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            
            6 : ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
            7 : ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv', None],
            8 : ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
            9 : ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
            10 : ['Male', 'Female'],
            11 : ['0', '<1K', '1K-3K', '3K-10K', '10K-30K', '30K+'],
            12 : ['1K', '1K-1.5K', '1.5K-2K', '2K-2.5K', '2.5K-5K'],
            13 : ['0-10', '10-20', '20-30', '30-35', '35-40', '40-45', '45-50', '50-60', '60-80', '80-100'],
            14 : ['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands', None],
        }

        readable_label_dict = {
            0 : "income",
            1 : "age",
            2 : "workclass",
            3 : "fnlwgt",
            4 : "education",
            5 : "education-num",
            6 : "marital-status",
            7 : "occupation",
            8 : "relationship",
            9 : "race",
            10 : "sex",
            11 : "capital-gain",
            12 : "capital-loss",
            13 : "hours-per-week",
            14 : "native-country",
        }

        bins_dictionary = {
            1 : [14.5, 19.5, 24.5, 29.5, 34.5, 39.5, 44.5, 49.5, 54.5, 59.5, 64.5, 69.5,79.5,89.5,99.5],
            3 : [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 1500000],
            
            11 : [-0.5,0.5,1000.5,3000.5,10000.5,   30000.5, 100000.5],
            12 : [-0.5,1000.5,1500.5,2000.5,2500.5,5000.5],
            13 : [0.5, 10.5, 20.5,30.5, 35.5, 40.5, 45.5, 50.5, 60.5, 80.5,100.5],
        }
        D=15


        I_ks = []
        for d in range(D):
            if d in CTS_VARS:
                I_ks.append( 1 )
                #I_ks.append( len(bins_dictionary[d])-1 )
                #assert len(bins_dictionary[d]) == len(event_dictionary[d])+1, 'mislabeled readable labels @ '+str(d) +\
                #                                                                '  ('+str(len(bins_dictionary[d])-1)+','+str(len(event_dictionary[d]))+')'
            else:
                I_ks.append( len(event_dictionary[d]) )
        I_ks=tuple(I_ks)
        print('I_ks    \t',I_ks)
        print("D",len(I_ks),"\t\tonehot D",sum(I_ks))

        cum_I_ks = get_cumulative_index_sizes(I_ks)
        print('cum_I_ks \t',cum_I_ks)

        rescale_cts_vars_dict = {
            1 : (40,10),
            3 : (200*1000,100*1000),
            
            11: (1000,8000),
            12: (80,400),
            13: (40,10),
        }

        N = 32561
        ONEHOT_D = sum(list(I_ks))
        X_arr = np.zeros( (N,ONEHOT_D), dtype=float )

        import csv
        with open(CSV_FILENAME, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',') #, quotechar='|')
            n=0
            for row in spamreader:
                if n<N: #white space at the end of the file
                    cum_n = 0
                    for d in range(D):
                        if d>0:
                            value = row[d-1].strip()
                        else:
                            value = row[-1].strip()

                        v_id = -1
                        if d in CTS_VARS:
                            value = int(value)

                            # v_id = -1
                            # bins_d = bins_dictionary[d]
                            # for bb,b in enumerate(bins_d):
                            #     if value>b:
                            #         v_id = bb

                            #NOTE: JAM ON 04/13/2025 -- this looks like where I made the original change
                            #                            everything seems fine, I just used mahgenta as a starting point
                            value_rescaled = (value - rescale_cts_vars_dict[d][0]) / rescale_cts_vars_dict[d][1]
                            X_arr[n,cum_n] = value_rescaled
                            cum_n += 1

                        else:
                            if d==5:
                                value = int(value)
                            if value not in event_dictionary[d]:
                                print('FAILURE\t',d)
                            v_id = event_dictionary[d].index(value)
        
                            X_arr[n,cum_n+v_id] = 1
                            cum_n += I_ks[d]
                pass
                n+=1
        print(n)



        #REMOVING FEATURE #5

        d_to_remove = 5
        print(I_ks[5],cum_I_ks[5])
        I_5 = I_ks[5]
        cum_I_5 = cum_I_ks[5]

        print('X_arr',X_arr.shape)
        X_arr = np.concatenate([X_arr[:,:cum_I_5],X_arr[:,cum_I_5+I_5:]],axis=-1)
        print('X_arr',X_arr.shape)

        for d in range(D):
            if d>5:
                readable_label_dict[d-1] = readable_label_dict[d]
                event_dictionary[d-1]    = event_dictionary[d]
        I_ks=list(I_ks)
        I_ks.pop(5)
        I_ks=tuple(I_ks)
        D=14
        del readable_label_dict[14]
        del event_dictionary[14]

        print("D",len(I_ks),"\t\tonehot D",sum(I_ks))
        print('I_ks    \t',I_ks)
        cum_I_ks = get_cumulative_index_sizes(I_ks)
        print('cum_I_ks \t',cum_I_ks)



        XY_stuff = (X_arr[:, 2:], X_arr[:, 1], None, True, 0)




        
        full_readable_labels = {
            # -1 : {"label" : "income level",
            #     "startdim" : 0, "numdims" : 2,
            #     "encoding" : "disc.onehot", "type" : "disc.ordinal",
            #     "count" : 2,
            #     "sublabels" : ['<=50K', '>50K'],
            # },
            -1 : {"label" : "income level",
                "startdim" : 0, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "disc.ordinal",
                "min" : 0, "count" : 2,
                "sublabels" : ['<=50K', '>50K'],
            },
            "task_type" : "binary_classification",
            "D0" : 13,

            0 : {"label" : "age", 
                "startdim" : 0, "numdims" : 1, 
                "encoding" : "cts.raw", "type" : "cts.",
                },
            1 : {"label" : "workclass",
                "startdim" : 1, "numdims" : 10,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 10,
                "sublabels" : ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked', None],
                },
            2 : {"label" : "fnlwgt",
                "startdim" : 11, "numdims" : 1, #NOTE: I shouldnt be manually inputting 'startdim' #TODO: add a cumulative one like above (but for all datasets)
                "encoding" : "cts.raw", "type" : "cts.",
                },
            3 : {"label" : "education",
                "startdim" : 12, "numdims" : 16,
                "encoding" : "disc.onehot", "type" : "disc.ordinal",
                "count" : 16,
                "sublabels" : ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'],
                },
            4 : {"label" : "marital-status",
                "startdim" : 28, "numdims" : 7,
                "encoding" : "cts.raw", "type" : "cts.",
                "count" : 7,
                "sublabels" :  	 ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
                },
            5 : {"label" : "occupation",
                "startdim" : 35, "numdims" : 16,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 16,
                "sublabels" :  	 ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv', None],
                },
            6 : {"label" : "relationship",
                "startdim" : 51, "numdims" : 6,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 6,
                "sublabels" :  	  ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
                },
            7 : {"label" : "race",
                "startdim" : 57, "numdims" : 5,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 5,
                "sublabels" :  	 ['white', 'black', 'asian-pac-islander', 'native-american-indian-eskimo', 'other'],
                },
            8 : {"label" : "sex",
                "startdim" : 62, "numdims" : 2,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 2,
                "sublabels" :  	['male', 'female'] ,
                },

            9 : {"label" : "capital-gain",
                "startdim" : 64, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            10: {"label" : "capital-loss",
                "startdim" : 65, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : None, "ub" : None},
            11: {"label" : "hours-per-week",
                "startdim" : 66, "numdims" : 2,
                "encoding" : "cts.positive", "type" : "cts.",
                "lb" : 0.0, "ub" : None},
                
            12 : {"label" : "native-country",
                "startdim" : 67, "numdims" : 43,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 43,
                "sublabels" :  	['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands', None]
                },
        }
        readable_labels={}
        for thing in full_readable_labels:
            if type(thing)==int:
                readable_labels[thing] = full_readable_labels[thing]['label']

        # label_stuff = (readable_labels, datatype_labels, full_readable_labels)
        label_stuff = (readable_labels, full_readable_labels)

        #FROM INSTASHAP CODE: TODO (maybe add these lower case versions)
        # 1 : ['state gov', 'self emp\n(not inc)', 'private', 'federal gov', 'local gov',
        #         '(missing value)', 'self emp\n(inc)', 'without pay', 'never worked', None],
        #     3 : ['preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS grad', 'some college',
        #         'assoc. voc', 'assoc. acdm', 'bachelors', 'masters', 'prof school', 'doctorate'],
        #     4 : ['never\nmarried', 'married\n(civ spouse)', 'divorced',
        #         'married\n(spouse\nabsent)', 'separated',
        #         'married\n(AF spouse)', 'widowed'],
        #     5 : ['admin/clerical', 'exec/managerial', 'handlers/cleaners', 'prof/specialty', 'other service',
        #         'sales', 'craft/repair', 'transport/moving', 'farming/fishing', 'machine op inspct', 'tech support',
        #         '(missing value)', 'protective serv', 'armed forces', 'priv house serv', None],
        #     6 : ['not in\nfamily', 'husband', 'wife', 'own\nchild', 'unmarried', 'other\nrelative'],
        #     #7 : ['white', 'black', 'asian/\npacific\nislander', 'american\nindian/\neskimo', 'other'],
        #     7 : ['white', 'black', 'asian', 'american\nindian', 'other'],
        #     8 : ['male', 'female'],


        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")
        





def preprocess_tree_cover_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):
    if preproc_owner=="InstaSHAP2025":







        readable_labels_dict = {
            'elevation (m)'     : 0,
            'aspect (azimuth)'  : 1,
            'slope (deg)'       : 2,
            'horizontal_dist_to_hydro (m)' : 3,
            'vertical_dist_to_hydro (m)'   : 4,
            'horizontal_dist_to_road (m)'  : 5,
            'hillshade_9am'  : 6, #[0,255)
            'hillshade_noon' : 7, #[0,255)
            'hillshade_3pm'  : 8, #[0,255)
            'horizontal_dist_to_fire_point (m)' : 9,
            
            'wilderness_area_label' : list(range(10,14)), #Rawah=1, Neota=2, Comanche Peak=3, Cache la Poudre =4
            'soil_type' : list(range(14,54)),
        }


        # original splits to have a balanced training set and unbalanced test set
        # (maybe not reasonable under modern ML frameworks). 
        # either way, we ignore it
        TRN_N =  11340
        VAL_N =   3780
        TST_N = 565892
        N = TRN_N+VAL_N+TST_N  #581,012

        all_data_array = np.zeros((N,55),dtype=int)

        CSV_PATH = load_dataset_path + "covtype.data"

        xd=0
        with open(CSV_PATH, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                all_data_array[xd] = [int(thing) for thing in row]
                xd+=1

        # quant_mus  = [2959.36, 155.65, 14.10, 269.43, 46.42, 2350.15, 212.15, 223.32, 142.53, 1980.29]
        # quant_vars = [ 279.98, 111.91,  7.49, 212.55, 58.30, 1559.25,  26.77,  19.77,  38.27, 1324.19]


        # soil_remappings = {
        #     0 : list(range(0,6)),
        #     1 : list(range(6,8)),
        #     2 : list(range(8,9)),
        #     3 : list(range(9,13)),
        #     4 : list(range(13,15)),
        #     5 : list(range(15,17)),
        #     6 : list(range(17,18)),
        #     7 : list(range(18,21)),
        #     8 : list(range(21,23)),
        #     9 : list(range(23,34)),
        #     10 : list(range(34,40)),    
        # }
        # soil_remapping_tensor = np.zeros((40,11))
        # for c2 in range(11):
        #     for c1 in soil_remappings[c2]:
        #         soil_remapping_tensor[c1,c2] = 1
                
        # soil_remappings2 = {
        #     0 : list(range(0,6)),
        #     1 : list(range(6,8)),
        #     2 : list(range(8,13)),
        #     3 : list(range(13,15)),
        #     4 : list(range(15,18)),
        #     5 : list(range(18,34)),
        #     6 : list(range(34,40)),    
        # }
        # soil_remapping_tensor2 = np.zeros((40,7))
        # for c2 in range(7):
        #     for c1 in soil_remappings2[c2]:
        #         soil_remapping_tensor2[c1,c2] = 1
                
        soil_remappings3 = { 
            0 : list(range(0,6)),   #2 --> lower montane
            1 : list(range(6,18)),  #3,4,5,6 --> upper montane
            2 : list(range(18,34)), #7 --> subalpine
            3 : list(range(34,40)), #8 --> alpine  
        }
        soil_remapping_tensor3 = np.zeros((40,4))
        for c2 in range(4):
            for c1 in soil_remappings3[c2]:
                soil_remapping_tensor3[c1,c2] = 1

        all_simple_soils_arr3 = np.matmul(all_data_array[:,14:54],soil_remapping_tensor3)
        all_data3 = np.concatenate([all_data_array[:,:14],all_simple_soils_arr3],axis=1)

        one_hot_labels = np.zeros((all_data_array.shape[0],7),dtype=int)
        one_hot_labels[np.arange(all_data_array.shape[0]) , (all_data_array[:, 54]-1).astype(int)] = 1


        XY_stuff = (all_data3, one_hot_labels, None, True, 0)



        """
        Name                                     Data Type    Measurement                       Description

        Elevation                               quantitative    meters                       Elevation in meters
        Aspect                                  quantitative    azimuth                      Aspect in degrees azimuth
        Slope                                   quantitative    degrees                      Slope in degrees
        Horizontal_Distance_To_Hydrology        quantitative    meters                       Horz Dist to nearest surface water features
        Vertical_Distance_To_Hydrology          quantitative    meters                       Vert Dist to nearest surface water features
        Horizontal_Distance_To_Roadways         quantitative    meters                       Horz Dist to nearest roadway
        Hillshade_9am                           quantitative    0 to 255 index               Hillshade index at 9am, summer solstice
        Hillshade_Noon                          quantitative    0 to 255 index               Hillshade index at noon, summer soltice
        Hillshade_3pm                           quantitative    0 to 255 index               Hillshade index at 3pm, summer solstice
        Horizontal_Distance_To_Fire_Points      quantitative    meters                       Horz Dist to nearest wildfire ignition points
        Wilderness_Area (4 binary columns)      qualitative     0 (absence) or 1 (presence)  Wilderness area designation
        Soil_Type (40 binary columns)           qualitative     0 (absence) or 1 (presence)  Soil Type designation
        Cover_Type (7 types)                    integer         1 to 7                       Forest Cover Type designation
        """
        pass        
        class_labels = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine","Cottonwood/Willow","Aspen","Douglas-fir","Krummholz"]
        full_readable_labels = {
            # -1 : {"label" : "tree species",
            #     "startdim" : 0, "numdims" : 7,
            #     "encoding" : "disc.ordinal", "type" : "disc.categorical",
            #     "min" : 1, "count" : 7,
            #     "sublabels" : class_labels,
            # },
            -1 : {"label" : "tree species",
                "startdim" : 0, "numdims" : 7,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 7,
                "sublabels" : class_labels,
            },
            "task_type" : "multiclass_classification",
            "D0" : 12,

            0 : {"label" : "elevation (m)", 
                "startdim" : 0, "numdims" : 1, 
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                },
            1 : {"label" : "aspect (azimuth)",
                "startdim" : 1, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'degrees azimuth',
                },
            2 : {"label" : "slope (deg)",
                "startdim" : 2, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'degrees',
                },
            3 : {"label" : "Horizontal_Distance_To_Hydrology",
                "startdim" : 3, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                'description' : "Horz Dist to nearest surface water features",
                },
            4 : {"label" : "Vertical_Distance_To_Hydrology",
                "startdim" : 4, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                'description' : "Vert Dist to nearest surface water features",
                },
            5 : {"label" : "Horizontal_Distance_To_Roadways",
                "startdim" : 5, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                'description' : "Horz Dist to nearest roadway",
                },
            9 : {"label" : "Horizontal_Distance_To_Fire_Points", #idk why put as 9 instead of putting here
                "startdim" : 9, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'meters',
                'description' : "Horz Dist to nearest wildfire ignition points",
                },

            
            6 : {"label" : "Hillshade_9am",
                "startdim" : 6, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'pixel brightness [0,256)',
                'description' : 'Hillshade index at 9am, summer solstice',
                },
            7 : {"label" : "Hillshade_Noon",
                "startdim" : 7, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'pixel brightness [0,256)',
                'description' : 'Hillshade index at noon, summer solstice',
                },
            8 : {"label" : "Hillshade_3pm",
                "startdim" : 8, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                'unit' : 'pixel brightness [0,256)',
                'description' : 'Hillshade index at 3pm, summer solstice',
                },


            10: {"label" : "wilderness area",
                "startdim" : 10, "numdims" : 4,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 4,
                "sublabels" : ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"],},
            11: {"label" : "soil type",
                "startdim" : 14, "numdims" : 4,
                "encoding" : "disc.onehot", "type" : "disc.categorical",
                "count" : 4,
                "sublabels" : ["lower montane", "upper montane", "subalpine", "alpine"],},
        }
        readable_labels={}
        for thing in full_readable_labels:
            if type(thing)==int:
                readable_labels[thing] = full_readable_labels[thing]['label']

        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")
        






# https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
def preprocess_song_year_prediction_dataset(load_dataset_path, save_dataset_path, preproc_owner):

    if preproc_owner=="SIAN2022":
        file_path = load_dataset_path+'YearPredictionMSD.txt'
        
        # Load the dataset
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        N = len(lines)
        readable_labels_list =  ['T1_mean', 'T2_mean', 'T3_mean', 'T4_mean', 'T5_mean', 'T6_mean',
                                 'T7_mean', 'T8_mean', 'T9_mean', 'T10_mean', 'T11_mean', 'T12_mean',
                            
                            'T1_var', 'T2_var', 'T3_var', 'T4_var', 'T5_var', 'T6_var',
                            'T7_var', 'T8_var', 'T9_var', 'T10_var', 'T11_var', 'T12_var',

                            'T1x2_cov', 'T2x3_cov', 'T3x4_cov', 'T4x5_cov', 'T5x6_cov', 'T6x7_cov', 'T7x8_cov', 'T8x9_cov', 'T9x10_cov', 'T10x11_cov', 'T11x12_cov',
                            'T1x3_cov', 'T2x4_cov', 'T3x5_cov', 'T4x6_cov', 'T5x7_cov', 'T6x8_cov', 'T7x9_cov', 'T8x10_cov', 'T9x11_cov', 'T10x12_cov',
                            'T1x4_cov', 'T2x5_cov', 'T3x6_cov', 'T4x7_cov', 'T5x8_cov', 'T6x9_cov', 'T7x10_cov', 'T8x11_cov', 'T9x12_cov',
                            'T1x5_cov', 'T2x6_cov', 'T3x7_cov', 'T4x8_cov', 'T5x9_cov', 'T6x10_cov', 'T7x11_cov', 'T8x12_cov',
                            'T1x6_cov', 'T2x7_cov', 'T3x8_cov', 'T4x9_cov', 'T5x10_cov', 'T6x11_cov', 'T7x12_cov',
                            'T1x7_cov', 'T2x8_cov', 'T3x9_cov', 'T4x10_cov', 'T5x11_cov', 'T6x12_cov',
                            'T1x8_cov', 'T2x9_cov', 'T3x10_cov', 'T4x11_cov', 'T5x12_cov',
                            'T1x9_cov', 'T2x10_cov', 'T3x11_cov', 'T4x12_cov',
                            'T1x10_cov', 'T2x11_cov', 'T3x12_cov',
                            'T1x11_cov', 'T2x12_cov',
                            'T1x12_cov']


        all_songs = np.zeros( (N,91) )
        xd=0

        with open(file_path, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            for row in spamreader:
                
                timbre_features = [float(thing) for thing in row[1:91]]
                year = int(row[0])
                
                all_songs[xd,:90]  = timbre_features
                all_songs[xd,90]  = year
                    
                if xd==0:
                    pass
                    # print(len(row))
                    for thing in row:
                        pass
                        # print(thing)

                xd+=1
        # print()
        # print(xd)




        #TODO: these also need to be cleaned up for certain
        timbre_mean_variances = [6.067552420614484, 51.58030078563849, 35.26855067805859, 16.32277403422655, 22.860763230451074, 12.857738981859695, 14.571859030694965, 7.9638197560638435, 10.582850261206456, 6.530225383743962, 4.370843897103604, 8.320181972025107]
        tmbr_mn_vr = [6.1, 52., 35., 16., 23., 13., 14.6, 8., 10.6, 6.5, 4.4, 8.3]
        timbre_variances_variances = [22.259610748445795, 1749.3659923352145, 1261.4835676661519, 1092.8299837016993, 475.7077416295253, 576.8654341655065, 317.49897988098854, 309.36443442068, 214.01329403172107, 165.69922177696276, 186.9605474775762, 153.47549996570197]
        tmbr_vr_vr = [22.3, 1750., 1260., 1090., 476., 577., 317., 309., 214., 166., 187., 153.]
        mean_loudness = 43.4

        ##temporary check
        ##tmbr_mn_vr = np.sqrt(tmbr_vr_vr)
        
        
        means = [1.446465670129212, 50.49371512287865, 0.31461869774512286, 21.5376066100241, 19.825133012414494, 746.451527348147, 0.013864693245705413, 0.026910949909348406, 3.6794805901703596, 0.038378515328097336, 1.3270331897643781, 1.4726627818596403, 1.9664048644540157, 2.5425893083354447, 0.09445148213833292, 4.156017228274639, 0.14532556371928046, 0.09668102356219914]
        varis = [2.8548110862198772, 145.10164785679845, 1.0331111250447251, 112.49296131838766, 38.3640446996599, 1619.4024158309426, 0.03980270042948673, 0.04370700533330614, 14.950876250430557, 0.16001205181532147, 1.2555693571459847, 1.7733637329533534, 2.4342244173592826, 3.742239355423344, 0.3068298176182689, 4.191033795419051, 0.35242877897371705, 0.29552293184316575]
        max_class = [8.0, 13.0, 9.0, 20.0, 2.0, 11.0, 1.0, 1.0] #og wrong
        means = [1.4, 50.,  0.3, 22.0, 20,  750., 0.014, 0.027, 3.7, 0.04, 1.3, 1.5, 2.0, 2.5, 0.09, 4.2, 0.15, 0.10]
        varis = [2.8, 150., 1.0, 110., 40, 1600., 0.040, 0.044, 15., 0.16, 1.3, 1.8, 2.4, 3.7, 0.31, 4.2, 0.35, 0.30]

        means = [2.3151662611516626, 80.81861053933578, 0.5035685320356853, 34.47239792772309, 31.731467964314678, 1194.7462199688305, 0.02219138047072152, 0.04307279776650371, 5.889257862693596, 0.0614274128142742, 2.124006488240065, 2.357096512570965, 3.1473641524736413, 4.069586374695864, 0.15117599351175995, 6.651987023519871, 0.23260340632603407, 0.15474452554744525]
        varis = [3.32164940009589, 176.77193866707827, 1.270104918190283, 140.74358671139055, 44.473699719976125, 1913.5916841375854, 0.04848635549219245, 0.048594569848851916, 18.567683614574786, 0.19890920659518405, 0.9112878723722002, 1.7172070359954965, 2.4014938466409914, 4.02500593049425, 0.3769739017961091, 3.392703352124569, 0.42249149304043987, 0.3616609701924769]
        max_class = [8.0, 13.0, 9.0, 20.0, 2.0, 11.0, 1.0, 1.0]

        all_songs_adjusted = np.zeros( (all_songs.shape[0],91) )
        k=0
        for i in range(1,12+1):
            print(all_songs[:5,k])
            if i==1:
                all_songs_adjusted[:,k] = (all_songs[:,k]-mean_loudness)/tmbr_mn_vr[i-1]
            else:
                all_songs_adjusted[:,k] = all_songs[:,k]/tmbr_mn_vr[i-1]
            print(all_songs_adjusted[:5,k])
            k+=1

        for d in range(12): #difference
            for f in range(1,12+1): #first
                i=f
                j=f+d
                if j<12+1:
                    if i==j:
                        all_songs_adjusted[:,k] = all_songs[:,k]/np.sqrt(tmbr_vr_vr[i-1]*tmbr_vr_vr[i-1])
                    else:
                        all_songs_adjusted[:,k] = all_songs[:,k]/np.sqrt(tmbr_vr_vr[i-1]*tmbr_vr_vr[j-1])
                    k+=1
        CALENDAR_YEAR_NORMALIZED = True
        if not CALENDAR_YEAR_NORMALIZED:
            all_songs_adjusted[:,90] = all_songs[:,90]
        else:
            all_songs_adjusted[:,90] = (all_songs[:,90]-2000)/10  #NOTE: this preprocessing is simple but important to note somewhere -- maybe if later autopreproc is better, we remove this

        XY_stuff = (all_songs_adjusted[:, :90], all_songs_adjusted[:, 90], None, True, 0)

        full_readable_labels = {
            -1: {
                "label": "year",
                "startdim": 0,
                "numdims": 1,
                #"encoding": "cts.disc",
                "encoding": "cts.raw",
                "type": "discrete.ordinal",
                "min": 1922,
                "max": 2010,
                "units" : "((calendar year)-2000)/10" if CALENDAR_YEAR_NORMALIZED else "calendar year",
            },
            "task_type": "regression",
            "D0": 90
        }
        for d in range(90):
            full_readable_labels[d] = {
                "label": readable_labels_list[d],
                "startdim": d,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": "cts."
            }
        readable_labels = get_readables_from_full_readables(full_readable_labels)

        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff

    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")



# original source should be: https://www.sciencedirect.com/science/article/pii/S016771529600140X
# 
# other sources:
#       https://www.kaggle.com/datasets/camnugent/california-housing-prices
#       https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
#       https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646
#       https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
def preprocess_california_housing_dataset(load_dataset_path, save_dataset_path, preproc_owner):

    if preproc_owner=="SIAN2022":
        file = open(load_dataset_path+'cal_housing.data','r') #looks like it was from "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html"
        cal_house_data = np.zeros((20640,9))

        i=0
        for line in file:
            # print(i)
            values = line.replace('\n','').split(',')
            #print(values)
            x = float(values[0])
            y = float(values[1])

            cal_house_data[i] = [float(val) for val in values]
            #print(x,y)
            i+=1

        
        log_scale_cal_house = np.array(cal_house_data)
        for i in range(3,8):
            # print(i)
            log_scale_cal_house[:,i] = np.log(1 + log_scale_cal_house[:,i])


        HOUSE_PRICE_IN_100K_UNITS = True
        if HOUSE_PRICE_IN_100K_UNITS:
            fullY = log_scale_cal_house[:, 8] / (100*1000)
        else:
            fullY = log_scale_cal_house[:, 8]

        XY_stuff = (log_scale_cal_house[:, :8], fullY, 0.8, False, None) #NOTE: check on this, do I need to do this, I don't remember doing it

        readable_labels = {
            0: "longitude",
            1: "latitude",
            2: "housing_median_age",
            3: "total_rooms",
            4: "total_bedrooms",
            5: "population",
            6: "households",
            7: "median_income"
        }

        full_readable_labels = {
            -1: {
                "label": "median_house_value",
                "startdim": 0,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": "cts.",
                "lb": 0.0,
                "ub": None,
                "units": "hundreds of thousands of dollars" if HOUSE_PRICE_IN_100K_UNITS else "dollars"
            },
            "task_type": "regression",
            "D0": 8,
            
            0: {
                "label": "longitude",
                "startdim": 0,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": "cts.",
                "lb": None,
                "ub": None
            },
            1: {
                "label": "latitude",
                "startdim": 1,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": "cts.",
                "lb": None,
                "ub": None
            },
            2: {
                "label": "housing_median_age",
                "startdim": 2,
                "numdims": 1,
                "encoding": "cts.",
                "type": "cts.log1p",
                "lb": None, #original variable is positive, but log-transformed one is not
                "ub": None,
                "transformation": "log(1+x)"
            },
            3: {
                "label": "total_rooms",
                "startdim": 3,
                "numdims": 1,
                "encoding": "cts.",
                "type": "cts.log1p",
                "lb": None, #original variable is positive, but log-transformed one is not
                "ub": None,
                "transformation": "log(1+x)"
            },
            4: {
                "label": "total_bedrooms",
                "startdim": 4,
                "numdims": 1,
                "encoding": "cts.",
                "type": "cts.log1p",
                "lb": None, #original variable is positive, but log-transformed one is not
                "ub": None,
                "transformation": "log(1+x)"
            },
            5: {
                "label": "population",
                "startdim": 5,
                "numdims": 1,
                "encoding": "cts.",
                "type": "cts.log1p",
                "lb": None, #original variable is positive, but log-transformed one is not
                "ub": None,
                "transformation": "log(1+x)"
            },
            6: {
                "label": "households",
                "startdim": 6,
                "numdims": 1,
                "encoding": "cts.",
                "type": "cts.log1p",
                "lb": None, #original variable is positive, but log-transformed one is not
                "ub": None,
                "transformation": "log(1+x)"
            },
            7: {
                "label": "median_income",
                "startdim": 7,
                "numdims": 1,
                "encoding": "cts.",
                "type": "cts.log1p",
                "lb": None, #original variable is positive, but log-transformed one is not
                "ub": None,
                "transformation": "log(1+x)"
            }
        }

        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")


def preprocess_higgs_boson_dataset(load_dataset_path, save_dataset_path, preproc_owner):

    if preproc_owner=="SIAN2022":

        readable_labels = {
            0: 'lepton pT',
            1: 'lepton eta',
            2: 'lepton phi',
            3: 'missing energy magnitude',
            4: 'missing energy phi',
            5: 'jet 1 pt',
            6: 'jet 1 eta',
            7: 'jet 1 phi',
            8: 'jet 1 b-tag',
            9: 'jet 2 pt',
            10: 'jet 2 eta',
            11: 'jet 2 phi',
            12: 'jet 2 b-tag',
            13: 'jet 3 pt',
            14: 'jet 3 eta',
            15: 'jet 3 phi',
            16: 'jet 3 b-tag',
            17: 'jet 4 pt',
            18: 'jet 4 eta',
            19: 'jet 4 phi',
            20: 'jet 4 b-tag',
            21: 'm_jj',
            22: 'm_jjj',
            23: 'm_lv',
            24: 'm_jlv',
            25: 'm_bb',
            26: 'm_wbb',
            27: 'm_wwbb'
        }

        with open(load_dataset_path+'HIGGS.csv', newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            N = sum(1 for row in spamreader)
        
        all_higgs = np.zeros((N, 1+28))
        xd = 0

        with open(load_dataset_path+'HIGGS.csv', newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            for row in spamreader:
                higgs_features = [float(thing) for thing in row]
                
                all_higgs[xd, :] = higgs_features
                    
                if xd == 0:
                    pass
                    print(len(row))
                    for thing in row:
                        pass
                        print(thing)

                xd += 1
        print()
        print(xd)

        # means = [0.5299202727272727, 0.9914658435843994, -8.2976178820622e-06, -1.3272252572679215e-05, 0.9985363574312471, 2.6134592495411797e-05, 0.9909152318068567, -2.0275203997251415e-05, 7.71619920710906e-06, 0.9999687478206591, 0.9927294304430038, -1.0264440172703127e-05, -2.0768873493851226e-05, 1.0000080177052564, 0.9922590513707101, 1.459561349773536e-05, 3.678631990462732e-06, 1.0000114192497513, 0.9861086617144861, -5.756954065664269e-06, 1.7449033596108414e-05, 1.0000001559677123, 1.0342903040056053, 1.0248048350282475, 1.0505538681766282, 1.009741840750048, 0.972959616608593, 1.033035574431563, 0.9598119879373501]
        # varis = [0.49910397442703996, 0.5653776754096951, 1.0088264812855468, 1.006346283885119, 0.6000184644551814, 1.0063261640156402, 0.47497472589232176, 1.009302952852424, 1.0059010877868422, 1.0278075278204606, 0.49999384024846355, 1.0093304676767396, 1.0061543903728194, 1.049397999042849, 0.4876623258003873, 1.0087467092311453, 1.0063049450318349, 1.193675521568018, 0.5057776635500334, 1.0076942258109045, 1.0063655876039794, 1.4002093224446897, 0.6746353374867367, 0.38080739505009764, 0.16457624382242395, 0.39744529874617945, 0.5254062490071941, 0.3652556048435137, 0.3133377767062806]

        # all_higgs_adjusted = np.zeros((all_higgs.shape[0], 29))
        # all_higgs_adjusted[:, 0] = all_higgs[:, 0]
        # for i in range(1, 28+1):
        #     all_higgs_adjusted[:, i] = (all_higgs[:, i] - means[i-1]) / varis[i-1]
        
        all_higgs_adjusted =  all_higgs

        full_readable_labels = {
            -1: {"label": "signal/background",
                 "startdim": 0, "numdims": 1,
                 "encoding": "disc.ordinal", "type": "disc.categorical",
                 "min": 0, "count": 2,
                 "sublabels": ["background", "signal"],
                 },
            "task_type": "binary_classification",
            "D0": 28,
        }
        for i, label in enumerate(readable_labels.values()):
            full_readable_labels[i] = {
                "label": label,
                "startdim": i, "numdims": 1,
                "encoding": "cts.raw", "type": "cts.",
                "lb": None, "ub": None
            }

        XY_stuff = (all_higgs_adjusted[:, 1:], all_higgs_adjusted[:, 0], 0.8, True, 0)
        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff


# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
def preprocess_telco_customer_churn_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):
    
    if preproc_owner == "FIS2025":
        file_path = os.path.join(load_dataset_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        with open(file_path, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            header = next(reader)
            data = list(reader)

        N = len(data)
        feature_count = 40
        X = np.zeros((N, feature_count))
        Y = np.zeros(N)
        print("N",N)

        categorical_mappings = {
            'gender': {'Male': 1, 'Female': 0},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0},
            'Churn': {'No': 0, 'Yes': 1},
            'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2},
            'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
            'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'PaperlessBilling': {'Yes': 1, 'No': 0},
            'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 
                            'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3},

            'Churn': {'No': 0, 'Yes': 1} #prediction target
        }

        feature_indices = {
            'gender': 0,
            'SeniorCitizen': 1,
            'Partner': 2,
            'Dependents': 3,
            'tenure': 4,
            'PhoneService': 5,
            'MultipleLines': slice(6, 9),  # 3 dimensions
            'InternetService': slice(9, 12),
            'OnlineSecurity': slice(12, 15),
            'OnlineBackup': slice(15, 18),
            'DeviceProtection': slice(18, 21),
            'TechSupport': slice(21, 24),
            'StreamingTV': slice(24, 27),
            'StreamingMovies': slice(27, 30),
            'Contract': slice(30, 33),
            'PaperlessBilling': 33,
            'PaymentMethod': slice(34, 38),
            'MonthlyCharges': 38,
            'TotalCharges': 39
        }
        def onehotify(index, I):
            onehot_list = [0]*I
            onehot_list[index] = 1
            return onehot_list

        for i, row in enumerate(data):
            X[i, feature_indices['gender']] = categorical_mappings['gender'][row[1]]
            X[i, feature_indices['SeniorCitizen']] = int(row[2])
            X[i, feature_indices['Partner']] = categorical_mappings['Partner'][row[3]]
            X[i, feature_indices['Dependents']] = categorical_mappings['Dependents'][row[4]]
            
            tenure = float(row[5]) if row[5] else 0
            X[i, feature_indices['tenure']] = tenure / 72.0 # Normalize by max tenure
            
            X[i, feature_indices['PhoneService']] = categorical_mappings['PhoneService'][row[6]]
            X[i, feature_indices['MultipleLines']] = onehotify(categorical_mappings['MultipleLines'][row[7]], I=3)
            X[i, feature_indices['InternetService']] = onehotify(categorical_mappings['InternetService'][row[8]], I=3)
            X[i, feature_indices['OnlineSecurity']] = onehotify(categorical_mappings['OnlineSecurity'][row[9]], I=3)
            X[i, feature_indices['OnlineBackup']] = onehotify(categorical_mappings['OnlineBackup'][row[10]], I=3)
            X[i, feature_indices['DeviceProtection']] = onehotify(categorical_mappings['DeviceProtection'][row[11]], I=3)
            X[i, feature_indices['TechSupport']] = onehotify(categorical_mappings['TechSupport'][row[12]], I=3)
            X[i, feature_indices['StreamingTV']] = onehotify(categorical_mappings['StreamingTV'][row[13]], I=3)
            X[i, feature_indices['StreamingMovies']] = onehotify(categorical_mappings['StreamingMovies'][row[14]], I=3)
            X[i, feature_indices['Contract']] = onehotify(categorical_mappings['Contract'][row[15]], I=3)
            X[i, feature_indices['PaperlessBilling']] = categorical_mappings['PaperlessBilling'][row[16]]
            X[i, feature_indices['PaymentMethod']] = onehotify(categorical_mappings['PaymentMethod'][row[17]], I=4)
            
            monthly_charges = float(row[18]) if row[18] else 0
            X[i, feature_indices['MonthlyCharges']] = monthly_charges / 120.0 # Normalize by approximate max
            total_charges = float(row[19]) if row[19].strip() else 0
            X[i, feature_indices['TotalCharges']] = total_charges / 10000.0 # Normalize by approximate max
            
            Y[i] = categorical_mappings['Churn'][row[20]]


        full_readable_labels = {
            -1: {
                'label': 'Churn',
                'startdim': 0,
                'numdims': 1,
                'sublabels': ['No', 'Yes'],
            },
            'task_type': 'binary_classification',
            'D': 40,
            'D0': 19,

            0: {
                'label': 'gender',
                'startdim': 0,
                'numdims': 1,
                'encoding': 'disc.ordinal',
                'type': 'disc.categorical',
                'min': 0,
                'count': 2,
                'sublabels': ['Female', 'Male']
            },
            1: {
                'label': 'SeniorCitizen',
                'startdim': 1,
                'numdims': 1,
                'encoding': 'disc.ordinal',
                'type': 'disc.categorical',
                'min': 0,
                'count': 2,
                'sublabels': ['No', 'Yes']
            },
            2: {
                'label': 'Partner',
                'startdim': 2,
                'numdims': 1,
                'encoding': 'disc.ordinal',
                'type': 'disc.categorical',
                'min': 0,
                'count': 2,
                'sublabels': ['No', 'Yes']
            },
            3: {
                'label': 'Dependents',
                'startdim': 3,
                'numdims': 1,
                'encoding': 'disc.ordinal',
                'type': 'disc.categorical',
                'min': 0,
                'count': 2,
                'sublabels': ['No', 'Yes']
            },
            4: {
                'label': 'tenure',
                'startdim': 4,
                'numdims': 1,
                'encoding': 'cts.normalized',
                'type': 'cts.',
                'lb': 0,
                'ub': 72
            },
            5: {
                'label': 'PhoneService',
                'startdim': 5,
                'numdims': 1,
                'encoding': 'disc.ordinal',
                'type': 'disc.categorical',
                'min': 0,
                'count': 2,
                'sublabels': ['No', 'Yes']
            },
            6: {
                'label': 'MultipleLines',
                'startdim': 6,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['No', 'Yes', 'No phone service']
            },
            7: {
                'label': 'InternetService',
                'startdim': 9,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['No', 'DSL', 'Fiber optic']
            },
            8: {
                'label': 'OnlineSecurity',
                'startdim': 12,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['No', 'Yes', 'No internet service']
            },
            9: {
                'label': 'OnlineBackup',
                'startdim': 15,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['No', 'Yes', 'No internet service']
            },
            10: {
                'label': 'DeviceProtection',
                'startdim': 18,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['No', 'Yes', 'No internet service']
            },
            11: {
                'label': 'TechSupport',
                'startdim': 21,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['No', 'Yes', 'No internet service']
            },
            12: {
                'label': 'StreamingTV',
                'startdim': 24,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['No', 'Yes', 'No internet service']
            },
            13: {
                'label': 'StreamingMovies',
                'startdim': 27,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['No', 'Yes', 'No internet service']
            },
            14: {
                'label': 'Contract',
                'startdim': 30,
                'numdims': 3,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 3,
                'sublabels': ['Month-to-month', 'One year', 'Two year']
            },
            15: {
                'label': 'PaperlessBilling',
                'startdim': 33,
                'numdims': 1,
                'encoding': 'disc.ordinal',
                'type': 'disc.categorical',
                'min': 0,
                'count': 2,
                'sublabels': ['No', 'Yes']
            },
            16: {
                'label': 'PaymentMethod',
                'startdim': 34,
                'numdims': 4,
                'encoding': 'disc.onehot',
                'type': 'disc.categorical',
                'count': 4,
                'sublabels': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
            },
            17: {
                'label': 'MonthlyAmountCharged',
                'startdim': 38,
                'numdims': 1,
                'encoding': 'cts.normalized',
                'type': 'cts.',
                'lb': 0,
                'ub': 120
            },
            18: {
                'label': 'TotalAmountCharged',
                'startdim': 39,
                'numdims': 1,
                'encoding': 'cts.normalized',
                'type': 'cts.',
                'lb': 0,
                'ub': 10000
            }
        }
        readable_labels = get_readables_from_full_readables(full_readable_labels)


        XY_stuff = (X, Y, None, True, 0)
        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")



def process_abalone_dataset(load_dataset_path, save_dataset_path, preproc_owner):

    if preproc_owner=="FIS2025":
        file = open(load_dataset_path+'abalone.data','r')
        N = 4177 #total number of samples
        abalone_data = np.zeros((N,9))

        file.readline()
        i=0
        for line in file:
            values = line.replace('\n','').split(',')
            gender_list = ["F","M","I"]
            values[0] = gender_list.index(values[0])

            abalone_data[i] = [float(val) for val in values]
            i+=1
            
            
        onehot_genders = np.zeros((N,3))
        onehot_genders[np.arange(N),abalone_data[:,0].astype(int)] = 1
        abalone_data = np.concatenate([onehot_genders,abalone_data[:,1:]],axis=1)
        XY_stuff = (abalone_data[:,:10], abalone_data[:,10], None, True, 0)

        
        full_readable_labels = {
            -1 : {"label" : "Rings",
                "startdim" : 0, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.discrete",
                "lb" : 0.0, "ub" : None,
                "description" : "equivalent to years+1.5 (age of the abalone)"},
            'task_type' : "regression",
            "D0" : 8,

            0 : {"label" : "Sex",
                "startdim" : 0, "numdims" : 3,
                "encoding" : "disc.onehot", "type" : "discrete.categorical",
                "count" : 3, 
                "sublabels" : ["female","male","infant"],},
                
            1 : {"label" : "Length",
                "startdim" : 3, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None,
                "units" : "mm (over 200)",
                "description" : "Longest shell measurement"},
            2 : {"label" : "Diameter",
                "startdim" : 4, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None,
                "units" : "mm (over 200)",
                "description" : "perpendicular to length"},
            3 : {"label" : "Height",
                "startdim" : 5, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None,
                "units" : "mm (over 200)",
                "description" : "with meat in shell"},
            4 : {"label" : "Whole_weight",
                "startdim" : 6, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None,
                "units" : "grams (over 200)",
                "description" : "whole abalone"},
            5 : {"label" : "Shucked_weight",
                "startdim" : 7, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None,
                "units" : "grams (over 200)",
                "description" : "weight of meat"},
            6 : {"label" : "Viscera_weight",
                "startdim" : 8, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None,
                "units" : "grams (over 200)",
                "description" : "gut weight (after bleeding)"},
            7 : {"label" : "Shell_weight",
                "startdim" : 9, "numdims" : 1,
                "encoding" : "cts.raw", "type" : "cts.",
                "lb" : 0.0, "ub" : None,
                "units" : "grams (over 200)",
                "description" : "after being dried"},
        }
        readable_labels = get_readables_from_full_readables(full_readable_labels)
        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff
    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")





# https://archive.ics.uci.edu/dataset/332/online+news+popularity
def preprocess_online_news_popularity_dataset(load_dataset_path, save_dataset_path, preproc_owner):
    
    if preproc_owner == "FIS2025":


        raw_readable_labels = {            
			0 :  	 "timedelta",
			1 :  	 "n_tokens_title",
			2 :  	 "n_tokens_content",
			3 :  	 "n_unique_tokens",
			4 :  	 "n_non_stop_words",
			5 :  	 "n_non_stop_unique_tokens",
			6 :  	 "num_hrefs",
			7 :  	 "num_self_hrefs",
			8 :  	 "num_imgs",
			9 :  	 "num_videos",
			10 :  	 "average_token_length",
			11 :  	 "num_keywords",
			12 :  	 "data_channel_is_lifestyle",
			13 :  	 "data_channel_is_entertainment",
			14 :  	 "data_channel_is_bus",
			15 :  	 "data_channel_is_socmed",
			16 :  	 "data_channel_is_tech",
			17 :  	 "data_channel_is_world",
			18 :  	 "kw_min_min",
			19 :  	 "kw_max_min",
			20 :  	 "kw_avg_min",
			21 :  	 "kw_min_max",
			22 :  	 "kw_max_max",
			23 :  	 "kw_avg_max",
			24 :  	 "kw_min_avg",
			25 :  	 "kw_max_avg",
			26 :  	 "kw_avg_avg",
			27 :  	 "self_reference_min_shares",
			28 :  	 "self_reference_max_shares",
			29 :  	 "self_reference_avg_sharess",
			30 :  	 "weekday_is_monday",
			31 :  	 "weekday_is_tuesday",
			32 :  	 "weekday_is_wednesday",
			33 :  	 "weekday_is_thursday",
			34 :  	 "weekday_is_friday",
			35 :  	 "weekday_is_saturday",
			36 :  	 "weekday_is_sunday",
			37 :  	 "is_weekend",
			38 :  	 "LDA_00",
			39 :  	 "LDA_01",
			40 :  	 "LDA_02",
			41 :  	 "LDA_03",
			42 :  	 "LDA_04",
			43 :  	 "global_subjectivity",
			44 :  	 "global_sentiment_polarity",
			45 :  	 "global_rate_positive_words",
			46 :  	 "global_rate_negative_words",
			47 :  	 "rate_positive_words",
			48 :  	 "rate_negative_words",
			49 :  	 "avg_positive_polarity",
			50 :  	 "min_positive_polarity",
			51 :  	 "max_positive_polarity",
			52 :  	 "avg_negative_polarity",
			53 :  	 "min_negative_polarity",
			54 :  	 "max_negative_polarity",
			55 :  	 "title_subjectivity",
			56 :  	 "title_sentiment_polarity",
			57 :  	 "abs_title_subjectivity",
			58 :  	 "abs_title_sentiment_polarity",
			-1 :  	 "shares",
        }
        
        readable_labels = {
			0 :  	 "Time Since Original Publication (Days)",
			1 :  	 "n_tokens_title",
			2 :  	 "n_tokens_content",
			3 :  	 "n_unique_tokens",
			4 :  	 "n_non_stop_words",
			5 :  	 "n_non_stop_unique_tokens",
			6 :  	 "num_hrefs",
			7 :  	 "num_self_hrefs",
			8 :  	 "num_imgs",
			9 :  	 "num_videos",
			10 :  	 "average_token_length",
			11 :  	 "num_keywords",

			12 :  	 "Article Category (Data Channel)", #["Lifestyle", "Entertainment", "Business", "Social Media", "Technology", "World"]
            
			13 :  	 "kw_min_min",
			14 :  	 "kw_max_min",
			15 :  	 "kw_avg_min",
			16 :  	 "kw_min_max",
			17 :  	 "kw_max_max",
			18 :  	 "kw_avg_max",
			19 :  	 "kw_min_avg",
			20 :  	 "kw_max_avg",
			21 :  	 "kw_avg_avg",
			22 :  	 "self_reference_min_shares",
			23 :  	 "self_reference_max_shares",
			24 :  	 "self_reference_avg_sharess",

			25 :  	 "Day of Week", #["Monday", ...]
			26 :  	 "Is Weekend?",

			27 :  	 "LDA_00",
			28 :  	 "LDA_01",
			29 :  	 "LDA_02",
			30 :  	 "LDA_03",
			31 :  	 "LDA_04",
			32 :  	 "global_subjectivity",
			33 :  	 "global_sentiment_polarity",
			34 :  	 "global_rate_positive_words",
			35 :  	 "global_rate_negative_words",
			36 :  	 "rate_positive_words",
			37 :  	 "rate_negative_words",
			38 :  	 "avg_positive_polarity",
			39 :  	 "min_positive_polarity",
			40 :  	 "max_positive_polarity",
			41 :  	 "avg_negative_polarity",
			42 :  	 "min_negative_polarity",
			43 :  	 "max_negative_polarity",
			44 :  	 "title_subjectivity",
			45 :  	 "title_sentiment_polarity",
			46 :  	 "abs_title_subjectivity",
			47 :  	 "abs_title_sentiment_polarity",
        }

        N = 39797
        all_news = np.zeros((N, 60))
        row_count = 0

        with open(load_dataset_path + 'OnlineNewsPopularity/OnlineNewsPopularity.csv', newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            next(spamreader)
            for row in spamreader:
                news_features = [float(thing.strip()) for thing in row[1:]]
                if len(news_features) != 60:
                    raise ValueError(f"Expected 60 attributes per row, got {len(news_features)}")
                all_news[row_count, :] = news_features
                
                row_count += 1
                if row_count % 10000 == 0:
                    print(f"Processed {row_count // 1000:d} thousand rows")
        print()
        print(row_count)

        if np.any(np.isnan(all_news)) or np.any(np.isinf(all_news)):
            raise ValueError("Dataset contains NaN or Inf values")
        
        shares = all_news[:, -1]
        shares_log = np.log1p(shares)
        

        
        full_readable_labels = {
            -1: {
                "label": "shares",
                "startdim": 0,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": "cts.log1p",
                "transform" : "log(1+x)",
                "units" : "shares (on twitter)",
                "lb": None,
                "ub": None
            },
            "task_type": "regression",
            "D0": 48,
        }
        
        totaldims=0
        for i in readable_labels:
            label=readable_labels[i]

            currdim = 1
            full_readable_labels[i] = {
                "label": label,
                "startdim": totaldims,
                "numdims": currdim,
                "encoding": "cts.raw",
                "type": "cts.",
                "lb": None,
                "ub": None
            }

            if i in list(range(1,10)): #discrete values in the sense of positive integer values
                full_readable_labels[i]["type"] = "cts.discrete"
                full_readable_labels[i]["lb"] = 0.0
            
            if i==12: # onehot article category
                currdim=6
                full_readable_labels[i] = {
                    "label": label,
                    "startdim": totaldims,
                    "numdims": currdim,
                    "encoding": "disc.onehot",
                    "type": "disc.categorical",
                    "count" : currdim,
                    "sublabels" : ["Lifestyle", "Entertainment", "Business", "Social Media", "Technology", "World"],
                }

            if i==25: # onehot day_of_week
                currdim=7
                full_readable_labels[i] = {
                    "label": label,
                    "startdim": totaldims,
                    "numdims": currdim,
                    "encoding": "disc.onehot",
                    "type": "disc.ordinal",
                    "count" : currdim,
                    "sublabels" : ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                }

            if i==26: # binary is_weekend
                currdim=1
                full_readable_labels[i] = {
                    "label": label,
                    "startdim": totaldims,
                    "numdims": currdim,
                    "encoding": "disc.ordinal",
                    "type": "disc.ordinal",
                    "count" : 2,
                    "sublabels" : ["No (workday)", "Yes (weekend)",],
                }


            totaldims+=currdim
            pass

        print('all_news',all_news.shape)
        XY_stuff = (all_news[:, :59], shares_log, None, True, 0)
        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff

    else:
        raise Exception("Preprocessing owner \"" + preproc_owner + "\" has no preprocessing pipeline.")







# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
def preprocess_credit_card_fraud_dataset(load_dataset_path, save_dataset_path, preproc_owner):

    if preproc_owner == "FIS2025":

        readable_labels = {
            0: 'Time',
            # 1: 'V1',
            #   .... 
            # 28: 'V28',
            # 29: 'Amount'
        }
        for i in range(28):
            readable_labels[i+1] = "V"+str(i+1)
        readable_labels[29] = "Amount"


        N = 284807
        all_transactions = np.zeros((N, 31))
        row_count = 0

        with open(load_dataset_path + 'creditcard.csv', newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            next(spamreader)
            for row in spamreader:
                transaction_features = [float(thing.strip()) for thing in row]
                if len(transaction_features) != 31:
                    raise ValueError(f"Expected 31 attributes per row, got {len(transaction_features)}")
                all_transactions[row_count, :] = transaction_features
                
                # if row_count == 0:
                #     print(len(row))
                #     for thing in row:
                #         print(thing)

                row_count += 1
                if row_count % 50000 == 0:
                    print(f"Processed {row_count / 1000:.1f}k rows")
        # print()
        print(row_count)

        if np.any(np.isnan(all_transactions)) or np.any(np.isinf(all_transactions)):
            raise ValueError("Dataset contains NaN or Inf values")


        all_transactions_adjusted = np.zeros((all_transactions.shape[0], 30))
        all_transactions_adjusted[:, :] = all_transactions[:, :-1]
        all_transactions_adjusted[:, 0] = all_transactions[:, 0] / 60. / 60. #time between transactions in hours not seconds
        all_transactions_adjusted[:, 29] = np.log(1.0+100*all_transactions[:, 29])


        target = all_transactions[:, -1]
        print(f"Target mean: {np.mean(target):.6f}, Target unique values: {np.unique(target)}")

        if np.any(np.isnan(target)) or np.any(np.isinf(target)):
            raise ValueError("Target contains NaN or Inf values")

        full_readable_labels = {
            -1: {
                "label": "transaction fraud label",
                "startdim" : 0, "numdims" : 1,
                "encoding" : "disc.ordinal", "type" : "disc.ordinal",
                "min" : 0, "count" : 2,
                "sublabels" : ['nonfraudulent', 'fraudulent'],
            },
            "task_type": "binary_classification",
            "D0": 30,
        }
        for i, label in enumerate(readable_labels.values()):
            full_readable_labels[i] = {
                "label": label,
                "startdim": i,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": "cts.",
                "lb": None,
                "ub": None
            }
        full_readable_labels[0]['units'] = "hours"
        full_readable_labels[0]['lb'] = 0.0
        full_readable_labels[0]['ub'] = 48.0
        full_readable_labels[29]['transformation'] = "log1p"
        full_readable_labels[29]['units'] = "log(1 + cents)"

        XY_stuff = (all_transactions_adjusted, target, None, True, 0)
        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff

    else:
        raise Exception("Preprocessing owner \"" + preproc_owner + "\" has no preprocessing pipeline.")





# https://www.kaggle.com/datasets/ishadss/eucalyptus
# https://ml.cms.waikato.ac.nz/publications/1996/Thomson-McQueen-96.pdf
def preprocess_eucalyptus_dataset(load_dataset_path, save_dataset_path, preproc_owner):

    if preproc_owner == "FIS2025":

        readable_labels = { #NOTE: these first few categorical variables should be encoded categorically instead of as ordinals and the same goes for "PMCno" (seedlot number) which if I am understanding correctly might even be an ID and so instead of onehot encoding it like the original author suggests, it might be better to just drop it
            0: 'Abbrev',
            1: 'Rep',
            2: 'Locality',
            3: 'Map_Ref',
            4: 'Latitude',
            5: 'Altitude',
            6: 'Rainfall',
            7: 'Frosts',
            8: 'Year',
            9: 'Sp',
            10: 'PMCno',
            11: 'DBH',
            12: 'Ht',
            13: 'Surv',
            14: 'Vig',
            15: 'Ins_res',
            16: 'Stem_Fm',
            17: 'Crown_Fm',
            18: 'Brnch_Fm'
        }

        N = 736 #later some entries are dropped
        all_data = np.zeros((N, 20))
        row_count = 0

        abbrev_map = {'Cra': 0, 'Cly': 1, 'Nga': 2, 'Wai': 3, 'K81': 4, 'Wak': 5, 'K82': 6, 'WSp': 7, 'K83': 8, 'Lon': 9, 'Puk': 10, 'Paw': 11, 'K81a': 12, 'Mor': 13, 'Wen': 14, 'WSh': 15}
        locality_map = {'Central_Hawkes_Bay': 0, 'Northern_Hawkes_Bay': 1, 'Southern_Hawkes_Bay': 2, 'Central_Hawkes_Bay_(coastal)': 3, 'Central_Wairarapa': 4, 'South_Wairarapa': 5, 'Southern_Hawkes_Bay_(coastal)': 6, 'Central_Poverty_Bay': 7}
        map_ref_map = {'N135_382/137': 0, 'N116_848/985': 1, 'N145_874/586': 2, 'N142_377/957': 3, 'N158_344/626': 4, 'N162_081/300': 5, 'N158_343/625': 6, 'N151_912/221': 7, 'N162_097/424': 8, 'N166_063/197': 9, 'N146_273/737': 10, 'N141_295/063': 11, 'N98_539/567': 12, 'N151_922/226': 13}
        #latitude_map = {'39__38': 0, '39__00': 1, '40__11': 2, '39__50': 3, '40__57': 4, '41__12': 5, '40__36': 6, '41__08': 7, '41__16': 8, '40__00': 9, '39__43': 10, '82__32': 11}
        sp_map = {'nd': 0, 're': 1, 'ov': 2, 'fa': 3, 'fr': 4, 'ob': 5, 'am': 6, 'pu': 7, 'rd': 8, 'ni': 9, 'br': 10, 'co': 11, 'ka': 12, 'bxs': 13, 'sm': 14, 'te': 15, 'el': 16, 'cr': 17, 'si': 18, 'jo': 19, 'ag': 20, 'pa': 21, 'ra': 22, 'nc': 23, 'ma': 24, 'mn': 25, 'ro': 26}
        utility_map = {'none': 0, 'low': 1, 'average': 2, 'good': 3, 'best': 4}
        def latitude_mapper(lat_string):
            lat_parts = lat_string.split("__")
            newlat = float(lat_parts[0]) + float(lat_parts[1])/60.
            return newlat

        with open(load_dataset_path + 'dataset_194_eucalyptus.csv', newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            next(spamreader)
            for row in spamreader:
                row = [item for item in row if item]
                if not row:
                    continue
                if len(row) != 20:
                    raise ValueError(f"Expected 20 attributes per row, got {len(row)}")
                
                features = []
                features.append(abbrev_map[row[0].strip()])
                features.append(float(row[1].strip()) if row[1].strip() != '?' else np.nan)
                features.append(locality_map[row[2].strip()])
                features.append(map_ref_map[row[3].strip()])
                features.append(latitude_mapper(row[4].strip()))
                features.append(float(row[5].strip()) if row[5].strip() != '?' else np.nan)
                features.append(float(row[6].strip()) if row[6].strip() != '?' else np.nan)
                features.append(float(row[7].strip()) if row[7].strip() != '?' else np.nan)
                features.append(float(row[8].strip()) if row[8].strip() != '?' else np.nan)
                features.append(sp_map[row[9].strip()])
                features.append(float(row[10].strip()) if row[10].strip() != '?' else np.nan)
                features.append(float(row[11].strip()) if row[11].strip() != '?' else np.nan)
                features.append(float(row[12].strip()) if row[12].strip() != '?' else np.nan)
                features.append(float(row[13].strip()) if row[13].strip() != '?' else np.nan)
                features.append(float(row[14].strip()) if row[14].strip() != '?' else np.nan)
                features.append(float(row[15].strip()) if row[15].strip() != '?' else np.nan)
                features.append(float(row[16].strip()) if row[16].strip() != '?' else np.nan)
                features.append(float(row[17].strip()) if row[17].strip() != '?' else np.nan)
                features.append(float(row[18].strip()) if row[18].strip() != '?' else np.nan)
                features.append(utility_map[row[19].strip()])

                if features[1]==22: #assumed to be a typo because this happens only a single time
                    features[1] = 2 

                all_data[row_count, :] = features

                if row_count == 0:
                    print(len(row))
                    for thing in row:
                        print(thing)

                row_count += 1

        print()
        print(row_count)
        #dropping the last 11 entries because: the 6 "Wen" entries have clearly wrong latitudes and the 5 "WSh" entries are the only ones with this abbreviation despite other "N151_922/226" entries
        N = 725 
        all_data = all_data[:N]

        #numerical_indices = [1, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        #categorical_indices = [0, 2, 3, 4, 9]
        numerical_indices = [4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        categorical_indices = [0, 1, 2, 3, 9]

        means = np.nanmean(all_data[:, numerical_indices], axis=0)
        for i, idx in enumerate(numerical_indices):
            all_data[np.isnan(all_data[:, idx]), idx] = means[i]

        raw_stds = np.std(all_data[:, numerical_indices], axis=0)
        print(f"Original numerical feature stds: {raw_stds}")
        if np.any(raw_stds == 0):
            print(f"Numerical features with zero std: {np.where(raw_stds == 0)[0]}")
        stds = np.where(raw_stds == 0, 1.0, raw_stds)

        all_data_adjusted = np.copy(all_data[:, :19])
        for i, idx in enumerate(numerical_indices):
            all_data_adjusted[:, idx] = (all_data_adjusted[:, idx] - means[i]) / stds[i]

        print(f"Adjusted numerical features mean: {np.mean(all_data_adjusted[:, numerical_indices], axis=0)}")
        print(f"Adjusted numerical features std: {np.std(all_data_adjusted[:, numerical_indices], axis=0)}")

        target = all_data[:, 19]
        print(f"Target unique values: {np.unique(target)}")

        import matplotlib.pyplot as plt
        for d in range(19):
            plt.title(readable_labels[d])
            plt.hist(all_data[:,d],bins=100)
            plt.yscale('log')
            plt.show()


        if np.any(np.isnan(target)) or np.any(np.isinf(target)):
            raise ValueError("Target contains NaN or Inf values")

        full_readable_labels = {
            -1: {
                "label": "Utility (for Soil Conservation)",
                "startdim": 0,
                "numdims": 1,
                "encoding": "categorical",
                "type": "categorical",
                "lb": 0,
                "ub": 4,
                "classes": ["none", "low", "average", "good", "best"]
            },
            "task_type": "multiclass_classification",
            "D0": 19,
        }
        
        for i in readable_labels:
            label = readable_labels[i]

            feature_type = "disc.ordinal" if i in categorical_indices else "cts."
            encoding = "disc.ordinal" if i in categorical_indices else "cts.raw"
            full_readable_labels[i] = {
                "label": label,
                "startdim": i,
                "numdims": 1,
                "encoding": encoding,
                "type": feature_type,
                "lb": None,
                "ub": None
            }

        XY_stuff = (all_data_adjusted[:, :], target, None, True, 0)
        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff

    else:
        raise Exception("Preprocessing owner \"" + preproc_owner + "\" has no preprocessing pipeline.")





# https://www.microsoft.com/en-us/research/project/mslr/
def preprocess_Microsoft_dataset(load_dataset_path, save_dataset_path, preproc_owner):
    if preproc_owner=="FIS2025":
        file1 = open(load_dataset_path+'MSLR-WEB10K/Fold1/train.txt','r')
        file2 = open(load_dataset_path+'MSLR-WEB10K/Fold1/vali.txt','r')
        file3 = open(load_dataset_path+'MSLR-WEB10K/Fold1/test.txt','r')

        OG_TRN_N = 723412
        OG_VAL_N = 235259
        OG_TST_N = 241521
        N = OG_TRN_N + OG_VAL_N + OG_TST_N
        microsoft_data = np.zeros((N,136+1))

        def load_microsoft_line(line):
            values = line.strip().replace('\n','').split(' ')
            
            relevance = values[0]
            #qid = values[1] #drop the query id from original data
            
            newvals =  [(dictval.split(":")[1]) for dictval in values[2:]]
            newvals.append(relevance)
            # print(newvals)
            return newvals


        i=0
        for line in file1:
            microsoft_data[i] = load_microsoft_line(line)
            i+=1
        for line in file2:
            microsoft_data[i] = load_microsoft_line(line)
            i+=1
        for line in file3:
            microsoft_data[i] = load_microsoft_line(line)
            i+=1

        trnval_ratio = (OG_TRN_N+OG_VAL_N) / N
        XY_stuff = (microsoft_data[:,:136], microsoft_data[:,136], trnval_ratio, False, None)


        NLP_DM_feature_description_list = [
            "covered_query_term_number",
            "covered_query_term_ratio",
            "stream_length",
            "IDF(Inverse_document_frequency)",

            # "{statistic}_of_term_frequency"
            "sum_of_term_frequency",
            "min_of_term_frequency",
            "max_of_term_frequency",
            "mean_of_term_frequency",
            "variance_of_term_frequency",

            # "{statistic}_of_stream_length_normalized_term_frequency"
            "sum_of_stream_length_normalized_term_frequency",
            "min_of_stream_length_normalized_term_frequency",
            "max_of_stream_length_normalized_term_frequency",
            "mean_of_stream_length_normalized_term_frequency",
            "variance_of_stream_length_normalized_term_frequency",

            # "{statistic}_of_tf*idf"
            "sum_of_tf*idf",
            "min_of_tf*idf",
            "max_of_tf*idf",
            "mean_of_tf*idf",
            "variance_of_tf*idf",
            
            "boolean_model",
            "vector_space_model",
            "BM25",
            "LMIR_ABS",
            "LMIR_DIR",
            "LMIR_JM",
        ]

        stream_list = [
            "body",
            "anchor",
            "title",
            "url",
            "whole_document",
        ]

        extra_feature_list = [
            "Length_of_URL",
            "Inlink_number",
            "Outlink_number",
            "PageRank",
            "SiteRank (Site level PageRank)",
            "Site_level_PageRank",
            "QualityScore",
            "QualityScore2",
            "queryXurl_click_count",
            "url_click_count",
            "url_dwell_time",
        ]



        full_readable_labels = {
            -1: {
                "label": "query x URL relevance - 0 (irrelevant) through 4 (perfectly relevant)",
                "startdim": 0, "numdims": 1,
                "encoding": "cts.raw", "type": "cts.disc",
            },
            "task_type": "regression",
            "D0": 136,
        }

        feat_id = 0
        for feat_description in NLP_DM_feature_description_list:
            for stream_source in stream_list:
                label = feat_description + "." + stream_source

                label_info = {
                    "label": label,
                    "startdim": feat_id,
                    "numdims": 1,
                    "encoding": "cts.raw", "type": "cts.",
                    "lb": None, "ub": None,
                }
                full_readable_labels[feat_id] = label_info
                feat_id += 1
                pass
        print('feat_id',feat_id)
        for label in extra_feature_list:
            label_info = {
                "label": label,
                "startdim": feat_id,
                "numdims": 1,
                "encoding": "cts.raw", "type": "cts.",
                "lb": None, "ub": None,
            }
            full_readable_labels[feat_id] = label_info
            feat_id += 1
        print('feat_id',feat_id)
        
        readable_labels = get_readables_from_full_readables(full_readable_labels)

        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff

    else:
        raise Exception("Preprocessing owner \""+preproc_owner+"\" has no preprocessing pipeline.")





def preprocess_madelon_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):

    if preproc_owner == "FIS2025":

        train_data = np.loadtxt(os.path.join(load_dataset_path, 'madelon/MADELON/madelon_train.data'))
        train_labels = (np.loadtxt(os.path.join(load_dataset_path, 'madelon/MADELON/madelon_train.labels')) + 1 )/2  #{-1,1} -> {0,1}
        

        valid_data = np.loadtxt(os.path.join(load_dataset_path, 'madelon/MADELON/madelon_valid.data'))
        valid_labels = (np.loadtxt(os.path.join(load_dataset_path, 'madelon/madelon_valid.labels')) + 1 )/2
        

        #feature_means = np.mean(train_data, axis=0) #NOTE: moving to dataset object
        #train_data = train_data - feature_means
        #valid_data = valid_data - feature_means
        

        fullX = np.vstack([train_data, valid_data])
        fullY = np.concatenate([train_labels, valid_labels])
        
        # 2000 train samples and 600 val samples. 2600 samples in total. split is 2000/2600 ≈ 0.77 
        trnval_ratio = 2000/2600 #0.77
        XY_stuff = (fullX, fullY, trnval_ratio, False, 0)  #I think it's not a big deal to shuffle instead because it is a synthetic dataset
        

        readable_labels = {i: f"feature_{i}" for i in range(500)}
        
        full_readable_labels = {
            -1: {
                "label": "class",
                "startdim": 0,
                "numdims": 1,
                "encoding": "disc.ordinal",
                "type": "disc.categorical",
                "sublabels": ["-1", "1"],
                "count": 2
            },
            "task_type": "binary_classification",
            "D0": 500
        }
        for i in range(500):
            full_readable_labels[i] = {
                "label": f"X{i+1}",
                "startdim": i,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": "cts.",
                "lb": 0,
                "ub": 999,
            }
            import matplotlib.pyplot as plt
            plt.title(full_readable_labels[i]['label'])
            plt.hist(fullX[:,i],bins=100)
            plt.yscale('log')
            plt.show()
        
        label_stuff = (readable_labels, full_readable_labels)

        return XY_stuff, label_stuff
    else:
        raise Exception(f"Preprocessing owner \"{preproc_owner}\" has no preprocessing pipeline.")


# https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
def preprocess_taiwanese_bankruptcy_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):

    if preproc_owner == "FIS2025":

        data = np.loadtxt(os.path.join(load_dataset_path, 'taiwanese+bankruptcy+prediction/data.csv'), delimiter=',', skiprows=1)
        labels = data[:, 0]

        # Exclude Net Income Flag -- a useless flag which is always =1
        feature_indices = list(range(1, 94)) + [95]  # Columns 1 to 93, plus 95 (skip 94)
        features = data[:, feature_indices]
        log_transform_cols = []
        for i in range(features.shape[1]):
            if np.max(np.abs(features[:, i])) > 1e6 or np.min(np.abs(features[:, i][features[:, i] != 0])) < 1e-4:
                log_transform_cols.append(i)

        for i in log_transform_cols:
            features[:, i] = np.log1p(np.abs(features[:, i])) * np.sign(features[:, i])
        print('log_transform_cols',log_transform_cols)

        fullX = features
        fullY = labels
        XY_stuff = (fullX, fullY, None, True, 0)
        num_features = 94

        readable_labels = {
            0: 'ROA(C) before interest and depreciation before interest',
            1: 'ROA(A) before interest and % after tax',
            2: 'ROA(B) before interest and depreciation after tax',
            3: 'Operating Gross Margin',
            4: 'Realized Sales Gross Margin',
            5: 'Operating Profit Rate',
            6: 'Pre-tax net Interest Rate',
            7: 'After-tax net Interest Rate',
            8: 'Non-industry income and expenditure/revenue',
            9: 'Continuous interest rate (after tax)',
            10: 'Operating Expense Rate',
            11: 'Research and development expense rate',
            12: 'Cash flow rate',
            13: 'Interest-bearing debt interest rate',
            14: 'Tax rate (A)',
            15: 'Net Value Per Share (B)',
            16: 'Net Value Per Share (A)',
            17: 'Net Value Per Share (C)',
            18: 'Persistent EPS in the Last Four Seasons',
            19: 'Cash Flow Per Share',
            20: 'Revenue Per Share (Yuan ¥)',
            21: 'Operating Profit Per Share (Yuan ¥)',
            22: 'Per Share Net profit before tax (Yuan ¥)',
            23: 'Realized Sales Gross Profit Growth Rate',
            24: 'Operating Profit Growth Rate',
            25: 'After-tax Net Profit Growth Rate',
            26: 'Regular Net Profit Growth Rate',
            27: 'Continuous Net Profit Growth Rate',
            28: 'Total Asset Growth Rate',
            29: 'Net Value Growth Rate',
            30: 'Total Asset Return Growth Rate Ratio',
            31: 'Cash Reinvestment %',
            32: 'Current Ratio',
            33: 'Quick Ratio',
            34: 'Interest Expense Ratio',
            35: 'Total debt/Total net worth',
            36: 'Debt ratio %',
            37: 'Net worth/Assets',
            38: 'Long-term fund suitability ratio (A)',
            39: 'Borrowing dependency',
            40: 'Contingent liabilities/Net worth',
            41: 'Operating profit/Paid-in capital',
            42: 'Net profit before tax/Paid-in capital',
            43: 'Inventory and accounts receivable/Net value',
            44: 'Total Asset Turnover',
            45: 'Accounts Receivable Turnover',
            46: 'Average Collection Days',
            47: 'Inventory Turnover Rate (times)',
            48: 'Fixed Assets Turnover Frequency',
            49: 'Net Worth Turnover Rate (times)',
            50: 'Revenue per person',
            51: 'Operating profit per person',
            52: 'Allocation rate per person',
            53: 'Working Capital to Total Assets',
            54: 'Quick Assets/Total Assets',
            55: 'Current Assets/Total Assets',
            56: 'Cash/Total Assets',
            57: 'Quick Assets/Current Liability',
            58: 'Cash/Current Liability',
            59: 'Current Liability to Assets',
            60: 'Operating Funds to Liability',
            61: 'Inventory/Working Capital',
            62: 'Inventory/Current Liability',
            63: 'Current Liabilities/Liability',
            64: 'Working Capital/Equity',
            65: 'Current Liabilities/Equity',
            66: 'Long-term Liability to Current Assets',
            67: 'Retained Earnings to Total Assets',
            68: 'Total income/Total expense',
            69: 'Total expense/Assets',
            70: 'Current Asset Turnover Rate',
            71: 'Quick Asset Turnover Rate',
            72: 'Working capitcal Turnover Rate',
            73: 'Cash Turnover Rate',
            74: 'Cash Flow to Sales',
            75: 'Fixed Assets to Assets',
            76: 'Current Liability to Liability',
            77: 'Current Liability to Equity',
            78: 'Equity to Long-term Liability',
            79: 'Cash Flow to Total Assets',
            80: 'Cash Flow to Liability',
            81: 'CFO to Assets',
            82: 'Cash Flow to Equity',
            83: 'Current Liability to Current Assets',
            84: 'Liability-Assets Flag',
            85: 'Net Income to Total Assets',
            86: 'Total assets to GNP price',
            87: 'No-credit Interval',
            88: 'Gross Profit to Sales',
            89: 'Net Income to Stockholder\'s Equity',
            90: 'Liability to Equity',
            91: 'Degree of Financial Leverage (DFL)',
            92: 'Interest Coverage Ratio (Interest expense to EBIT)',
            93: 'Equity to Liability'
        }

        full_readable_labels = {
            -1: {
                "label": "class",
                "startdim": 0,
                "numdims": 1,
                "encoding": "disc.categorical",
                "type": "disc.categorical",
                "sublabels": ["0", "1"],
                "count": 2
            },
            "task_type": "binary_classification",
            "D0": num_features
        }

        for i in range(num_features):
            cts_type = "cts.log_standardized" if i in log_transform_cols else "cts.standardized"
            transform = "sign(x) * log1p(|x|)" if i in log_transform_cols else None
            full_readable_labels[i] = {
                "label": readable_labels[i] + "".join([" (log)"]*(i in log_transform_cols)),
                "startdim": i,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": cts_type,
                "transform": transform,
                "lb": None,
                "ub": None,
            }


        label_stuff = (readable_labels, full_readable_labels)
        
        return XY_stuff, label_stuff
    else:
        raise Exception(f"Preprocessing owner \"{preproc_owner}\" has no preprocessing pipeline.")








def preprocess_mnist_dataset(load_dataset_path, save_dataset_path, preproc_owner=None):

    if preproc_owner == "FIS2025":

        #NOTE: source is original lecun source

        def load_mnist_images(filepath):
            with open(filepath, "rb") as f:
                bytes4 = f.read(4)
                int4 = int.from_bytes(bytes4, byteorder='big')
                assert int4 == 2051 #magic number
                bytes4 = f.read(4)
                N = int.from_bytes(bytes4, byteorder='big')
                bytes4 = f.read(4)
                H = int.from_bytes(bytes4, byteorder='big')
                bytes4 = f.read(4)
                W = int.from_bytes(bytes4, byteorder='big')

                all_mnist_images = np.zeros((N,H*W),dtype=int)
                for n in range(N):
                    for y in range(H):
                        for x in range(W):
                            byte = f.read(1)
                            int1 = int.from_bytes(byte, byteorder='big')
                            all_mnist_images[n,y*W+x] = int1
            return all_mnist_images

        def load_mnist_labels(filepath):
            with open(filepath, "rb") as f:
                bytes4 = f.read(4)
                int4 = int.from_bytes(bytes4, byteorder='big')
                assert int4 == 2049 #magic number
                bytes4 = f.read(4)
                N = int.from_bytes(bytes4, byteorder='big') #N=10k or 60k (test or train)
                
                all_mnist_labels = np.zeros((N),dtype=int)
                for n in range(N):
                            byte = f.read(1)
                            int1 = int.from_bytes(byte, byteorder='big')
                            #print('int1',int1)
                            all_mnist_labels[n] = int1
            return all_mnist_labels


        OG_TRN_N = 60*1000
        OG_TST_N = 10*1000
        H = 28
        W = 28


        N = OG_TRN_N + OG_TST_N
        mnist_images = np.zeros((N,784),dtype=int)
        mnist_labels = np.zeros((N,),dtype=int)

        trn_images_path = load_dataset_path + "MNIST/raw/" + "train-images-idx3-ubyte"
        tst_images_path = load_dataset_path + "MNIST/raw/" + "t10k-images-idx3-ubyte"
        trn_labels_path = load_dataset_path + "MNIST/raw/" + "train-labels-idx1-ubyte"
        tst_labels_path = load_dataset_path + "MNIST/raw/" + "t10k-labels-idx1-ubyte"
        mnist_trn_images = load_mnist_images( trn_images_path )
        mnist_tst_images = load_mnist_images( tst_images_path )
        mnist_trn_labels = load_mnist_labels( trn_labels_path )
        mnist_tst_labels = load_mnist_labels( tst_labels_path )
        mnist_images[:OG_TRN_N] = mnist_trn_images
        mnist_images[-OG_TST_N:] = mnist_tst_images
        mnist_labels[:OG_TRN_N] = mnist_trn_labels
        mnist_labels[-OG_TST_N:] = mnist_tst_labels

        mnist_images_rescaled = mnist_images.astype(float) / 255.0
        mnist_labels_onehot = np.zeros((N,10),dtype=int)
        mnist_labels_onehot[np.arange(N),mnist_labels] = 1
        
        trnval_ratio = (OG_TRN_N) / N
        XY_stuff = (mnist_images_rescaled, mnist_labels_onehot, trnval_ratio, False, None)
        num_features = H*W





        readable_labels = {}
        for y in range(H):
            for x in range(W):
                readable_labels[y*W+x] = "x_{" + f"y{y},x{x}" + "}" #matrix notation for coordinates


        full_readable_labels = {
            -1: {
                "label": "digit class",
                "startdim": 0,
                "numdims": 10,
                "encoding": "disc.onehot",
                "type": "disc.categorical",
                "sublabels" : [str(i) for i in list(range(10))],
                "count": 10
            },
            "task_type" : "multiclass_classification",
            "D0": num_features
        }

        for i in range(num_features):
            full_readable_labels[i] = {
                "label": readable_labels[i],
                "startdim": i,
                "numdims": 1,
                "encoding": "cts.raw",
                "type": "cts.",
                "transform": "pixel intensity / 255.0",
                "lb": 0.0,
                "ub": 1.0,
            }

        label_stuff = (readable_labels, full_readable_labels)
        
        return XY_stuff, label_stuff
    else:
        raise Exception(f"Preprocessing owner \"{preproc_owner}\" has no preprocessing pipeline.")






