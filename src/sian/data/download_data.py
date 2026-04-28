
import subprocess






def download_dataset(dataset_id, load_dataset_path="data/"):

    # datetimestr = gettimestamp()
    # header_dict = {
    #     "Preprocessed Datetime" : datetimestr, 
    #     "dataset_id" : dataset_id,
    #     "preproc_owner" : preproc_owner,
    #     "load_dataset_path" : load_dataset_path,
    #     "save_dataset_path" : save_dataset_path,
    # }


    downloading_function = None
    if dataset_id == "UCI_275_bike_sharing_dataset":
        downloading_function = download_bike_sharing_dataset
    elif dataset_id == "UCI_186_wine_quality":
        downloading_function = download_wine_dataset
    elif dataset_id == "UCI_374_appliances_energy_prediction":
        downloading_function = download_energy_dataset
    elif dataset_id == "UCI_2_adults_dataset":
        downloading_function = download_adults_dataset
    elif dataset_id == "UCI_31_tree_cover_type_dataset":
        downloading_function = download_treecover_dataset
    elif dataset_id == "otherSource_cal_housing": 
        downloading_function = download_california_housing_dataset
    elif dataset_id == "UCI_203_yearpredictionMSD":
        downloading_function = download_song_year_dataset
    elif dataset_id == "UCI_280_higgs_boson_dataset":
        downloading_function = download_higgs_boson_dataset
    elif dataset_id == "UCI_1_abalone_dataset":
        downloading_function = download_abalone_dataset
    # elif dataset_id == "Kaggle_blastchar_telco_customer_churn":
    #     downloading_function = preprocess_telco_customer_churn_dataset
    elif dataset_id == "UCI_332_online_news_popularity":
        downloading_function = download_online_news_popularity_dataset
    # elif dataset_id == "Kaggle_mlgulb_credit_card_fraud_dataset":
    #     downloading_function = preprocess_credit_card_fraud_dataset
    # elif dataset_id == "Kaggle_ishadss_eucalyptus_dataset":
    #     downloading_function = preprocess_eucalyptus_dataset
    elif dataset_id == "otherSource_Microsoft_search_queries":
        downloading_function = download_microsoft_dataset
    elif dataset_id == "UCI_171_madelon_dataset":
        downloading_function = download_madelon_dataset
    elif dataset_id == "UCI_572_taiwanese_bankruptcy_dataset":
        downloading_function = download_taiwanese_bankruptcy_dataset
    else:
        raise NotImplementedError(f'preprocessing for dataset_id=\"{dataset_id}\" not supported right now -- possibly dataset object changes')
    if True:
        pass
    else:
        raise Exception(f"dataset_id not found\"{dataset_id}\"")

    downloading_function(load_dataset_path)


def wget_helper(download_path, download_url):
    try:
        subprocess.run(["wget", "-P", download_path, download_url], check=True) 
    except FileNotFoundError:
        print("wget command not found. Please ensure wget is installed and in your system's PATH.")

def unzip_helper(zip_path, unzip_path):
    try:
        subprocess.run(["unzip", zip_path, "-d", unzip_path], check=True) 
    except FileNotFoundError:
        print("unzip command not found. Please ensure unzip is installed and in your system's PATH.")


def untar_helper(tar_path, untar_path):
    try:
        subprocess.run(["tar", "-xvzf", tar_path, "-C", untar_path], check=True) 
    except FileNotFoundError:
        print("tar command not found. Please ensure tar is installed and in your system's PATH.")

def gunzip_helper(gzip_path):
    try:
        subprocess.run(["gunzip", gzip_path], check=True) 
    except FileNotFoundError:
        print("gunzip command not found. Please ensure unzip is installed and in your system's PATH.")










def download_california_housing_dataset(load_dataset_path):
    url = "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz"
    
    wget_helper(load_dataset_path, url)
    untar_helper(load_dataset_path+"cal_housing.tgz", load_dataset_path)
    
    try:
        subprocess.run(["mv", load_dataset_path+"CaliforniaHousing/"+"cal_housing.data", load_dataset_path], check=True) 
        subprocess.run(["mv", load_dataset_path+"CaliforniaHousing/"+"cal_housing.domain", load_dataset_path], check=True) 
        subprocess.run(["rmdir", load_dataset_path+"CaliforniaHousing/"], check=True) 
        subprocess.run(["rm", load_dataset_path+"cal_housing.tgz"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")


def download_bike_sharing_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"bike+sharing+dataset.zip", load_dataset_path)

    try:
        subprocess.run(["mv", load_dataset_path+"Readme.txt",load_dataset_path+"bikeshare_readme.txt"], check=True) 
        subprocess.run(["rm", load_dataset_path+"bike+sharing+dataset.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")
        
    
def download_treecover_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"covertype.zip", load_dataset_path)
    gunzip_helper(load_dataset_path+"covtype.data.gz")
    try:
        subprocess.run(["rm", load_dataset_path+"covertype.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")


def download_higgs_boson_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/280/higgs.zip"

    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"higgs.zip", load_dataset_path)
    gunzip_helper(load_dataset_path+"HIGGS.csv.gz")
    try:
        subprocess.run(["rm", load_dataset_path+"higgs.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")

def download_song_year_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"

    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"yearpredictionmsd.zip", load_dataset_path)
    try:
        subprocess.run(["rm", load_dataset_path+"yearpredictionmsd.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")

def download_microsoft_dataset(load_dataset_path):
    url = "https://www.microsoft.com/en-us/research/project/mslr/"
    url = "https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78"    

    raise Exception("Microsoft does not want you to download this dataset automatically >_<, please visit https://www.microsoft.com/en-us/research/project/mslr/")

    
def download_wine_dataset(load_dataset_path):
    ###url = "https://archive.ics.uci.edu/static/public/109/wine.zip" #heads up, not this wine dataset
    url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"wine+quality.zip", load_dataset_path)
    try:
        subprocess.run(["rm", load_dataset_path+"wine+quality.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")
            
def download_energy_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"appliances+energy+prediction.zip", load_dataset_path)
    try:
        subprocess.run(["rm", load_dataset_path+"appliances+energy+prediction.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")
            
def download_adults_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"adult.zip", load_dataset_path)
    try:
        subprocess.run(["rm", load_dataset_path+"Index"], check=True) 
        subprocess.run(["rm", load_dataset_path+"adult.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")
            
def download_abalone_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/1/abalone.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"abalone.zip", load_dataset_path)
    try:
        subprocess.run(["rm", load_dataset_path+"Index"], check=True) 
        subprocess.run(["rm", load_dataset_path+"abalone.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")
        
def download_online_news_popularity_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"online+news+popularity.zip", load_dataset_path)
    try:
        subprocess.run(["rm", load_dataset_path+"online+news+popularity.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")
        
def download_madelon_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/171/madelon.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"madelon.zip", load_dataset_path+"madelon/")
    try:
        subprocess.run(["rm", load_dataset_path+"madelon/"+"Dataset.pdf"], check=True) 
        subprocess.run(["rm", load_dataset_path+"madelon.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")
        
def download_taiwanese_bankruptcy_dataset(load_dataset_path):
    url = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
    
    wget_helper(load_dataset_path, url)
    unzip_helper(load_dataset_path+"taiwanese+bankruptcy+prediction.zip", load_dataset_path)
    try:
        subprocess.run(["mkdir", load_dataset_path+"taiwanese+bankruptcy+prediction/"], check=True) 
        subprocess.run(["mv", load_dataset_path+"data.csv", load_dataset_path+"taiwanese+bankruptcy+prediction/"+"data.csv"], check=True) 
        subprocess.run(["rm", load_dataset_path+"taiwanese+bankruptcy+prediction.zip"], check=True) 
    except FileNotFoundError:
        print("Error while moving files around.")

        
        
        
# def download_XXXXXXXX_dataset(load_dataset_path):
#     url = ""
    
#     wget_helper(load_dataset_path, url)
#     unzip_helper(load_dataset_path+"xxxxxxxxx.zip", load_dataset_path)
#     try:
#         subprocess.run(["rm", load_dataset_path+"xxxxxxxxx.zip"], check=True) 
#     except FileNotFoundError:
#         print("Error while moving files around.")
        
# def download_XXXXXXXX_dataset(load_dataset_path):
#     url = ""
    
#     wget_helper(load_dataset_path, url)
#     unzip_helper(load_dataset_path+"xxxxxxxxx.zip", load_dataset_path)
#     try:
#         subprocess.run(["rm", load_dataset_path+"xxxxxxxxx.zip"], check=True) 
#     except FileNotFoundError:
#         print("Error while moving files around.")

        
        




