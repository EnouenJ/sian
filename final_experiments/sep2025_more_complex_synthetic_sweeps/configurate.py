from dotenv import load_dotenv
import os
import sys
import json



def setup_config():
    load_dotenv()

    CONFIGURATION_FLAG = os.getenv("CONFIGURATION_FLAG")

    if CONFIGURATION_FLAG == "james":
        pass
    elif CONFIGURATION_FLAG == "rahil" or CONFIGURATION_FLAG == "gcloud":
        sys.path.append(os.path.abspath("../../src"))
    else:
        raise ValueError(f"Got configuration_flag={CONFIGURATION_FLAG} instead of 'james' or 'rahil'")

    return CONFIGURATION_FLAG

def setup_debugging_config():
    load_dotenv()

    DEBUGGING_FLAG = os.getenv("DEBUGGING_FLAG")

    if DEBUGGING_FLAG in ["debugging", "on", "true"]:
        return True
    elif DEBUGGING_FLAG in ["not", "off", "false"]:
        return False
    else:
        raise ValueError(f"Got debugging_flag={DEBUGGING_FLAG} instead of 'on' or 'off'")

    return DEBUGGING_FLAG




CONFIGURATION_FLAG = setup_config()
DEBUGGING_FLAG = setup_debugging_config()
PLOTTING_FLAG = True #TODO


class CommonConfig:
    shell_save_dir = "shell_scripts/"
    data_base_path = "data/"
    USE_LONG_JOB_NAME = True


class CommonDatasetConfigs:

    # dataset_preproc_pairs

    realworld_debugging =  [
        ("UCI_275_bike_sharing_dataset", None),
    ]
    
    realworld_demo =  [
        ("UCI_275_bike_sharing_dataset", None),
        ("UCI_31_tree_cover_type_dataset", None),
        ("UCI_2_adults_dataset", None),
    ]

    realworld_all_datasets = [
        ("UCI_275_bike_sharing_dataset", None),
        ("otherSource_cal_housing", None),
        ("UCI_186_wine_quality", None),
        ("UCI_374_appliances_energy_prediction", None),
        
        ("UCI_31_tree_cover_type_dataset", None),
        ("UCI_203_yearpredictionMSD", None),
        ("UCI_280_higgs_boson_dataset", None),
        
        ("UCI_2_adults_dataset", None),
        ("UCI_1_abalone_dataset", None),
        ("UCI_332_online_news_popularity", None),
        ("UCI_171_madelon_dataset", None),
        ("UCI_572_taiwanese_bankruptcy_dataset", None),

        ("otherSource_Microsoft_search_queries", None),
        ("Kaggle_blastchar_telco_customer_churn", None),
        ("Kaggle_mlgulb_credit_card_fraud_dataset", None),
        ("Kaggle_ishadss_eucalyptus_dataset", None),
    ]

    synthetic_sweep_BS_LR_and_tester_datasets = [
        ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0", None),
        ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed1", None),
        ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed2", None),
        ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed3", None),
        ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed4", None),
    ]

    check_inflation_dataset = [
        ("otherSource_MNIST", "FIS2025"),
    ]

    FIS_compare_dataset = [
        ("SYNTH_simple_discrete_synthwave_v12_D9_Ik5_depth1_s5.5.5_conditionalHookerANOVA_strongestheredity_seed0", None),
    ]



def generate_slurm_script(shell_save_dir, json_path, py_script_name):
    if CONFIGURATION_FLAG=="rahil":
        command = f'python -u {py_script_name} --experiment_details_file_path "{json_path}"'
        
        base_script = f"""#!/bin/bash
#SBATCH --job-name={json_path}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=19:00:00

module purge

eval "$(conda shell.bash hook)"

conda activate /home1/rahilpar/sian

{command}
"""
    if CONFIGURATION_FLAG=="gcloud":
        command = f'python3 -u {py_script_name} --experiment_details_file_path "{json_path}"'
        
        base_script = f"""#!/bin/bash
        

{command}
"""
    if CONFIGURATION_FLAG=="james":
        command = f'python -u {py_script_name} --experiment_details_file_path "{json_path}"'
        
        base_script = f"""#!/bin/bash
        

{command}
"""

    return base_script

def save_json_and_shell(shell_save_dir, json_data, job_name, py_script_name='train_sian_models.py'):
    json_path = shell_save_dir + job_name + ".json"
    with open(json_path, "w") as file:
        json.dump(json_data, file, indent=4)

    script_content = generate_slurm_script(shell_save_dir=shell_save_dir, json_path=json_path, py_script_name=py_script_name)
    
    shell_path = shell_save_dir + f"slurm_{job_name}.sh"
    with open(shell_path, "w") as file:
        file.write(script_content)
    print(f'"{shell_path}"')





