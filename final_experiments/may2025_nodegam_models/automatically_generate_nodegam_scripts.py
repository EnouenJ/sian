import os

def generate_slurm_script(model_name, dataset_name, preproc_owner, job_partition='gpu', 
                        conda_env_path='/home1/rahilpar/sian'):
    job_name = f"{model_name}__{dataset_name}__seed2025_2029"

    # Assign preproc_owner based on dataset
    if dataset_name in ["UCI_275_bike_sharing_dataset", "UCI_186_wine_quality", 
                       "UCI_374_appliances_energy_prediction", "otherSource_cal_housing"]:
        preproc_owner = "SIAN2022"
    elif dataset_name in ["UCI_2_adults_dataset", "UCI_31_tree_cover_type_dataset"]:
        preproc_owner = "InstaSHAP2025"
    elif dataset_name in ["UCI_1_abalone_dataset", "Kaggle_blastchar_telco_customer_churn",
                        "UCI_332_online_news_popularity", "Kaggle_mlgulb_credit_card_fraud_dataset",
                        "Kaggle_ishadss_eucalyptus_dataset", "otherSource_Microsoft_search_queries",
                        "UCI_171_madelon_dataset", "UCI_572_taiwanese_bankruptcy_dataset"]:
        preproc_owner = "FIS2025"
    else:
        preproc_owner = "FIS2025"  # Default value

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={job_partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

module purge

eval "$(conda shell.bash hook)"

conda activate {conda_env_path}

python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 0 --seed 2025
python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 1 --seed 2025

python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 0 --seed 2026
python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 1 --seed 2026

python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 0 --seed 2027
python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 1 --seed 2027

python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 0 --seed 2028
python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 1 --seed 2028

python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 0 --seed 2029
python nodegam_baseline.py --dataset_str "{dataset_name}" --preproc_owner "{preproc_owner}" --IS_GA2M 1 --seed 2029
"""
    
    filename = f"shell_scripts/{job_name}.sh"
    with open(filename, "w") as f:
        f.write(script)
    print(f"Generated SLURM script: {filename}")

datasets = [
    "UCI_275_bike_sharing_dataset",
    "UCI_186_wine_quality",
    "UCI_374_appliances_energy_prediction",
    "UCI_2_adults_dataset",
    "UCI_31_tree_cover_type_dataset",
    "otherSource_cal_housing",
    "UCI_280_higgs_boson_dataset",
    "UCI_203_yearpredictionMSD",
    "UCI_1_abalone_dataset",
    "Kaggle_blastchar_telco_customer_churn",
    "UCI_332_online_news_popularity",
    "Kaggle_mlgulb_credit_card_fraud_dataset",
    "Kaggle_ishadss_eucalyptus_dataset",
    "otherSource_Microsoft_search_queries",
    "UCI_171_madelon_dataset",
    "UCI_572_taiwanese_bankruptcy_dataset"
]

model_name = "NodeGAM"

for dataset_name in datasets:
    generate_slurm_script(
        model_name=model_name,
        dataset_name=dataset_name,
        preproc_owner="Auto_assign",
    )