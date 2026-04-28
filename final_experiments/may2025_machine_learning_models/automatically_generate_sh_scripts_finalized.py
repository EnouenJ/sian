import os

def generate_slurm_script(model_name, dataset_name, preproc_owner, time_limit, job_partition='gpu', conda_env_path='/home1/rahilpar/sian', optuna_trials=100, use_optuna=True, seed_list=None):
    model_name_safe = model_name.replace(" ", "")
    #job_name = f"{model_name_safe}__{dataset_name}__use_optuna{use_optuna}__seed{seed}"
    job_name = f"{model_name_safe}__{dataset_name}__use_optuna{use_optuna}__seed{int(min(seed_list))}_{int(max(seed_list))}"

    # if dataset_name == "UCI_275_bike_sharing_dataset" or dataset_name == "UCI_186_wine_quality" or dataset_name == "UCI_374_appliances_energy_prediction" or dataset_name == "otherSource_cal_housing":
    #     preproc_owner = "SIAN2022"
    # elif dataset_name == "UCI_2_adults_dataset" or dataset_name == "UCI_31_tree_cover_type_dataset":
    #     preproc_owner = "InstaSHAP2025"
    # elif dataset_name == "UCI_1_abalone_dataset" or dataset_name == "Kaggle_blastchar_telco_customer_churn" or dataset_name == "UCI_332_online_news_popularity" or dataset_name == "Kaggle_mlgulb_credit_card_fraud_dataset" or dataset_name == "Kaggle_ishadss_eucalyptus_dataset" or dataset_name == "otherSource_Microsoft_search_queries" or dataset_name == "UCI_171_madelon_dataset" or dataset_name == "UCI_572_taiwanese_bankruptcy_dataset":
    #     preproc_owner = "FIS2025"
    # else:
    #     pass

    if model_name == "XGB" or model_name == "LightGBM":
        time_limit = "15:00:00"
    elif model_name == "Random Forest" or model_name == "CatBoost" or model_name == "EBM":
        time_limit = "24:00:00"
    elif model_name == "SVM":
        time_limit = "48:00:00"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={job_partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time={time_limit}

module purge

eval "$(conda shell.bash hook)"

conda activate {conda_env_path}

cd ..

"""
    for use_optuna in [False, True]:
        for seed in seed_list:
            final_optuna_trials = optuna_trials
            if not use_optuna:
                final_optuna_trials = 0
            script += f"""python machine_learning_model_general_script.py --model_to_train "{model_name}" --dataset_str "{dataset_name}" --preproc_owner "{None}" --optuna_n_trials {final_optuna_trials} --use_optuna {use_optuna} --seed {seed}\n"""
    
    filename = f"shell_scripts/{job_name}.sh"
    with open(filename, "w") as f:
        f.write(script)
    #print(f"Generated script: {filename}")


for model_name in ["XGB", "LightGBM", "Lasso", "Random Forest", "CatBoost", "EBM", "SVM"]:
    for dataset_name in [
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
    ]:  
        curryear = 2025
        seed_list = list(range(curryear,curryear+5))
        generate_slurm_script(
            model_name=model_name,
            dataset_name=dataset_name,
            preproc_owner="Auto_assign",
            time_limit="Auto_assign",
            seed_list=seed_list,
        )

# generate_slurm_script(
#     model_name="Random Forest",
#     dataset_name="Kaggle_blastchar_telco_customer_churn",
#     preproc_owner="Auto_assign",
#     time_limit="Auto_assign"
# )