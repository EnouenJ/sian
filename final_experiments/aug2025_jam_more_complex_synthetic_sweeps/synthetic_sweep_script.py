# please write code here for sweeping across these datasets
import itertools
import numpy as np
import os

dataset_preproc_pairs = [
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik5_depth0_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik5_depth1_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik5_depth2_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik10_depth0_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik10_depth1_s3.9.27_conditionalHookerANOVA_seed0", None),
    # ("SYNTH_simple_discrete_synthwave_v11_D9_Ik10_depth2_s3.9.27_conditionalHookerANOVA_seed0", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed1", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed2", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed3", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed4", None),
]

# masked_combinations = [(True, True), (False, False)]  # (is_masked_mlp, is_masked_sian)
masked_combinations = [(False, False), (True, True)]  #JAM: turn off while we debug masking
# FIS_style_list = ["batchwise", "layerwise"]  # Updated to include both styles
use_mnist_scaling = [True, False]
use_mnist_scaling = ["mnist", "smooth"] #08/30/2025 @ 12:30am
# FIS_style_list = ["batchwise", "layerwise"]
FIS_style_list = ["batchwise", "layerwise", "maximal"] #08/21/2025 @ 5:30pm
FIS_style_list = ["batchwise", "maximal"]    #08/21/2025 @ 5:30pm
FIS_style_list = ["maximal"]    #08/21/2025 @ 6:00pm -- actually this version is fine for the one where we just update the results



masked_combinations = [(False, False), ]
use_mnist_scaling = ["smooth", ]
FIS_style_list = ["maximal"] 
MAX_K_list = [1, 2, 3]
# number_of_rounds_list = [3, 6, 9]  # For batchwise only - off now
number_of_rounds_list_dict = {
    1 : [3, 6, 9],
    2 : [3, 6, 9],
    3 : [3, 6, 9],
}
inters_per_round_list_dict = {
    1 : [1],
    2 : [3],
    3 : [10],
}

def generate_slurm_script(dataset_str, preproc_owner, is_masked_mlp, is_masked_sian, FIS_style, MAX_K, number_of_rounds, inters_per_round, use_mnist_scaling, job_name):
    preproc_owner_str = preproc_owner if preproc_owner is not None else "None"
    # Include number_of_rounds and use_mnist_scaling in the command only for batchwise
    if FIS_style == "batchwise":
        if inters_per_round is None:
            raise ValueError(f"inters_per_round is None for MAX_K={MAX_K}. Check inters_per_round_list_dict.")
        command = f'python -u train_sian_models.py --dataset_str "{dataset_str}" --preproc_owner "{preproc_owner_str}" --is_masked_mlp "{is_masked_mlp}" --is_masked_sian "{is_masked_sian}" --use_mnist_scaling "{use_mnist_scaling}" --FIS_style "{FIS_style}" --MAX_K "{MAX_K}" --number_of_rounds "{number_of_rounds}" --inters_per_round "{inters_per_round}"'
    else:
        command = f'python -u train_sian_models.py --dataset_str "{dataset_str}" --preproc_owner "{preproc_owner_str}" --is_masked_mlp "{is_masked_mlp}" --is_masked_sian "{is_masked_sian}" --use_mnist_scaling "{use_mnist_scaling}" --FIS_style "{FIS_style}" --MAX_K "{MAX_K}"'
    
    base_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

module purge

eval "$(conda shell.bash hook)"

conda activate /home1/rahilpar/sian

{command}
"""
    return base_script

shell_save_dir = "shell_scripts/"
if not os.path.exists(shell_save_dir):
    os.mkdir(shell_save_dir)
# Generate all combinations for each dataset
count_number_of_jobs = 0
for dataset_str, preproc_owner in dataset_preproc_pairs:
    for (is_masked_mlp, is_masked_sian), FIS_style, MAX_K, use_mnist in itertools.product(
        masked_combinations, FIS_style_list, MAX_K_list, use_mnist_scaling
    ):
        if FIS_style == "batchwise":
            number_of_rounds_list = number_of_rounds_list_dict.get(MAX_K, [])
            inters_per_round_list = inters_per_round_list_dict.get(MAX_K, [])
            if not number_of_rounds_list:
                raise ValueError(f"No number_of_rounds defined for MAX_K={MAX_K} in number_of_rounds_list_dict.")
            if not inters_per_round_list:
                raise ValueError(f"No inters_per_round defined for MAX_K={MAX_K} in inters_per_round_list_dict.")
            
            for number_of_rounds in number_of_rounds_list:
                for inters_per_round in inters_per_round_list:
                    job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}_Rounds{number_of_rounds}_IntersPerRound{inters_per_round}"
                    script_content = generate_slurm_script(
                        dataset_str=dataset_str,
                        preproc_owner=preproc_owner,
                        is_masked_mlp=is_masked_mlp,
                        is_masked_sian=is_masked_sian,
                        use_mnist_scaling=use_mnist,
                        FIS_style=FIS_style,
                        MAX_K=MAX_K,
                        number_of_rounds=number_of_rounds,
                        inters_per_round=inters_per_round,
                        job_name=job_name
                    )
                    filename = shell_save_dir + f"slurm_{job_name}.sh"
                    with open(filename, "w") as f:
                        f.write(script_content)
                    print(f'"{filename}"')
                    count_number_of_jobs += 1
        elif FIS_style == "batchwise":
            job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}"
            script_content = generate_slurm_script(
                dataset_str=dataset_str,
                preproc_owner=preproc_owner,
                is_masked_mlp=is_masked_mlp,
                is_masked_sian=is_masked_sian,
                use_mnist_scaling=use_mnist,
                FIS_style=FIS_style,
                MAX_K=MAX_K,
                number_of_rounds=None,
                inters_per_round=None,
                job_name=job_name
            )
            filename = shell_save_dir + f"slurm_{job_name}.sh"
            with open(filename, "w") as f:
                f.write(script_content)
            print(f'"{filename}"')
            count_number_of_jobs += 1
        elif FIS_style == "maximal": #exact same as batchwise which has no hypers right now
            job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'none'}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_MNIST{use_mnist}_FIS{FIS_style}_K{MAX_K}"
            script_content = generate_slurm_script(
                dataset_str=dataset_str,
                preproc_owner=preproc_owner,
                is_masked_mlp=is_masked_mlp,
                is_masked_sian=is_masked_sian,
                use_mnist_scaling=use_mnist,
                FIS_style=FIS_style,
                MAX_K=MAX_K,
                number_of_rounds=None,
                inters_per_round=None,
                job_name=job_name
            )
            filename = shell_save_dir + f"slurm_{job_name}.sh"
            with open(filename, "w") as f:
                f.write(script_content)
            print(f'"{filename}"')
            count_number_of_jobs += 1
            pass
        else:
            raise NotImplementedError(f"FIS_style={FIS_style} not implemented right now")

print(f"Total number of jobs generated: {count_number_of_jobs}")