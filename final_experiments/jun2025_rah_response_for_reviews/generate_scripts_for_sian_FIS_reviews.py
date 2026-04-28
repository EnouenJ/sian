import itertools
import numpy as np
import os

dataset_preproc_pairs = [
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed1", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed2", None),

    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s4.16.64_seed0", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s4.16.64_seed1", None),
    ("SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s4.16.64_seed2", None),

    ("SYNTH_simple_discrete_synthwave_v9_D20_Ik5_s3.9.27_seed0", None),
    ("SYNTH_simple_discrete_synthwave_v9_D20_Ik5_s3.9.27_seed1", None),
    ("SYNTH_simple_discrete_synthwave_v9_D20_Ik5_s3.9.27_seed2", None),
]

# masked_combinations = [(True, True), (False, False)]  # (is_masked_mlp, is_masked_sian)
masked_combinations = [(False, False)]  #JAM: turn off while we debug masking
# FIS_style_list = ["batchwise", "layerwise"]  # Updated to include both styles
FIS_style_list = ["batchwise"]
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

def generate_slurm_script(dataset_str, preproc_owner, is_masked_mlp, is_masked_sian, FIS_style, MAX_K, number_of_rounds, inters_per_round, job_name):
    preproc_owner_str = preproc_owner if preproc_owner is not None else "None"
    # Include number_of_rounds in the command only for batchwise
    if FIS_style == "batchwise":
        if inters_per_round is None:
            raise ValueError(f"inters_per_round is None for MAX_K={MAX_K}. Check inters_per_round_list_dict.")
        command = f'python -u sian_python_for_reviews.py --dataset_str "{dataset_str}" --preproc_owner "{preproc_owner_str}" --is_masked_mlp "{is_masked_mlp}" --is_masked_sian "{is_masked_sian}" --FIS_style "{FIS_style}" --MAX_K "{MAX_K}" --number_of_rounds "{number_of_rounds}" --inters_per_round "{inters_per_round}"'
    else:
        command = f'python -u sian_python_for_reviews.py --dataset_str "{dataset_str}" --preproc_owner "{preproc_owner_str}" --is_masked_mlp "{is_masked_mlp}" --is_masked_sian "{is_masked_sian}" --FIS_style "{FIS_style}" --MAX_K "{MAX_K}"'
    
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

# Generate all combinations for each dataset
count_number_of_jobs = 0
for dataset_str, preproc_owner in dataset_preproc_pairs:
    for (is_masked_mlp, is_masked_sian), FIS_style, MAX_K in itertools.product(
        masked_combinations, FIS_style_list, MAX_K_list
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
                    job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'None'}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_FIS{FIS_style}_K{MAX_K}_Rounds{number_of_rounds}_IntersPerRound{inters_per_round}"
                    script_content = generate_slurm_script(
                        dataset_str=dataset_str,
                        preproc_owner=preproc_owner,
                        is_masked_mlp=is_masked_mlp,
                        is_masked_sian=is_masked_sian,
                        FIS_style=FIS_style,
                        MAX_K=MAX_K,
                        number_of_rounds=number_of_rounds,
                        inters_per_round=inters_per_round,
                        job_name=job_name
                    )
                    filename = f"shell_scripts/slurm_{job_name}.sh"
                    with open(filename, "w") as f:
                        f.write(script_content)
                    print(f'"shell_scripts/{filename}"')
                    count_number_of_jobs += 1
        else:
            raise NotImplementedError("layerwise not implemented right now")
            # job_name = f"{dataset_str}_PO{preproc_owner if preproc_owner else 'None'}_MLP{is_masked_mlp}_SIAN{is_masked_sian}_FIS{FIS_style}_K{MAX_K}"
            # script_content = generate_slurm_script(
            #     dataset_str=dataset_str,
            #     preproc_owner=preproc_owner,
            #     is_masked_mlp=is_masked_mlp,
            #     is_masked_sian=is_masked_sian,
            #     FIS_style=FIS_style,
            #     MAX_K=MAX_K,
            #     number_of_rounds=None,  # Not used for layerwise
            #     job_name=job_name
            # )
            # filename = f"slurm_{job_name}.sh"
            # with open(filename, "w") as f:
            #     f.write(script_content)
            # print(f'"../notebooks/hyperparameter_sweep/{filename}"')
            # count_number_of_jobs += 1

print(f"Total number of jobs generated: {count_number_of_jobs}")