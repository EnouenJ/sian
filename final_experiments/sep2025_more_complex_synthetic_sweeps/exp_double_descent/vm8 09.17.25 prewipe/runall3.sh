#!/usr/bin/env bash


# https://stackoverflow.com/questions/79297468/how-to-write-a-gpu-worker-pool-to-run-multiple-tasks-at-the-same-time-in-bash

mkfifo free_gpus3
exec 3<>free_gpus3

# write the GPUs you want to use (0,1,2,3) into the fifo.
printf '%s\n' 4 5 6 7 >&3  &


allscripts=(
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS32_N566.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N566.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS32_N673.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N673.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS32_N800.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N800.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS32_N951.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N951.sh"


# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K3_BS128_N6400.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K3_BS128_N7611.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K3_BS128_N9051.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K3_BS128_N10763.sh"


#lambda1 is zero
"shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K3_BS128_N6400.sh"
"shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K3_BS128_N7611.sh"
"shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K3_BS128_N9051.sh"
"shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K3_BS128_N10763.sh"
)


for exe in ${allscripts[@]}
do
    read gpu
    sleep 3
    sleep 15
    { printf "GPU ${gpu}  for ${exe}\n" >> "allscripts_logs.txt"; CUDA_VISIBLE_DEVICES="$gpu" "$exe"; echo "$gpu" >&3; } &
done <&3
wait

# rm free_gpus
