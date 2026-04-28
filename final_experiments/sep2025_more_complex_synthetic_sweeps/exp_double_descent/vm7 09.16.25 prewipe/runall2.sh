#!/usr/bin/env bash


# https://stackoverflow.com/questions/79297468/how-to-write-a-gpu-worker-pool-to-run-multiple-tasks-at-the-same-time-in-bash

mkfifo free_gpus2
exec 3<>free_gpus2

# write the GPUs you want to use (0,1,2,3) into the fifo.
printf '%s\n' 0 1 2 3 >&3  &


allscripts=(
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS32_N566.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N566.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS32_N673.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N673.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS32_N800.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N800.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS32_N951.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N951.sh"




#runall2.sh

# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N566.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N673.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N800.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS128_N951.sh"


#512 and l1shape = 5.0e-4
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N566.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N673.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N800.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N951.sh"



#512 and l1shape = 5.0e-4 and LR=1.0e-2
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N566.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N673.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N800.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N951.sh"


#Turning L1 off to see the peak immediately; even gam1 isnt showing grokking yet

# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N566.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N673.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N800.sh"
# "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N951.sh"



#GCLOUD 09/16/25 @ 10:00pm -- doing larger N sweep for GAM2 on unregularized path
"shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N951.sh"
"shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N1131.sh"
"shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N1345.sh"
"shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0_POnone_MLPFalse_SIANFalse_MNISTsmooth_FISmaximal_K2_BS512_N1600.sh"


)


for exe in ${allscripts[@]}
do
    read gpu
    sleep 3
    { printf "GPU ${gpu}  for ${exe}\n" >> "allscripts_logs.txt"; CUDA_VISIBLE_DEVICES="$gpu" "$exe"; echo "$gpu" >&3; } &
done <&3
wait

# rm free_gpus
