#!/bin/bash
# SLURM_DIR="../notebooks/hyperparameter_sweep"
LOG_FILE="slurm_submission_log.txt"
echo "Starting SLURM job submissions at $(date)" > "$LOG_FILE"
SUBMITTED=0
SLURM_SCRIPTS=(
    "shell_scripts/slurm_SYNTH_simple_discrete_synthwave_v11_D9_Ik10_depth0_s3.3.3_conditionalHookerANOVA_seed0_PONone_MLPFalse_SIANFalse_MNISTFalse_FISbatchwise_K1_Rounds3_IntersPerRound1.sh"
)
for SCRIPT in "${SLURM_SCRIPTS[@]}"; do
    if [[ -f "$SCRIPT" ]]; then
        echo "Submitting $SCRIPT..." | tee -a "$LOG_FILE"
        JOB_ID=$(sbatch "$SCRIPT" | awk '{print $4}')
        if [[ -n "$JOB_ID" ]]; then
            echo "Submitted $SCRIPT with Job ID: $JOB_ID" | tee -a "$LOG_FILE"
            ((SUBMITTED++))
        else
            echo "Error submitting $SCRIPT" | tee -a "$LOG_FILE"
        fi
            sleep 1
    else
        echo "Script not found: $SCRIPT" | tee -a "$LOG_FILE"
    fi
done
echo "Submission complete. Total jobs submitted: $SUBMITTED" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"