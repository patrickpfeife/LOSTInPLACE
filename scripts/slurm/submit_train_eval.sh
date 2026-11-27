#!/bin/bash
# Training on the GPU, then the evaluation on dev (s06b) on the CPU as soon as the training has finished.
# usage (from any folder):
#   ./submit_train_eval.sh          single run with the ARGS of 06_train_model.slurm
#   ./submit_train_eval.sh grid     grid search over 06_hparam_grid.txt, one report for all runs
set -euo pipefail

# the slurm scripts use paths relative to this folder (cd ../python)
cd "$(dirname "$0")"

if [[ "${1:-}" == "grid" ]]; then
    # one array task per line of the grid file
    N=$(grep -c . 06_hparam_grid.txt)
    T=$(sbatch --parsable --array=0-$((N - 1)) 06_train_model.slurm)
    # afterany: the evaluation also starts if single grid tasks failed, s06b skips incomplete runs
    sbatch --dependency=afterany:$T 06b_baseline_comparison.slurm "grid_${T}_*"
else
    T=$(sbatch --parsable 06_train_model.slurm)
    sbatch --dependency=afterany:$T 06b_baseline_comparison.slurm "single_${T}"
fi

echo "training job $T submitted, the evaluation waits for it"
