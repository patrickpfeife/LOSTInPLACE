#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=../logs/train_%A_%a.out
#SBATCH --error=../logs/train_%A_%a.err
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=hpda-c
#SBATCH --qos=normal

# One script for both modes:
#   single run:   sbatch 04_train_model.slurm
#                 -> trains once with the ARGS below
#   grid search:  sbatch --array=0-$(( $(grep -c . 04_hparam_grid.txt) - 1 )) 04_train_model.slurm
#                 -> array task i trains with the ARGS below plus line i+1 of 04_hparam_grid.txt
#                    (the grid arguments come last and override the ones below, argparse keeps the last value)
# %A = job id, %a = array task index (shows as 4294967294 for a single run)

set -euo pipefail

cd ../python

module load uv

# the arguments every run shares; in the grid search the varied ones are overridden per task
ARGS=(
    --project-dir="/dss/dsstbyfs02/scratch/0C/di54haf/LOSTInPLACE"   # the weights go to <project-dir>/checkpoints/<run-name>/
    --n-graphs=30000      # train pools drawn at random (leave out for all ~188k)
    --k-graph=3           # every node of a drawn subset is connected to its k nearest nodes
    --subset-min=0.4      # smallest share of a pool's points drawn per epoch
    --subset-max=0.9      # largest share of a pool's points drawn per epoch
    --min-graph-size=7    # pools with fewer points are skipped, no subset gets smaller
    --hidden-dim=96       # width of the node embeddings
    --n-layers=2          # number of message passing layers
    --heads=4             # attention heads per GATv2 layer (hidden-dim must be divisible by it)
    --dropout=0.1         # dropout rate in the message passing layers
    --lr=3e-4             # learning rate of AdamW
    --weight-decay=1e-4   # weight decay of AdamW
    --batch-size=128       # graphs per batch
    --max-epochs=100      # upper bound on the number of epochs
    --patience=10         # early stopping: epochs without a new best dev loss
    --grad-clip=5.0       # maximum gradient norm
    --seed=42             # seed of the pool sample, the draws and the weight initialisation
    --num-workers=14      # DataLoader worker processes (keep below --cpus-per-task, keep it fixed within a grid)
    # --use-gse           # uncomment to use the 64 GSE columns as node features
)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # grid search: the line of this array task (task 0 = line 1) holds its hyperparameters
    GRID_LINE=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" ../slurm/04_hparam_grid.txt)
    read -r -a GRID_ARGS <<< "${GRID_LINE}"
    RUN_NAME="grid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "array task ${SLURM_ARRAY_TASK_ID}: ${GRID_LINE}"
else
    GRID_ARGS=()
    RUN_NAME="single_${SLURM_JOB_ID}"
fi

uv run s04_train_model.py "${ARGS[@]}" "${GRID_ARGS[@]}" --run-name="${RUN_NAME}"
