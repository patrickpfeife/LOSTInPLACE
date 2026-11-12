#!/bin/bash
#SBATCH --job-name=eval_dev
#SBATCH --output=../logs/eval_dev_%j.out
#SBATCH --error=../logs/eval_dev_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=hpda2_compute
#SBATCH --account=hpda-c
#SBATCH --qos=normal

# Evaluation of trained s04 runs on dev (CPU): baselines (cached per dataset), GNN vs baselines, one report.
# Normally submitted by submit_train_eval.sh after the training; by hand:
#   sbatch 04b_baseline_comparison.slurm single_123          one run
#   sbatch 04b_baseline_comparison.slurm 'grid_456_*'        all runs of a grid (quotes, so the shell doesn't expand it)

set -euo pipefail

cd ../python

module load uv

# the run names or glob patterns come from the command line
uv run s04b_baseline_comparison.py --project-dir="/dss/dsstbyfs02/scratch/0C/di54haf/LOSTInPLACE" --runs "$@"
