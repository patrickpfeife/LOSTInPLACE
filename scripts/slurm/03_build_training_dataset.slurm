#!/bin/bash
#SBATCH --job-name=build_train
#SBATCH --output=../logs/analysis_%j.out
#SBATCH --error=../logs/analysis_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=hpda2_compute
#SBATCH --account=hpda-c
#SBATCH --qos=normal

set -euo pipefail

cd ../python

module load uv

ARGS=(
    --k-nearest=25        # number of pool points per seed (also sets the seed capacity: n_center // (4 * k) * density)
    --density=0.66        # share (0, 1] of the burst capacity that is used for seeds
    --min-dist=50         # minimum distance (m) between any two seeds
    --buffer-m=2000       # edge margin (m) inside the overlap of burst and core: no seeds there, but pool points
    --max-dist=2000       # maximum distance (m) of a pool point from its seed
)

uv run s03_build_training_dataset.py "${ARGS[@]}"
