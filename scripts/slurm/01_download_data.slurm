#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --output=../logs/analysis_%j.out
#SBATCH --error=../logs/analysis_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --partition=hpda2_compute
#SBATCH --account=hpda-c
#SBATCH --qos=normal

set -euo pipefail

cd ../python

module load uv

ARGS=(
    --project-dir "/dss/dsstbyfs02/scratch/0C/di54haf/LOSTInPLACE"  # root of the data tree, data/{train,dev,cal,val}/ is created under it
    --real-sub-buffer-value=150000   # minimum gap (m) between the query areas of the real subsidence and the random regions
    --n-rnd=35                       # number of random regions sampled across the EGMS coverage area
    --mindist-rnd=150000             # minimum distance (m) between the random region centres
    --maxit-rnd=200                  # upper bound on sampling attempts for the random regions
    --egms-rand-extension=15000      # half side length (m) of the square core of a random region (15000 -> 30 x 30 km)
)

uv run s01_download_data.py "${ARGS[@]}"