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

uv run s01_download_data.py --projectdir "/dss/dsstbyfs02/scratch/0C/di54haf/LOSTInPLACE" 