#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --output=../logs/analysis_%j.out
#SBATCH --error=../logs/analysis_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=hpda2_compute
#SBATCH --account=hpda-c
#SBATCH --qos=normal

set -euo pipefail

cd ../python

module load uv

uv run 01_download_data.py