#!/bin/bash
#SBATCH --job-name=gnss_trend_comparison
#SBATCH --output=../logs/gnss_trend_%j.out
#SBATCH --error=../logs/gnss_trend_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=hpda2_compute
#SBATCH --account=hpda-c
#SBATCH --qos=normal

set -euo pipefail

cd ../python

module load uv

uv run ../../test/gnss_trend_comparison.py
