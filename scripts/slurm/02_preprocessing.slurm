#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --output=../logs/analysis_%j.out
#SBATCH --error=../logs/analysis_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=hpda2_compute
#SBATCH --account=hpda-c
#SBATCH --qos=normal

set -euo pipefail

cd ../python

module load uv

# The EGMS quality filter needs thresholds, either computed from the data OR given as fixed values, never both
# (s02 stops with an error otherwise): keep exactly one of the two blocks below active.
ARGS=(
    # A) thresholds computed from the data: quantiles of a sample of the csvs
    --egms-compute-thresholds
    --egms-coherence-q=0.1      # coherence threshold = this quantile of temporal_coherence (lower 5 % removed)
    --egms-ampdisp-q=0.90        # amplitude dispersion threshold = this quantile (upper 5 % removed)
    --egms-sample-every=20       # for the quantiles, keep every n-th row of each sampled csv
    --egms-file-every=3          # for the quantiles, use every n-th csv

    # B) fixed thresholds (e.g. the values of the 2019-2023 run): comment out block A and uncomment these two lines
    # --egms-coherence=0.32      # minimum temporal_coherence to keep a point
    # --egms-ampdisp-max=0.57    # maximum amplitude_dispersion to keep a point

    # skip single steps (all run by default)
    # --no-egms                  # skip the EGMS preprocessing (quality filter, normalization stats)
    --no-gnss                  # skip the GNSS station filter and velocity fits
    --no-worldcover            # skip the WorldCover merge
)

uv run s02_preprocessing.py "${ARGS[@]}"
