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
    --min-dist-seeds=150  # minimum distance (m) between any two seeds
    --buffer-m=2000       # edge margin (m) inside the overlap of burst and core: no seeds there, but pool points
    --max-dist=2000       # maximum distance (m) of a pool point from its seed
    --min-dist-pool=50    # minimum distance between the center node and the next pool point
    # distance distribution of the pool points (find_pool_points): per sector iteration a mean is drawn from
    # N(mu-mu-dist, sigma-mu-dist) and a spread from N(mu-sigma-dist, sigma-sigma-dist), then the target distance
    # from N(mean, spread); values as in the prototype (test/distance_pools_prototype.py)
    --mu-mu-dist=150      # mean (m) of the distribution the target means are drawn from
    --sigma-mu-dist=75    # std (m) of the distribution the target means are drawn from
    --mu-sigma-dist=60    # mean (m) of the distribution the target spreads are drawn from
    --sigma-sigma-dist=30 # std (m) of the distribution the target spreads are drawn from
    --tolerance=50        # a point is only taken if its distance to the seed is within this many metres of the target
    --max-tries=5         # rounds over all sectors in a row without a pick after which the pool is closed
)

uv run s03_build_training_dataset.py "${ARGS[@]}"
