#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/topology_distances_1000_%j.out
#SBATCH --error=logs/topology_distances_1000_%j.err
#
# Usage:
#   sbatch scripts/run_topology_distances.sh
#       (no args) runs on the default pertussis subsampled1000 trees.
#   sbatch scripts/run_topology_distances.sh SKY_TREES CONST_TREES OUTPUT
#       runs on the given trees/output paths instead.
#
# Example (sarscov):
#   sbatch scripts/run_topology_distances.sh \
#     data/sarscov/beast_skygrid/SG_b117_combined.subsampled1000.trees \
#     data/sarscov/beast_constcoal/SG_constant_b117_combined.subsampled1000.trees \
#     data/sarscov/topology_distances_1000.csv

source $(conda info --base)/etc/profile.d/conda.sh
conda activate beast_tools

SKY_TREES=${1:-"data/pertussis_france/beast_skygrid/pertussis_skygrid_2010_2024_allseeds_10k.subsampled1000.trees"}
CONST_TREES=${2:-"data/pertussis_france/beast_constcoal/pertussis_constcoal_2010_2024_allseeds_10k.subsampled1000.trees"}
OUTPUT=${3:-"data/pertussis_france/topology_distances_1000.csv"}

python scripts/compute_topology_distances.py \
    --skygrid   "$SKY_TREES" \
    --constcoal "$CONST_TREES" \
    --output    "$OUTPUT" \
    --n-workers "${SLURM_CPUS_PER_TASK:-16}"
