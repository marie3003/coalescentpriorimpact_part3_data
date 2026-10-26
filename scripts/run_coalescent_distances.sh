#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/coalescent_distances_1000_%j.out
#SBATCH --error=logs/coalescent_distances_1000_%j.err
#
# Usage:
#   sbatch scripts/run_coalescent_distances.sh
#       (no args) runs on the default pertussis subsampled1000 trees.
#   sbatch scripts/run_coalescent_distances.sh SKY_TREES CONST_TREES METADATA OUTPUT
#       runs on the given trees/metadata/output paths instead.
#
# Example (sarscov):
#   sbatch scripts/run_coalescent_distances.sh \
#     data/sarscov/beast_skygrid/SG_b117_combined.subsampled1000.trees \
#     data/sarscov/beast_constcoal/SG_constant_b117_combined.subsampled1000.trees \
#     data/sarscov/taxon_dates.tsv \
#     data/sarscov/coalescent_tip_distances_1000.tsv

source $(conda info --base)/etc/profile.d/conda.sh
conda activate beast_tools

SKY_TREES=${1:-"data/pertussis_france/beast_skygrid/pertussis_skygrid_2010_2024_allseeds_10k.subsampled1000.trees"}
CONST_TREES=${2:-"data/pertussis_france/beast_constcoal/pertussis_constcoal_2010_2024_allseeds_10k.subsampled1000.trees"}
METADATA=${3:-"data/pertussis_france/full_metadata.csv"}
OUTPUT=${4:-"data/pertussis_france/coalescent_tip_distances_1000.tsv"}

python scripts/compute_coalescent_distances.py \
    --skygrid   "$SKY_TREES" \
    --constcoal "$CONST_TREES" \
    --metadata  "$METADATA" \
    --output    "$OUTPUT"
