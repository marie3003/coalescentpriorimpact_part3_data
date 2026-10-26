#!/bin/bash
#SBATCH --job-name=gubbins_pertussis
#SBATCH --output=logs/gubbins_%j.out
#SBATCH --error=logs/gubbins_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G

set -euo pipefail

WORKDIR="$(pwd)"
FASTA="$WORKDIR/data/pertussis_france/MSA_pertussis_2010_2024_France.fasta"
OUTDIR="$WORKDIR/data/pertussis_france/gubbins"
PREFIX="$OUTDIR/pertussis_france"
THREADS=${SLURM_CPUS_PER_TASK:-2}

mkdir -p "$OUTDIR" "$WORKDIR/logs"
cd "$OUTDIR"


# defaults:
# --filter-percentage 25.0 (filterout sequences with more than 25% gaps)
# --min-window-size 100 (min window size for recombination detection)
# --p-value 0.05 (Uncorrected p value used to identify recombinations)
# --iterations 5 (number of iterations to run)
# --min-snps 3 (Min SNPs to identify a recombination block; for increased sensitivity, set to 2)

run_gubbins.py \
    --prefix "$PREFIX" \
    --threads "$THREADS" \
    --first-tree-builder iqtree-fast \
    --tree-builder iqtree \
    --first-model JC \
    --model GTR \
    --min-snps 3 \
    --iterations 20 \
    "$FASTA"
