#!/bin/bash
#SBATCH --job-name=beast_skygrid
#SBATCH --output=logs/beast_skygrid_%j.out
#SBATCH --error=logs/beast_skygrid_%j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=standard

module load stack/2024-06
module load openjdk/21.0.3_9
module load gcc/12.2.0
module load beast1/1.10.4
module load libbeagle/3.1.2

set -euo pipefail

WORKDIR="$(pwd)"
XML="$WORKDIR/scripts/Pertussis_skygrid.xml"
SEED=${SEED:-44}
OUTDIR="$WORKDIR/data/pertussis_france/beast_skygrid"
THREADS=${SLURM_CPUS_PER_TASK:-10}

mkdir -p "$OUTDIR" "$WORKDIR/logs"

# create a seed-specific XML with substituted output file names
TMP_XML="$OUTDIR/Pertussis_skygrid_seed${SEED}.xml"
sed "s/pertussis_2010_2024_skygrid/pertussis_2010_2024_skygrid_seed${SEED}/g" "$XML" > "$TMP_XML"

CHECKPOINT_IN="$OUTDIR/checkpoint_seed${SEED}.xml"
CHECKPOINT_RESUMED2="$OUTDIR/checkpoint_seed${SEED}_resumed2.xml"
CHECKPOINT_RESUMED3="$OUTDIR/checkpoint_seed${SEED}_resumed3.xml"

cd "$OUTDIR"

if [ -f "$CHECKPOINT_RESUMED2" ]; then
    beast \
        -seed "$SEED" \
        -threads "$THREADS" \
        -load_state "$CHECKPOINT_RESUMED2" \
        -save_every 1000000 \
        -save_state "$CHECKPOINT_RESUMED3" \
        -prefix resumed3_ \
        -force_resume \
        "$TMP_XML"
elif [ -f "$CHECKPOINT_IN" ]; then
    beast \
        -seed "$SEED" \
        -threads "$THREADS" \
        -load_state "$CHECKPOINT_IN" \
        -save_every 1000000 \
        -save_state "$CHECKPOINT_RESUMED2" \
        -prefix resumed2_ \
        -force_resume \
        "$TMP_XML"
else
    beast \
        -seed "$SEED" \
        -threads "$THREADS" \
        -overwrite \
        -save_every 1000000 \
        -save_state "$CHECKPOINT_IN" \
        "$TMP_XML"
fi
