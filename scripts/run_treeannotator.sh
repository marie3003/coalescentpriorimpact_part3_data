#!/bin/bash
#
# Submit treeannotator jobs for a skygrid and/or constcoal posterior trees file.
#
# Usage:
#   run_treeannotator.sh --skygrid-in PATH --skygrid-out PATH \
#                         --constcoal-in PATH --constcoal-out PATH \
#                         [--burnin N] [--heights median|mean|keep]
#
# Either the skygrid pair or the constcoal pair (or both) may be given;
# a model is only submitted if both its --*-in and --*-out are set.
#
# Example (sarscov):
#   scripts/run_treeannotator.sh \
#     --skygrid-in   data/sarscov/beast_skygrid/SG_b117_combined.trees \
#     --skygrid-out  data/sarscov/beast_skygrid/SG_b117_summary.tree \
#     --constcoal-in  data/sarscov/beast_constcoal/SG_constant_b117_combined.trees \
#     --constcoal-out data/sarscov/beast_constcoal/SG_constant_b117_summary.tree

set -euo pipefail

WORKDIR="/cluster/work/stadler/beckermar/pop_prior_impact_data"
BURNIN=0
HEIGHTS=median
SKYGRID_IN=""
SKYGRID_OUT=""
CONSTCOAL_IN=""
CONSTCOAL_OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skygrid-in)    SKYGRID_IN="$2"; shift 2 ;;
        --skygrid-out)   SKYGRID_OUT="$2"; shift 2 ;;
        --constcoal-in)  CONSTCOAL_IN="$2"; shift 2 ;;
        --constcoal-out) CONSTCOAL_OUT="$2"; shift 2 ;;
        --burnin)        BURNIN="$2"; shift 2 ;;
        --heights)       HEIGHTS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$WORKDIR/logs"

submit_job () {
    local model="$1" infile="$2" outfile="$3"

    # resolve to absolute paths under WORKDIR if a relative path was given
    [[ "$infile" = /* ]]  || infile="$WORKDIR/$infile"
    [[ "$outfile" = /* ]] || outfile="$WORKDIR/$outfile"

    echo "Submitting treeannotator for ${model}"
    echo "  in:  $infile"
    echo "  out: $outfile"

    TMPSCRIPT=$(mktemp "$WORKDIR/logs/treeannotator_${model}_XXXX.sh")
    cat > "$TMPSCRIPT" << EOF
#!/usr/bin/env bash
#SBATCH --job-name=treeannotator_${model}
#SBATCH --output=$WORKDIR/logs/treeannotator_${model}_%j.out
#SBATCH --error=$WORKDIR/logs/treeannotator_${model}_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=standard

source /etc/profile
module load stack/2024-06
module load openjdk/21.0.3_9
module load gcc/12.2.0
module load beast1/1.10.4
module load libbeagle/3.1.2

set -euo pipefail

export _JAVA_OPTIONS="-Xmx30g"

echo "Running treeannotator for ${model}"
treeannotator -burnin ${BURNIN} -heights ${HEIGHTS} "${infile}" "${outfile}"

echo "Done: ${model}"
EOF

    sbatch --export=ALL "$TMPSCRIPT"
    rm "$TMPSCRIPT"
}

if [[ -n "$SKYGRID_IN" && -n "$SKYGRID_OUT" ]]; then
    submit_job skygrid "$SKYGRID_IN" "$SKYGRID_OUT"
fi

if [[ -n "$CONSTCOAL_IN" && -n "$CONSTCOAL_OUT" ]]; then
    submit_job constcoal "$CONSTCOAL_IN" "$CONSTCOAL_OUT"
fi

if [[ -z "$SKYGRID_IN$SKYGRID_OUT$CONSTCOAL_IN$CONSTCOAL_OUT" ]]; then
    echo "Nothing to submit — provide at least one of the --skygrid-* or --constcoal-* path pairs." >&2
    exit 1
fi
