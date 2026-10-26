"""
Count A, C, T, G bases at non-SNP (invariant) positions in a whole-genome alignment.

Usage:
    python count_invariant_bases.py --full <full_alignment.fasta>

Gaps (-) and ambiguous bases (N) are excluded from counts.
"""

import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", required=True, help="Full alignment FASTA (input to snp-sites)")
    return p.parse_args()


def read_fasta(path):
    sequences = {}
    current_id = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                sequences[current_id] = []
            elif current_id is not None:
                sequences[current_id].append(line)
    return {sid: "".join(parts) for sid, parts in sequences.items()}


def main():
    args = parse_args()

    full_seqs = read_fasta(args.full)
    full_ids  = list(full_seqs.keys())

    print(f"Sequences        : {len(full_ids)}")
    print("Building matrix ...")

    # shape: (n_seqs, aln_len) as uint8 character codes
    matrix = np.frombuffer(
        b"".join(s.upper().encode() for s in full_seqs.values()),
        dtype=np.uint8
    ).reshape(len(full_ids), -1)

    aln_len = matrix.shape[1]
    print(f"Alignment length : {aln_len}")

    A, C, T, G, GAP, N = ord("A"), ord("C"), ord("T"), ord("G"), ord("-"), ord("N")

    # mask of valid bases per cell
    valid = (matrix == A) | (matrix == C) | (matrix == T) | (matrix == G)

    print("Identifying SNP positions ...")
    snp_mask = np.zeros(aln_len, dtype=bool)
    for b1, b2 in [(A, C), (A, T), (A, G), (C, T), (C, G), (T, G)]:
        has_b1 = np.any(valid & (matrix == b1), axis=0)
        has_b2 = np.any(valid & (matrix == b2), axis=0)
        snp_mask |= (has_b1 & has_b2)

    invariant_mask = ~snp_mask
    print(f"SNP positions    : {snp_mask.sum()}")
    print(f"Invariant positions: {invariant_mask.sum()}")

    # for each invariant column take the first valid base (all sequences agree)
    # work on full columns to avoid large submatrix allocations
    any_valid = np.any(valid, axis=0)  # shape: (aln_len,)

    counts = {}
    for base_char in ("A", "C", "T", "G"):
        has_base = np.any(matrix == ord(base_char), axis=0)
        counts[base_char] = int((has_base & invariant_mask).sum())

    no_valid_inv = invariant_mask & ~any_valid
    has_N   = np.any(matrix == N,   axis=0) & no_valid_inv
    has_gap = np.any(matrix == GAP, axis=0) & no_valid_inv
    n_only_N   = int(np.sum( has_N & ~has_gap))
    n_only_gap = int(np.sum(~has_N &  has_gap))
    n_mixed    = int(np.sum( has_N &  has_gap))

    total_acgt = sum(counts.values())
    total_inv  = int(invariant_mask.sum())

    print("\nBase counts at invariant positions:")
    for base in ("A", "C", "T", "G"):
        print(f"  {base}         : {counts[base]:>12,}")
    print(f"  N only    : {n_only_N:>12,}")
    print(f"  Gap only  : {n_only_gap:>12,}")
    print(f"  N+Gap mix : {n_mixed:>12,}")
    print(f"  ----------------------------------------")
    print(f"  Total     : {total_acgt + n_only_N + n_only_gap + n_mixed:>12,}  (== invariant positions: {total_inv:,})")


if __name__ == "__main__":
    main()
