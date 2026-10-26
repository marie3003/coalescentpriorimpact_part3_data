"""
Compute Robinson-Foulds (RF) distances between posterior tree samples
(within-skygrid, within-constcoal, and between the two models), over the
full set of pairs, parallelised across CPUs.

Each tree's non-trivial bipartitions are encoded once as a frozenset of
canonical bitmask integers (the same representation dendropy's
treecompare.symmetric_difference uses internally); RF distance is then just
the size of the symmetric difference of two such sets, which is cheap enough
to farm out in large batches to worker processes.

Usage
-----
python scripts/compute_topology_distances.py \
    --skygrid   data/sarscov/beast_skygrid/SG_b117_combined.subsampled1000.trees \
    --constcoal data/sarscov/beast_constcoal/SG_constant_b117_combined.subsampled1000.trees \
    --output    data/sarscov/topology_distances_1000.csv \
    --n-workers 8
"""

import argparse
import os
import sys
from itertools import combinations

import dendropy
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

sys.path.insert(0, ".")
from evaluation_helpers import load_trees


# ---------------------------------------------------------------------------
# globals populated before forking workers (cheap via copy-on-write)
# ---------------------------------------------------------------------------
_BP_A = None
_BP_B = None


def _init_pool(bp_a, bp_b):
    global _BP_A, _BP_B
    _BP_A = bp_a
    _BP_B = bp_b


def _rf_chunk(args):
    """Compute RF for a batch of (i, j) index pairs, comparing _BP_A[i] vs _BP_B[j]."""
    pairs, comparison = args
    out = []
    for i, j in pairs:
        rf = len(_BP_A[i] ^ _BP_B[j])
        out.append((comparison, i, j, rf))
    return out


def _bipartition_sets(trees):
    """
    Encode each tree's bipartitions as a frozenset of canonical bitmask ints.
    Matches dendropy.calculate.treecompare.symmetric_difference, which compares
    the full `bipartition_encoding` sets (trivial splits included — they're
    identical across trees sharing a taxon namespace, so they cancel out).
    """
    sets = []
    for t in trees:
        t.encode_bipartitions()
        bp = frozenset(b.split_bitmask for b in t.bipartition_encoding)
        sets.append(bp)
    return sets


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def compute_all_distances(sky_bp, const_bp, n_workers, chunk_size=5000):
    n_sky, n_const = len(sky_bp), len(const_bp)

    within_sky_pairs   = list(combinations(range(n_sky), 2))
    within_const_pairs = list(combinations(range(n_const), 2))
    between_pairs      = [(i, j) for i in range(n_sky) for j in range(n_const)]

    print(f"within_skygrid: {len(within_sky_pairs)} pairs")
    print(f"within_constcoal: {len(within_const_pairs)} pairs")
    print(f"between: {len(between_pairs)} pairs")

    # Run one comparison group at a time so the pool's shared globals (_BP_A/_BP_B)
    # match the pair of tree sets being compared in that group.
    records = []
    ctx = mp.get_context("fork")
    for pairs, bp_a, bp_b, comparison in [
        (within_sky_pairs,   sky_bp,   sky_bp,   'within_skygrid'),
        (within_const_pairs, const_bp, const_bp, 'within_constcoal'),
        (between_pairs,      sky_bp,   const_bp, 'between'),
    ]:
        chunks = [(chunk, comparison) for chunk in _chunked(pairs, chunk_size)]

        print(f"Computing {comparison} distances ({len(pairs)} pairs) "
              f"with {n_workers} workers …")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                  initializer=_init_pool, initargs=(bp_a, bp_b)) as pool:
            done = 0
            for result in pool.map(_rf_chunk, chunks):
                records.extend(result)
                done += len(result)
                print(f"  {done}/{len(pairs)} done", flush=True)

    return pd.DataFrame(records, columns=['comparison', 'tree1_idx', 'tree2_idx', 'rf'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skygrid",   required=True, help="Skygrid posterior trees file")
    parser.add_argument("--constcoal", required=True, help="Constcoal posterior trees file")
    parser.add_argument("--output",    required=True, help="Output CSV path")
    parser.add_argument("--n-workers", type=int, default=None,
                         help="Number of worker processes (default: all available CPUs)")
    args = parser.parse_args()

    n_workers = args.n_workers or os.cpu_count() or 1

    print("Loading trees …")
    tns = dendropy.TaxonNamespace()
    sky_trees   = load_trees(args.skygrid,   taxon_namespace=tns)
    const_trees = load_trees(args.constcoal, taxon_namespace=tns)
    n_taxa = len(tns)
    print(f"Loaded {len(sky_trees)} skygrid trees and {len(const_trees)} constcoal trees, "
          f"{n_taxa} taxa.")

    print("Encoding bipartitions …")
    sky_bp   = _bipartition_sets(sky_trees)
    const_bp = _bipartition_sets(const_trees)

    df = compute_all_distances(sky_bp, const_bp, n_workers=n_workers)

    rf_max = 2 * (n_taxa - 3)
    df['rf_norm'] = df['rf'] / rf_max

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
