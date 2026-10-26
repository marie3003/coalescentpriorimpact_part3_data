"""
For each pair of tips and each posterior tree, compute the branch-length
distance from each tip to their MRCA (coalescent event).  Summarise over
the posterior samples with median and 95% HPD, then output one combined
dataframe covering both the skygrid and constcoal runs, plus the per-tree
absolute and relative differences (constcoal - skygrid) with their HPDs.

Usage
-----
python scripts/compute_coalescent_distances.py \
    --skygrid  data/pertussis_france/beast_skygrid/pertussis_skygrid_2010_2024_allseeds_10k.subsampled200.trees \
    --constcoal data/pertussis_france/beast_constcoal/pertussis_constcoal_2010_2024_allseeds_10k.subsampled200.trees \
    --metadata  data/pertussis_france/full_metadata.csv \
    --output    data/pertussis_france/coalescent_tip_distances.tsv
"""

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
import dendropy

sys.path.insert(0, ".")
from evaluation_helpers import hpd_interval


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _path_length_to_mrca(tip, mrca):
    """Sum of branch lengths from tip up to (not including) mrca."""
    total = 0.0
    node = tip
    while node is not mrca:
        if node.edge_length is not None:
            total += node.edge_length
        node = node.parent_node
    return total


def compute_pair_distances(trees_path):
    """
    Read a posterior trees file and, for every pair of tips, collect the
    distance from tip1→MRCA and tip2→MRCA across all trees.

    Returns
    -------
    dict  {(label1, label2): {"tip1": [d, ...], "tip2": [d, ...]}}
          label1 < label2 lexicographically
    """
    print(f"  Streaming trees from {trees_path} …", flush=True)

    samples = defaultdict(lambda: {"tip1": [], "tip2": []})
    taxon_namespace = dendropy.TaxonNamespace()

    tree_idx = -1
    for tree_idx, tree in enumerate(
        dendropy.Tree.yield_from_files(
            files=[trees_path],
            schema="nexus",
            preserve_underscores=True,
            taxon_namespace=taxon_namespace,
        )
    ):
        tips = list(tree.leaf_node_iter())
        for n1, n2 in combinations(tips, 2):
            l1, l2 = n1.taxon.label, n2.taxon.label
            if l1 > l2:
                l1, l2, n1, n2 = l2, l1, n2, n1
            mrca = tree.mrca(taxa=[n1.taxon, n2.taxon])
            samples[(l1, l2)]["tip1"].append(_path_length_to_mrca(n1, mrca))
            samples[(l1, l2)]["tip2"].append(_path_length_to_mrca(n2, mrca))

        if (tree_idx + 1) % 10 == 0:
            print(f"    … processed {tree_idx + 1} trees", flush=True)

    print(f"  Processed {tree_idx + 1} trees total.", flush=True)
    return dict(samples)


def summarise(values):
    """Return (median, hpd_lower, hpd_upper) for a list of floats."""
    arr = np.array(values)
    lo, hi = hpd_interval(arr)
    return float(np.median(arr)), float(lo), float(hi)


def build_dataframe(samples, model_label):
    """Turn the raw per-pair samples into a summary dataframe for one model."""
    records = []
    for (l1, l2), dists in samples.items():
        med1, lo1, hi1 = summarise(dists["tip1"])
        med2, lo2, hi2 = summarise(dists["tip2"])
        records.append({
            "tip1": l1,
            "tip2": l2,
            f"{model_label}_tip1_to_mrca_median": med1,
            f"{model_label}_tip1_to_mrca_hpd_lower": lo1,
            f"{model_label}_tip1_to_mrca_hpd_upper": hi1,
            f"{model_label}_tip2_to_mrca_median": med2,
            f"{model_label}_tip2_to_mrca_hpd_lower": lo2,
            f"{model_label}_tip2_to_mrca_hpd_upper": hi2,
        })
    return pd.DataFrame(records)


def build_diff_dataframe(sky_samples, const_samples):
    """
    For each tip pair compute per-tree absolute and relative differences
    (constcoal - skygrid) for tip1 and tip2, then summarise with median + HPD.
    Assumes sky_samples and const_samples have matching keys and equal-length lists.
    """
    records = []
    for key in sky_samples:
        if key not in const_samples:
            continue
        l1, l2 = key
        record = {"tip1": l1, "tip2": l2}
        for tip in ("tip1", "tip2"):
            sky_arr   = np.array(sky_samples[key][tip])
            const_arr = np.array(const_samples[key][tip])
            n = min(len(sky_arr), len(const_arr))
            sky_arr, const_arr = sky_arr[:n], const_arr[:n]

            abs_diff = const_arr - sky_arr
            with np.errstate(invalid='ignore', divide='ignore'):
                rel_diff = np.where(sky_arr > 0, (const_arr - sky_arr) / sky_arr, np.nan)

            abs_med, abs_lo, abs_hi = summarise(abs_diff)
            valid_rel = rel_diff[np.isfinite(rel_diff)]
            if len(valid_rel) > 1:
                rel_med, rel_lo, rel_hi = summarise(valid_rel)
            else:
                rel_med = rel_lo = rel_hi = np.nan

            record.update({
                f"abs_diff_{tip}_median":    abs_med,
                f"abs_diff_{tip}_hpd_lower": abs_lo,
                f"abs_diff_{tip}_hpd_upper": abs_hi,
                f"rel_diff_{tip}_median":    rel_med,
                f"rel_diff_{tip}_hpd_lower": rel_lo,
                f"rel_diff_{tip}_hpd_upper": rel_hi,
            })
        records.append(record)
    return pd.DataFrame(records)


def attach_tip_dates(df, meta_df):
    """
    Add tip1_date and tip2_date columns from the metadata.
    """
    date_lookup = (
        meta_df
        .assign(key=meta_df["Isolate"].str.replace(" ", "_"))
        .set_index("key")["decimal_year"]
        .to_dict()
    )
    df = df.copy()
    df["tip1_norm"] = df["tip1"].str.replace(" ", "_")
    df["tip2_norm"] = df["tip2"].str.replace(" ", "_")
    df["tip1_date"] = df["tip1_norm"].map(date_lookup)
    df["tip2_date"] = df["tip2_norm"].map(date_lookup)
    df = df.drop(columns=["tip1_norm", "tip2_norm"])
    missing = df["tip1_date"].isna().sum() + df["tip2_date"].isna().sum()
    if missing:
        print(f"  Warning: {missing} tip–date lookups failed (tip absent from metadata).")
    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skygrid",   required=True, help="Skygrid posterior trees file")
    parser.add_argument("--constcoal", required=True, help="Constcoal posterior trees file")
    parser.add_argument("--metadata",  required=False, default=None,
                        help="Optional CSV with 'Isolate' and 'decimal_year' columns; "
                             "if provided, tip1_date and tip2_date are added to the output")
    parser.add_argument("--output",    required=True, help="Output TSV path")
    args = parser.parse_args()

    print("Processing skygrid and constcoal trees in parallel …")
    with ProcessPoolExecutor(max_workers=2) as pool:
        sky_future = pool.submit(compute_pair_distances, args.skygrid)
        const_future = pool.submit(compute_pair_distances, args.constcoal)
        sky_samples = sky_future.result()
        const_samples = const_future.result()

    sky_df = build_dataframe(sky_samples, "skygrid")
    const_df = build_dataframe(const_samples, "constcoal")

    print("Computing per-tree differences …")
    diff_df = build_diff_dataframe(sky_samples, const_samples)

    print("Merging …")
    df = sky_df.merge(const_df, on=["tip1", "tip2"], how="outer")
    df = df.merge(diff_df, on=["tip1", "tip2"], how="left")

    if args.metadata is not None:
        sep = "\t" if args.metadata.endswith(".tsv") else ","
        meta_df = pd.read_csv(args.metadata, sep=sep)
        df = attach_tip_dates(df, meta_df)
        id_cols = ["tip1", "tip2", "tip1_date", "tip2_date"]
    else:
        id_cols = ["tip1", "tip2"]

    other_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + other_cols]

    df.to_csv(args.output, sep="\t", index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
