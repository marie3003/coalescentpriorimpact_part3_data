"""
Subsample a FASTA file keeping only sequences whose IDs appear in a text file.

Usage:
    python subsample_fasta.py --fasta <input.fasta> --ids <ids.txt> --out <output.fasta>

The IDs file should contain one sequence ID per line (without the leading '>').
A FASTA sequence is kept if its ID (up to the first whitespace) starts with any
of the IDs listed in the file — i.e. the listed ID must match the full prefix,
but the FASTA ID may have additional characters after it.
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", required=True, help="Input FASTA file")
    p.add_argument("--ids",   required=True, help="Text file with one sequence ID per line")
    p.add_argument("--out",   required=True, help="Output FASTA file")
    return p.parse_args()


def load_ids(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def subsample_fasta(fasta_path, ids, out_path):
    kept = 0
    total = 0
    write = False

    with open(fasta_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                total += 1
                # sequence id is everything after '>' up to first whitespace
                seq_id = line[1:].split()[0].strip()
                write = any(seq_id.startswith(tid) for tid in ids)
                if write:
                    kept += 1
                    short_id = seq_id.split("_")[0]
                    fout.write(f">{short_id}\n")
            elif write:
                fout.write(line)


    print(f"Kept {kept} / {total} sequences -> {out_path}")


def main():
    args = parse_args()
    ids = load_ids(args.ids)
    print(f"Loaded {len(ids)} IDs from {args.ids}")
    subsample_fasta(args.fasta, ids, args.out)


if __name__ == "__main__":
    main()
