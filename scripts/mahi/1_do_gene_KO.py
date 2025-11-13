"""
Remove all edges touching a given Entrez ID in the across-tissues multigraph CSV.
"""

import argparse
import pandas as pd

def main(args):
    g = pd.read_csv(args.graph_csv)
    s = g["source"].astype(str); t = g["target"].astype(str)
    mask = (s == str(args.perturb_gene)) | (t == str(args.perturb_gene))

    removed = int(mask.sum())
    if removed > 0:
        g2 = g.loc[~mask].copy()
        print(f"KO {args.perturb_gene} FOUND: removed {removed} edges; new edge count {len(g2)}", flush=True)
    else:
        g2 = g.copy()
        print(f"KO {args.perturb_gene} NOT found. No edges removed.", flush=True)
    g2.to_csv(args.output_csv, index=False)
    print(f"[OK] wrote {args.output_csv}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_csv", required=True)
    ap.add_argument("--perturb_gene", required=True)
    ap.add_argument("--output_csv", required=True)
    main(ap.parse_args())