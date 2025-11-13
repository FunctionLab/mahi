import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def load_pickle(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

def fold_change_rank(df, avg_df, eps=1e-8, log_base=2):
    for col in ("gene", "distance"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in df")

    if not {"gene", "avg_distance"}.issubset(avg_df.columns):
        raise ValueError("Columns 'gene' or 'avg_distance' not found in avg_df")

    df["gene"] = df["gene"].astype(str)
    avg_df["gene"] = avg_df["gene"].astype(str)

    df = df.sort_values("gene")
    merged = pd.merge(df, avg_df, on="gene", how="inner")

    ratio = (np.abs(merged["distance"]) + eps) / (np.abs(merged["avg_distance"]) + eps)

    merged["fold_change"] = ratio

    merged = merged.sort_values("fold_change", ascending=False)

    return merged

def main():
    p = argparse.ArgumentParser(description="Compute WT–KO distances, fold-change vs average, and save top-N CSV.")
    p.add_argument("--wt", required=True, help="WT Mahi pickle for tissue")
    p.add_argument("--ko", required=True, help="KO Mahi pickle for same tissue")
    p.add_argument("--avg", required=True, help="CSV with per-gene average distances (cols: gene, avg_distance)")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--top", type=int, default=1000, help="How many top genes to keep (by fold_change). Default: 1000")
    args = p.parse_args()

    wt = load_pickle(args.wt)
    ko = load_pickle(args.ko)

    common = wt.keys() & ko.keys()
    print(f"Found {len(common)} common genes between WT and KO", flush=True)

    rows = []
    for g in common:
        a = np.asarray(wt[g], dtype=float)
        b = np.asarray(ko[g], dtype=float)
        if a.shape != b.shape:
            raise SystemExit(f"Shape mismatch for {g}: {a.shape} vs {b.shape}")
        rows.append((g, euclidean_distance(a, b)))
    
    df = pd.DataFrame(rows, columns=["gene", "distance"]).sort_values(
        "distance", ascending=False, kind="mergesort"
    )

    avg_df = pd.read_csv(args.avg)
    ranked = fold_change_rank(df, avg_df)

    total_genes = len(ranked)
    topn = min(total_genes, max(1, int(args.top)))
    out_df = ranked[["gene", "fold_change"]].head(topn)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Saved top {topn} genes to {args.out}")

if __name__ == "__main__":
    main()