import pickle
import numpy as np
import pandas as pd
import argparse

def load_pickle(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

def main():
    p = argparse.ArgumentParser(description="Compute WT–KO distances per gene and save CSV.")
    p.add_argument("--wt", required=True, help="WT pickle: dict[gene] -> 1D array")
    p.add_argument("--ko", required=True, help="KO pickle: dict[gene] -> 1D array")
    p.add_argument("--out", required=True, help="Output CSV path")
    #p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
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

    gtf = pd.read_csv('/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/hg38.p14.ncbiRefSeq.transcript_final_gtf.csv')
    gtf = gtf.rename(columns={"entrez_id": "gene", "gene_id": "gene_name"})
    
    df = pd.DataFrame(rows, columns=["gene", "distance"]).sort_values(
        "distance", ascending=False, kind="mergesort"
    )

    df["gene"] = df["gene"].astype(str)
    gtf["gene"] = gtf["gene"].astype(str)

    df = df.merge(gtf[["gene", "gene_name"]], on="gene", how="left")
    df = df[["gene", "gene_name", "distance"]]
    
    df.to_csv(args.out, index=False)
    print(f"Saved CSV file", flush=True)

if __name__ == "__main__":
    main()