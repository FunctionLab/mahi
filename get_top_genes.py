import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def load_pickle(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d

def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

def fold_change_rank(df, avg_df, eps=1e-8):
    for col in ("gene", "distance"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in df")

    if not {"gene", "avg_distance"}.issubset(avg_df.columns):
        raise ValueError("Columns 'gene' or 'avg_distance' not found in avg_df")

    df = df.copy()
    avg_df = avg_df.copy()

    df["gene"] = df["gene"].astype(str)
    avg_df["gene"] = avg_df["gene"].astype(str)

    df = df.sort_values("gene")
    merged = pd.merge(df, avg_df, on="gene", how="inner")

    ratio = (np.abs(merged["distance"]) + eps) / (np.abs(merged["avg_distance"]) + eps)
    merged["fold_change"] = ratio
    merged = merged.sort_values("fold_change", ascending=False)
    return merged

def resolve_tissues(tissue: str | None, tissues: list[str] | None, tissues_txt: str | None) -> list[str]:
    # Priority: tissues_txt > tissues > tissue
    if tissues_txt:
        lines = Path(tissues_txt).read_text().splitlines()
        out = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
        if not out:
            raise ValueError(f"No tissues found in {tissues_txt}")
        return out

    if tissues and len(tissues) > 0:
        out = [t.strip() for t in tissues if t.strip()]
        if out:
            return out

    if tissue and tissue.strip():
        return [tissue.strip()]

    raise ValueError("Provide --tissue, --tissues, or --tissues_txt.")

def compute_top_genes_for_tissue(wt_pkl: Path, ko_pkl: Path, avg_df: pd.DataFrame, out_csv: Path, top: int):
    wt = load_pickle(wt_pkl)
    ko = load_pickle(ko_pkl)

    common = wt.keys() & ko.keys()
    print(f"[{out_csv.name}] Found {len(common)} common genes between WT and KO", flush=True)

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

    ranked = fold_change_rank(df, avg_df)

    total_genes = len(ranked)
    topn = min(total_genes, max(1, int(top)))
    out_df = ranked[["gene", "fold_change"]].head(topn)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved top {topn} genes to {out_csv}", flush=True)

def main():
    p = argparse.ArgumentParser(
        description="Rank KO effects: WT–KO distance fold-change vs average (supports single or multiple tissues)."
    )
    p.add_argument("--dir", required=True, help="Base directory (same as wt_mahi.py / perturb_mahi.py)")
    p.add_argument("--gene", required=True, help="Entrez gene ID used for perturbation (KO stored under <dir>/<gene>/)")
    p.add_argument("--avg", required=True, help="CSV with per-gene average distances (cols: gene, avg_distance)")
    p.add_argument("--top", type=int, default=1000, help="How many top genes to keep (by fold_change). Default: 1000")

    p.add_argument("--tissue", help="Single tissue name (backward-compatible)")
    p.add_argument("--tissues", nargs="+", help="One or more tissues (space-separated)")
    p.add_argument("--tissues_txt", help="Path to txt file with one tissue per line")

    # optional output control
    p.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory. Default: <dir>/<gene>/"
    )

    args = p.parse_args()

    DIR = Path(args.dir)
    GENE = str(args.gene)

    tissues = resolve_tissues(args.tissue, args.tissues, args.tissues_txt)
    avg_df = pd.read_csv(args.avg)

    wt_base = DIR / "mahi_embeddings"
    ko_base = DIR / GENE / "mahi_embeddings"

    if args.out_dir is None:
        out_base = DIR / GENE
    else:
        out_base = Path(args.out_dir)

    for tissue in tissues:
        wt_pkl = wt_base / f"{tissue}.pkl"
        ko_pkl = ko_base / f"{tissue}.pkl"
        out_csv = out_base / f"top_genes_fc.{tissue}.csv"

        if not wt_pkl.exists():
            raise SystemExit(f"Missing WT pickle for tissue '{tissue}': {wt_pkl}")
        if not ko_pkl.exists():
            raise SystemExit(f"Missing KO pickle for tissue '{tissue}': {ko_pkl}")

        compute_top_genes_for_tissue(wt_pkl, ko_pkl, avg_df, out_csv, args.top)

if __name__ == "__main__":
    main()