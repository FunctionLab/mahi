"""
Generate Mahi embeddings via tissue-specific propagation, optionally with a single gene KO.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import APPNP
from torch_geometric.utils import to_undirected
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import argparse
import random

def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)

def normalize_key(k):
    try:
        return str(int(k))
    except Exception:
        return str(k)

def read_tissues(path):
    with open(path, "r") as f:
        tissues = [line.strip() for line in f if line.strip()]
    return tissues

def load_embeddings(path):
    with open(path, "rb") as f:
        emb_dict = pickle.load(f)

    emb_dict_norm = {}
    for k, v in emb_dict.items():
        emb_dict_norm[normalize_key(k)] = np.asarray(v, dtype=np.float32)

    keys = sorted(emb_dict_norm.keys())
    key2row = {k: i for i, k in enumerate(keys)}

    d = emb_dict_norm[keys[0]].shape[0]
    X = np.zeros((len(keys), d), dtype=np.float32)
    for k, i in key2row.items():
        X[i] = emb_dict_norm[k]

    x = torch.from_numpy(X)
    return emb_dict_norm, keys, key2row, x

def load_edges(edge_path, key2row, perturb_key=None):
    df = pd.read_csv(edge_path, sep=r"\s+", header=None, names=["src", "dst", "w"], engine="python")

    df["src_key"] = df["src"].apply(lambda v: normalize_key(v))
    df["dst_key"] = df["dst"].apply(lambda v: normalize_key(v))

    if perturb_key is not None:
        before = len(df)
        df = df[(df["src_key"] != perturb_key) & (df["dst_key"] != perturb_key)].copy()
        removed = before - len(df)
        print(f"[perturb] removed {removed} edges touching {perturb_key}", flush=True)

    df = df[df["src_key"].isin(key2row) & df["dst_key"].isin(key2row)].copy()

    src_idx = df["src_key"].map(key2row).to_numpy(dtype=np.int64)
    dst_idx = df["dst_key"].map(key2row).to_numpy(dtype=np.int64)
    w = df["w"].to_numpy(dtype=np.float32)

    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    return df, edge_index, edge_weight

# make the directed edges undirected by summing weights and then applying GCN normalization
def build_norm_graph(edge_index, edge_weight, N):
    # input file is an undirected graph with directed lines.
    # sum weights for u->v and v->u to get a single undirected weight.
    edge_index_ud, edge_weight_ud = to_undirected(
        edge_index, edge_attr=edge_weight, num_nodes=N, reduce="sum"
    )

    # GCN normalization (adds self-loops + degree normalize)
    edge_index_norm, edge_weight_norm = gcn_norm(
        edge_index_ud,
        edge_weight_ud,
        num_nodes=N,
        add_self_loops=True,
        dtype=torch.float32,
    )
    return edge_index_norm, edge_weight_norm

def run_appnp(x, edge_index, edge_weight, k, alpha, device):
    appnp = APPNP(K=k, alpha=alpha, dropout=0.0).to(device)
    appnp.eval()
    with torch.no_grad():
        out = appnp(x, edge_index, edge_weight=edge_weight).cpu().numpy()
    return out

def save_embedding_dict(keys, mat, out_path):
    out_dict = {k: mat[i] for i, k in enumerate(keys)}
    outdir = os.path.dirname(out_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)
    print(f"[OK] wrote: {out_path}", flush=True)

def main(args):
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # load embeddings
    emb_dict, keys, key2row, x_cpu = load_embeddings(args.embeddings)
    N = len(keys)
    print(f"[info] N={N} nodes, d={x_cpu.shape[1]} dims", flush=True)
    x = x_cpu.to(device, non_blocking=True)

    # read all the tissues
    all_tissues = read_tissues(args.tissues_list)    
    if args.tissue:
        assert args.tissue in all_tissues, f"tissue '{args.tissue}' not in list"
        tissues = [args.tissue]
        print(f"[info] running single tissue: {args.tissue}", flush=True)
    else:
        tissues = all_tissues
        print(f"[info] running all tissues: {len(tissues)}", flush=True)

    outdir = os.path.join(args.outdir, "mahi_embeddings")
    os.makedirs(outdir, exist_ok=True)

    perturb_key = None
    if args.perturb_gene and args.perturb_gene.lower() != "none":
        perturb_key = normalize_key(args.perturb_gene)
        print(f"[perturb] KO gene: {perturb_key}", flush=True)
    else:
        print("[perturb] NONE (no edges will be removed)", flush=True)

    # go through each of the tissues, load their graphs, normalize the undirected weights, and run APPNP
    for tissue in tissues:
        graph_path = os.path.join(args.graph_dir, f"{tissue}_filtered_top3.dat")
        if not os.path.exists(graph_path):
            print(f"[warn] missing graph for {tissue}: {graph_path}", flush=True)
            continue

        out_path = os.path.join(outdir, f"{tissue}.pkl")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] exists: {out_path}", flush=True)
            continue
    
        print(f"[run] {tissue}: loading {graph_path}", flush=True)
        _, edge_index_cpu, edge_weight_cpu = load_edges(graph_path, key2row, perturb_key=perturb_key)
        print(f"[info] {tissue}: {edge_index_cpu.shape[1]} directed edges", flush=True)
    
        # normalize graph
        ei_norm_cpu, ew_norm_cpu = build_norm_graph(edge_index_cpu, edge_weight_cpu, N)
        ei_norm = ei_norm_cpu.to(device, non_blocking=True)
        ew_norm = ew_norm_cpu.to(device, non_blocking=True)
        print(f"[info] {tissue}: graph normalized", flush=True)
    
        # run APPNP
        print(f"[run] {tissue}: APPNP: K={args.K}, alpha={args.alpha}", flush=True)
        out = run_appnp(x, ei_norm, ew_norm, args.K, args.alpha, device)
        assert out.shape == (N, x_cpu.shape[1]), f"unexpected shape {out.shape}"
        save_embedding_dict(keys, out, out_path)

    print("[done] all tissues complete", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Mahi embeddings with optional KO.")
    ap.add_argument("--embeddings", required=True, help="Path to concatenated embeddings (.pkl): {id->vector}")
    ap.add_argument("--graph_dir", required=True, help="Dir containing <tissue>_filtered_top3.dat")
    ap.add_argument("--tissues_list", required=True, help="Text file with tissue names (one per line)")
    ap.add_argument("--outdir", required=True, help="Output directory (will make outdir/mahi_embeddings)")
    ap.add_argument("--tissue", default=None, help="If set, only run this tissue")
    ap.add_argument("--perturb_gene", default=None, help="Entrez ID to KO (drop all touching edges); use 'none' for WT")
    ap.add_argument("--K", type=int, default=8, help="K (propagation steps)")
    ap.add_argument("--alpha", type=float, default=0.6, help="propagation teleport probability")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing per-tissue outputs")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)