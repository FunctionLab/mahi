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

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.use_deterministic_algorithms(True)

DROPOUT = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# returns keys as str(int(k))
def normalize_key(k):
    try:
        return str(int(k))
    except Exception:
        return str(k)

# load the raw concatenated embeddings
def load_embeddings(path):
    with open(path, "rb") as f:
        emb_dict = pickle.load(f)

    # normalize keys to string entrez ids
    emb_dict_norm = {}
    for k, v in emb_dict.items():
        emb_dict_norm[normalize_key(k)] = np.asarray(v, dtype=np.float32)

    # sort the nodes for consistency
    keys = sorted(emb_dict_norm.keys())
    key2row = {k: i for i, k in enumerate(keys)}

    # generate a placeholder for the appnp embeddings
    d = emb_dict_norm[keys[0]].shape[0]
    X = np.zeros((len(keys), d), dtype=np.float32)
    for k, i in key2row.items():
        X[i] = emb_dict_norm[k]

    x = torch.from_numpy(X)  # [N, d]
    return emb_dict_norm, keys, key2row, x

def load_edges(edge_path, key2row, perturb_key=None):
    # .dat: "src  dst  weight"
    df = pd.read_csv(edge_path, sep=r"\s+", header=None, names=["src", "dst", "w"], engine="python")

    # normalize endpoints to str entrez
    df["src_key"] = df["src"].apply(lambda v: normalize_key(v))
    df["dst_key"] = df["dst"].apply(lambda v: normalize_key(v))

    if perturb_key is not None:
        before = len(df)
        df = df[(df["src_key"] != perturb_key) & (df["dst_key"] != perturb_key)].copy()
        removed = before - len(df)
        print(f"[perturb] removed {removed} edges touching {perturb_key}", flush=True)

    # keep edges whose endpoints exist in embedding dict
    df = df[df["src_key"].isin(key2row) & df["dst_key"].isin(key2row)].copy()

    # map to row indices
    src_idx = df["src_key"].map(key2row).to_numpy(dtype=np.int64)
    dst_idx = df["dst_key"].map(key2row).to_numpy(dtype=np.int64)
    w = df["w"].to_numpy(dtype=np.float32)

    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)   # [2, E]
    edge_weight = torch.tensor(w, dtype=torch.float32)                            # [E]
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

def run_appnp(x, edge_index, edge_weight, k, alpha, dropout=DROPOUT, device=DEVICE):
    appnp = APPNP(K=k, alpha=alpha, dropout=dropout).to(device)
    appnp.eval()
    with torch.no_grad():
        out = appnp(x, edge_index, edge_weight=edge_weight).cpu().numpy()
    return out  # [N, d]

def save_embedding_dict(keys, mat, out_path):
    out_dict = {k: mat[i] for i, k in enumerate(keys)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)
    print(f"[OK] wrote: {out_path}", flush=True)

def read_tissues(path):
    with open(path, "r") as f:
        tissues = [line.strip() for line in f if line.strip()]
    return tissues

def main():
    parser = argparse.ArgumentParser(description="Generate MAHI embeddings via APPNP with single-gene perturbation.")
    parser.add_argument("--perturb_gene", type=str, default=None,
                        help="Entrez ID to perturb (drop all edges touching this node).")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to concatenated baseline embeddings .pkl (node->vector).")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory. Will create <outdir>/mahi_embeddings/ containing {tissue}.pkl files.")
    parser.add_argument("--tissue", type=str, default=None,
                    help="If provided, only generate embeddings for this single tissue.")

    args = parser.parse_args()

    out_base = os.path.join(args.outdir, "mahi_embeddings")
    os.makedirs(out_base, exist_ok=True)

    concat_embed_path = args.embeddings
    tissues_list_path = "/mnt/home/aaggarwal/gates_proj/target_genes/MAGE/20_mahi_final_test/resources/all_mage_tissues.txt"
    graph_dir = "/mnt/home/aaggarwal/ceph/gates_proj/MAGE_networks/all_mage_2025_networks"
    outdir = out_base
    
    fixed_k = 8
    fixed_alpha = 0.6
        
    # 1) load embeddings
    emb_dict, keys, key2row, x_cpu = load_embeddings(concat_embed_path)
    N = len(keys)
    print(f"[info] N={N} nodes, d={x_cpu.shape[1]} dims", flush=True)

    x = x_cpu.to(DEVICE, non_blocking=True)

    # 2) read all the tissues
    all_tissues = read_tissues(tissues_list_path)
    print(f"[info] found {len(all_tissues)} tissues", flush=True)
    
    if args.tissue is not None:
        if args.tissue not in all_tissues:
            print(f"[error] tissue '{args.tissue}' not found in master list", flush=True)
            return
        tissues = [args.tissue]
        print(f"[info] running only for tissue: {args.tissue}", flush=True)
    else:
        tissues = all_tissues

    # normalize perturb gene key once
    perturb_key = None
    if args.perturb_gene and args.perturb_gene.lower() != "none":
        perturb_key = normalize_key(args.perturb_gene)
        print(f"[perturb] KO gene: {perturb_key}", flush=True)
    else:
        print("[perturb] NONE (no edges will be removed)", flush=True)

    # 3) go through each of the tissues, load their graphs, normalize the undirected weights, and run APPNP
    for tissue in tissues:
        graph_path = os.path.join(graph_dir, f"{tissue}_filtered_top3.dat")
        if not os.path.exists(graph_path):
            print(f"[warn] missing graph for {tissue}: {graph_path}", flush=True)
            continue

        out_path = os.path.join(outdir, f"{tissue}.pkl")
        if os.path.exists(out_path):
            print(f"[skip] already exists: {out_path}", flush=True)
            continue
    
        print(f"[run] {tissue}: loading edges from {graph_path}", flush=True)
        df_edges, edge_index_cpu, edge_weight_cpu = load_edges(graph_path, key2row, perturb_key=perturb_key)
        print(f"[info] loaded {tissue}: loaded {edge_index_cpu.shape[1]} directed edges", flush=True)
    
        # normalize graph
        ei_norm_cpu, ew_norm_cpu = build_norm_graph(edge_index_cpu, edge_weight_cpu, N)
        ei_norm = ei_norm_cpu.to(DEVICE, non_blocking=True)
        ew_norm = ew_norm_cpu.to(DEVICE, non_blocking=True)
        print(f"[info] {tissue}: graph normalized", flush=True)
    
        # run APPNP
        print(f"[run] {tissue}: APPNP: alpha={fixed_alpha:.1f}, K={fixed_k}", flush=True)
        out = run_appnp(x, ei_norm, ew_norm, k=fixed_k, alpha=fixed_alpha)
        assert out.shape == (N, x_cpu.shape[1]), f"unexpected shape {out.shape}"
        save_embedding_dict(keys, out, out_path)

    print("[done] all tissues complete", flush=True)

if __name__ == "__main__":
    main()