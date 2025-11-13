"""
Concatenate requested embeddings (esm, deepsea, graph) by key intersection.
Saves a dict: {entrez_id(str): torch.FloatTensor} with concat vectors.
"""

import argparse
import pickle
import torch
import os

def main(args):
    def load_if(path):
        if path is None: return None
        with open(path, "rb") as f: return pickle.load(f)
            
    esm = load_if(args.esm_embeddings_path)
    deepsea = load_if(args.deepsea_embeddings_path)
    graph = load_if(args.graph_embeddings_path)

    pools = []
    if "esm" in args.which_embeddings:
        assert esm is not None, "esm requested but --esm_embeddings_path missing"
        pools.append(set(esm.keys()))
    if "deepsea" in args.which_embeddings:
        assert deepsea is not None, "deepsea requested but --deepsea_embeddings_path missing"
        pools.append(set(deepsea.keys()))
    if "graph" in args.which_embeddings:
        assert graph is not None, "graph requested but --graph_embeddings_path missing"
        pools.append(set(graph.keys()))

    common = set.intersection(*pools) if pools else set()
    
    out = {}
    for k in common:
        parts = []
        if "esm" in args.which_embeddings:
            parts.append(esm[k].cpu() if hasattr(esm[k], "cpu") else torch.tensor(esm[k]))
        if "deepsea" in args.which_embeddings:
            parts.append(deepsea[k].cpu() if hasattr(deepsea[k], "cpu") else torch.tensor(deepsea[k]))
        if "graph" in args.which_embeddings:
            parts.append(graph[k].cpu() if hasattr(graph[k], "cpu") else torch.tensor(graph[k]))
        out[k] = torch.cat(parts, dim=-1)
    
    outdir = os.path.dirname(args.output_embeddings_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.output_embeddings_path, "wb") as f:
        pickle.dump(out, f)
        
    print(f"{len(common)} overlapping nodes saved to {args.output_embeddings_path}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm_embeddings_path", type=str)
    ap.add_argument("--deepsea_embeddings_path", type=str)
    ap.add_argument("--graph_embeddings_path", type=str)
    ap.add_argument("--which_embeddings", nargs="+", choices=["esm","deepsea","graph"], required=True)
    ap.add_argument("--output_embeddings_path", type=str, required=True)
    args = ap.parse_args()
    main(args)