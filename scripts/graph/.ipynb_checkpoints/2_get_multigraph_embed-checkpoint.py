"""
Extract multigraph node embeddings from trained masked-edge-reconstruction model.
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.norm import PairNorm
from tqdm import tqdm

import pytorch_lightning as pl

# utilities

def set_seed(seed: int = 42):
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def read_txt(path: str):
    df = pd.read_csv(path, header=None)
    nodes = df[0].tolist()
    return nodes

def create_node_mapping(all_nodes):
    all_nodes = sorted(list(all_nodes))
    return {node: i for i, node in enumerate(all_nodes)}

def process_graph(graph, node_to_idx, tissue_cols):
    graph = graph.copy()
    graph['source'] = graph['source'].map(lambda x: node_to_idx.get(x, -1))
    graph['target'] = graph['target'].map(lambda x: node_to_idx.get(x, -1))
    
    graph = graph[(graph['source'] != -1) & (graph['target'] != -1)]

    edges = graph[['source', 'target']].to_numpy()
    rev_edges = edges[:, [1, 0]]
    all_edges = np.vstack((edges, rev_edges))
    edge_index = torch.tensor(all_edges.T, dtype=torch.long)

    edge_features = graph[tissue_cols].to_numpy()
    all_edge_features = np.vstack((edge_features, edge_features))
    edge_attr = torch.tensor(all_edge_features, dtype=torch.float)

    return edge_index, edge_attr

def normalize_entrez_id(n):
    if isinstance(n, str):
        s = n.strip()
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except ValueError:
            pass
        if s.isdigit():
            return s
        return s

    if isinstance(n, (int, np.integer)):
        return str(int(n))
    if isinstance(n, (float, np.floating)):
        return str(int(n)) if float(n).is_integer() else str(n)

    return str(n)

# model
class TransformerMaskedModel(pl.LightningModule):
    def __init__(self,
                 emb_dim,
                 edge_dim,
                 num_layers,
                 heads,
                 dropout,
                 learning_rate,
                 weight_decay,
                 batch_size,
                 num_neighbors,
                 epochs,
                 masking_ratio,
                 train_data,
                 val_data,
                 test_data,
                 norm_type,
                 global_skip,
                 use_residual):
        super().__init__()

        self.val_outputs = []
        self.test_outputs = []

        self.save_hyperparameters({
            "emb_dim": emb_dim,
            "edge_dim": edge_dim,
            "num_layers": num_layers,
            "heads": heads,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_neighbors": num_neighbors,
            "epochs": epochs,
            "masking_ratio": masking_ratio,
            "norm_type": norm_type,
            "global_skip": global_skip,
            "use_residual": use_residual,
        })

        self.masking_ratio = masking_ratio
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # --- GNN stack (constant width) ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        for _ in range(self.hparams.num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=emb_dim,
                    out_channels=emb_dim,
                    edge_dim=edge_dim,
                    heads=self.hparams.heads,
                    concat=False,
                    dropout=self.hparams.dropout
                )
            )
            self.norms.append(
                nn.LayerNorm(emb_dim) if self.hparams.norm_type == "layernorm" else PairNorm()
            )
            self.residual_projections.append(nn.Identity())  # same dims

        self.dropout = nn.Dropout(dropout)

        final_node_dim = emb_dim + (emb_dim if self.hparams.global_skip else 0)

        # edge predictor head
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * final_node_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, edge_dim)
        )

        self._lr = learning_rate
        self._wd = weight_decay
        self._epochs = epochs

    def _stack_forward(self, x, edge_index, edge_attr):
        x0 = x
        for conv, norm, proj in zip(self.convs, self.norms, self.residual_projections):
            residual = proj(x)
            x = conv(x, edge_index, edge_attr)
            if self.hparams.use_residual:
                x = x + residual
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        if self.hparams.global_skip:
            x = torch.cat([x, x0], dim=-1)
        return x

    def forward(self, x, edge_index, edge_attr):
        return self._stack_forward(x, edge_index, edge_attr)

def merge_hparams_from_json(args):
    hp_path = os.path.join(os.path.dirname(args.checkpoint), "hyperparameters.json")
    if not os.path.exists(hp_path):
        print(f"[info] no hyperparameters.json found at {hp_path}; using CLI values.", flush=True)
        return args

    try:
        with open(hp_path, "r") as f:
            hp = json.load(f)
    except Exception as e:
        print(f"[warn] failed to read hyperparameters.json ({e}); using CLI values.", flush=True)
        return args

    def set_from_hp(field, json_key=None, cast=None):
        key = json_key or field
        if key in hp:
            val = hp[key]
            if cast is not None:
                try:
                    val = cast(val)
                except Exception:
                    pass
            if not args.prefer_cli:
                setattr(args, field, val)

    set_from_hp("emb_dim",          "emb_dim",          int)
    set_from_hp("edge_dim",         "edge_dim",         int)
    set_from_hp("num_layers",       "num_layers",       int)
    set_from_hp("heads",            "heads",            int)
    set_from_hp("dropout",          "dropout",          float)
    set_from_hp("learning_rate",    "learning_rate",    float)
    set_from_hp("weight_decay",     "weight_decay",     float)
    set_from_hp("batch_size",       "batch_size",       int)
    set_from_hp("epochs",           "epochs",           int)
    set_from_hp("masking_ratio",    "masking_ratio",    float)
    set_from_hp("norm_type",        "norm_type",        str)
    set_from_hp("global_skip",      "global_skip",      bool)
    set_from_hp("use_residual",     "use_residual",     bool)

    if "num_neighbors" in hp and isinstance(hp["num_neighbors"], list) and len(hp["num_neighbors"]) > 0:
        if not args.prefer_cli:
            try:
                args.neighbors = int(hp["num_neighbors"][0])
            except Exception:
                pass

    print("[info] merged hyperparameters from checkpoint hyperparameters.json", flush=True)
    return args

def main(args):
    set_seed(args.seed)
    args = merge_hparams_from_json(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = pd.read_csv(args.graph_csv)
    print(f"Loaded graph from: {args.graph_csv} ({len(graph)} edges)", flush=True)
    
    all_nodes = read_txt(args.nodes_txt)
    tissue_cols = read_txt(args.tissues_txt)

    node_to_idx = create_node_mapping(all_nodes)
    full_edge_index, full_edge_attr = process_graph(graph, node_to_idx, tissue_cols)

    emb_dim = int(args.emb_dim)
    torch.manual_seed(args.seed)
    _embeddings_tensor = torch.empty((len(all_nodes), emb_dim), dtype=torch.float32)
    torch.nn.init.normal_(_embeddings_tensor, mean=0.0, std=0.02)

    _all_nodes_sorted = sorted(all_nodes)
    _node_to_row = {n: i for i, n in enumerate(_all_nodes_sorted)}
    embeddings_dict = {
        int(n) if isinstance(n, (np.integer, str)) and str(n).isdigit() else n:
        _embeddings_tensor[_node_to_row[n]].clone().numpy()
        for n in _all_nodes_sorted
    }

    def get_embeddings(node_to_idx, embeddings_dict):
        embedding_dim = next(iter(embeddings_dict.values())).shape[0]
        x = torch.zeros((len(node_to_idx), embedding_dim), dtype=torch.float)
    
        for node, idx in node_to_idx.items():
            if node in embeddings_dict:
                x[idx] = torch.tensor(embeddings_dict[node], dtype=torch.float).clone().detach()
            else:
                print(f'warning: node {node} missing from embeddings_dict')
    
        return x

    x_all = get_embeddings(node_to_idx, embeddings_dict)
    
    data = Data(x=x_all, edge_index=full_edge_index, edge_attr=full_edge_attr)
    data.num_nodes = x_all.shape[0]

    num_neighbors = [args.neighbors] * args.num_layers

    model = TransformerMaskedModel.load_from_checkpoint(
        args.checkpoint,
        emb_dim=args.emb_dim,
        edge_dim=args.edge_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_neighbors=num_neighbors,
        epochs=args.epochs,
        masking_ratio=args.masking_ratio,
        train_data=None,
        val_data=None,
        test_data=None,
        norm_type=args.norm_type,
        global_skip=args.global_skip,
        use_residual=args.use_residual,
    ).to(device)
    
    model.eval()
    model.freeze()

    inference_loader = NeighborLoader(
        data,
        input_nodes=torch.arange(data.num_nodes),
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        generator=torch.Generator(device='cpu').manual_seed(args.seed)
    )
    
    final_embeddings = torch.zeros((data.num_nodes, emb_dim + (emb_dim if args.global_skip else 0)), dtype=torch.float)

    with torch.no_grad():
        for batch in tqdm(inference_loader):
            batch = batch.to(device)
            out = model.forward(batch.x, batch.edge_index, batch.edge_attr)
            final_embeddings[batch.n_id.cpu()] = out.cpu()
        
    emb_dict = {}
    for node_id, idx in node_to_idx.items():
        key = normalize_entrez_id(node_id)
        val = final_embeddings[idx].detach().cpu().clone()
        emb_dict[key] = val

    outdir = os.path.dirname(args.output_pkl)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Embeddings saved to: {args.output_pkl}", flush=True)

def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract embeddings from a trained MER GNN")

    ap.add_argument("--graph_csv", type=str, required=True, help="Path to network CSV (source,target,<tissues>)")
    ap.add_argument("--nodes_txt", type=str, required=True, help="Text file with one node ID per line")
    ap.add_argument("--tissues_txt", type=str, required=True, help="Text file with one tissue column per line")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to best-checkpoint.ckpt")
    ap.add_argument("--output_pkl", type=str, required=True, help="Where to save embeddings dict (.pkl)")

    ap.add_argument("--prefer_cli", type=str2bool, default=False,
                    help="If true, CLI hyperparams override those from hyperparameters.json")

    ap.add_argument("--edge_dim", type=int, default=35)
    ap.add_argument("--emb_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--neighbors", type=int, default=75)
    ap.add_argument("--masking_ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--norm_type", choices=["layernorm", "pairnorm"], default="layernorm")
    ap.add_argument("--global_skip", type=str2bool, default=False)
    ap.add_argument("--use_residual", type=str2bool, default=True)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)