import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
import pandas as pd
import pickle
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup
import os
import json
import argparse
import random

parser = argparse.ArgumentParser(description="Extract embeddings from trained MAGE model")
parser.add_argument("--graph_csv", type=str, required=True,
                    help="Path to the graph CSV file (source, target, tissue columns)")
parser.add_argument("--output_pkl", type=str, required=True,
                    help="Path to save the output embeddings pickle file")
args = parser.parse_args()

seed = 42
pl.seed_everything(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
gen = torch.Generator(device='cpu').manual_seed(seed)

torch.use_deterministic_algorithms(True)

os.environ["WANDB_DIR"] = "/mnt/home/aaggarwal/ceph/gates_proj/mage_networks/wandb/xgboost_idea"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# hyperparameters
model_name = 'mahi_v5'
top = 'top3'

# doesn't change
edge_dim = 35

# can change
emb_dim = 512
num_layers = 4
heads = 1
dropout = 0.2
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 128
neighbors = 75
masking_ratio = 0.15
epochs = 20
norm_type = 'layernorm'
global_skip = False
use_residual = True

num_neighbors = [neighbors] * num_layers

# process the graph and the associated embeddings
graph_path = args.graph_csv
graph = pd.read_csv(graph_path)
print(f"Loaded graph from: {graph_path} ({len(graph)} edges)", flush=True)

# get all the nodes we are representing in the MAGE networks
all_nodes = pd.read_csv('/mnt/home/aaggarwal/gates_proj/target_genes/MAGE/17_xgboost_idea/resources/common_nodes.txt', header=None)
all_nodes = all_nodes[0].tolist()
#print(len(all_nodes), flush=True)

tissue_cols = pd.read_csv('/mnt/home/aaggarwal/gates_proj/target_genes/MAGE/17_xgboost_idea/resources/mage_tissues.txt', header=None)
tissue_cols = tissue_cols[0].tolist()
#print(len(tissue_cols), flush=True)

# create node-to-index mappings
def create_node_mapping(all_nodes):
    all_nodes = sorted(list(all_nodes))  # sort for consistency
    return {node: i for i, node in enumerate(all_nodes)}

node_to_idx = create_node_mapping(all_nodes)

# process graphs into pyg format
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
    all_edge_features = np.vstack((edge_features, edge_features))  # for reverse edges
    edge_attr = torch.tensor(all_edge_features, dtype=torch.float)

    return edge_index, edge_attr

full_edge_index, full_edge_attr = process_graph(graph, node_to_idx, tissue_cols)

def get_embeddings(node_to_idx, embeddings_dict):
    embedding_dim = next(iter(embeddings_dict.values())).shape[0]
    x = torch.zeros((len(node_to_idx), embedding_dim), dtype=torch.float)

    for node, idx in node_to_idx.items():
        if node in embeddings_dict:
            x[idx] = torch.tensor(embeddings_dict[node], dtype=torch.float).clone().detach()
        else:
            print(f'warning: node {node} missing from embeddings_dict')

    return x

emb_dim_trial = int(emb_dim)
torch.manual_seed(seed)
_embeddings_tensor = torch.empty((len(all_nodes), emb_dim_trial), dtype=torch.float32)
torch.nn.init.normal_(_embeddings_tensor, mean=0.0, std=0.02)

_all_nodes_sorted = sorted(all_nodes)
_node_to_row = {n: i for i, n in enumerate(_all_nodes_sorted)}
embeddings_dict = {
    int(n) if isinstance(n, (np.integer, str)) and str(n).isdigit() else n:
    _embeddings_tensor[_node_to_row[n]].clone().numpy()
    for n in _all_nodes_sorted
}

x_all = get_embeddings(node_to_idx, embeddings_dict)

# create the data objects
data = Data(x=x_all, edge_index=full_edge_index, edge_attr=full_edge_attr)
data.num_nodes = x_all.shape[0]

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

    # kept for API compatibility
    def forward(self, x, edge_index, edge_attr):
        return self._stack_forward(x, edge_index, edge_attr)

checkpoint_root = "/mnt/home/aaggarwal/ceph/gates_proj/mage_networks/wandb/xgboost_idea"
checkpoint_dir = os.path.join(checkpoint_root, model_name)
checkpoint_path = os.path.join(checkpoint_dir, "best-checkpoint.ckpt")

model = TransformerMaskedModel.load_from_checkpoint(
    checkpoint_path,
    emb_dim=emb_dim,
    edge_dim=edge_dim,
    num_layers=num_layers,
    heads=heads,
    dropout=dropout,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    batch_size=batch_size,
    num_neighbors=num_neighbors,
    epochs=epochs,
    masking_ratio=masking_ratio,
    train_data=None,
    val_data=None,
    test_data=None,
    norm_type=norm_type,
    global_skip=global_skip,
    use_residual=use_residual,
).to(device)

model.eval()
model.freeze()

print("Extracting embeddings from the trained model...", flush=True)

inference_loader = NeighborLoader(
    data,
    input_nodes=torch.arange(data.num_nodes),
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    generator=gen
)

final_embeddings = torch.zeros((data.num_nodes, emb_dim), dtype=torch.float)

with torch.no_grad():
    for batch in tqdm(inference_loader):
        batch = batch.to(device)

        # forward pass
        out = model.forward(batch.x, batch.edge_index, batch.edge_attr)

        # store output using global node indices
        final_embeddings[batch.n_id.cpu()] = out.cpu()

node_ids = list(node_to_idx.keys())

emb_dict = {}
for node_id, idx in node_to_idx.items():
    key = normalize_entrez_id(node_id)
    val = final_embeddings[idx].detach().cpu().clone()
    emb_dict[key] = val

if len(emb_dict) != len(node_to_idx):
    print("[warning] some IDs collided after normalization ('123.0' and 123)", flush=True)

# save as pickle
os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)
with open(args.output_pkl, "wb") as f:
    pickle.dump(emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Embeddings saved to: {args.output_pkl}", flush=True)
