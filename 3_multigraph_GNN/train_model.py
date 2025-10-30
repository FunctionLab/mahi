import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import pandas as pd
import pickle
import numpy as np
from torch_geometric.data import Data
import wandb
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup
from torch_geometric.nn.norm import PairNorm
from sklearn.metrics import roc_auc_score, average_precision_score

seed = 42
pl.seed_everything(seed)

os.environ["WANDB_DIR"] = "/mnt/home/aaggarwal/ceph/gates_proj/mage_networks/wandb/xgboost_idea"

# hyperparameters
project_name = 'mahi_experiments'
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
neighbors = 20
masking_ratio = 0.15
epochs = 20
norm_type = 'layernorm'
global_skip = False
use_residual = True

num_neighbors = [neighbors] * num_layers

# process the graph and the associated embeddings
graph_path = f'/mnt/home/aaggarwal/ceph/gates_proj/MAGE_networks/xgboost_mage_2025_networks/all_filtered_edges_across_tissues_{top}.csv'
graph = pd.read_csv(graph_path)

connected_nodes = np.unique(graph[['source', 'target']].values.flatten())
print(len(connected_nodes))

# get all the nodes we are representing in the MAGE networks
all_nodes = pd.read_csv('/mnt/home/aaggarwal/gates_proj/target_genes/MAGE/17_xgboost_idea/resources/common_nodes.txt', header=None)
all_nodes = all_nodes[0].tolist()
print(len(all_nodes), flush=True)

tissue_cols = pd.read_csv('/mnt/home/aaggarwal/gates_proj/target_genes/MAGE/17_xgboost_idea/resources/mage_tissues.txt', header=None)
tissue_cols = tissue_cols[0].tolist()
print(len(tissue_cols), flush=True)

# split the nodes: train (70%), val (10%), test (20%)
np.random.seed(42)
np.random.shuffle(connected_nodes)

num_train = int(0.7 * len(connected_nodes))
num_val = int(0.1 * len(connected_nodes))

train_nodes = set(connected_nodes[:num_train])
val_nodes = set(connected_nodes[num_train:num_train + num_val])
test_nodes = set(connected_nodes[num_train + num_val:])

# assign unconnected nodes
unconnected_nodes = list(set(all_nodes) - set(connected_nodes))
np.random.shuffle(unconnected_nodes)
num_train_unconn = int(0.7 * len(unconnected_nodes))
num_val_unconn = int(0.1 * len(unconnected_nodes))

train_nodes.update(unconnected_nodes[:num_train_unconn])
val_nodes.update(unconnected_nodes[num_train_unconn:num_train_unconn + num_val_unconn])
test_nodes.update(unconnected_nodes[num_train_unconn + num_val_unconn:])

# filter graphs to only edges where both nodes are in the same set
def get_subgraph(graph, nodes):
    return graph[(graph['source'].isin(nodes)) & (graph['target'].isin(nodes))]

train_graph = get_subgraph(graph, train_nodes)
val_graph = get_subgraph(graph, val_nodes)
test_graph = get_subgraph(graph, test_nodes)

# create node-to-index mappings
def create_node_mapping(all_nodes):
    all_nodes = sorted(list(all_nodes))  # sort for consistency
    return {node: i for i, node in enumerate(all_nodes)}

train_node_to_idx = create_node_mapping(train_nodes)
val_node_to_idx = create_node_mapping(val_nodes)
test_node_to_idx = create_node_mapping(test_nodes)

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

train_edges, train_attr = process_graph(train_graph, train_node_to_idx, tissue_cols)
val_edges, val_attr = process_graph(val_graph, val_node_to_idx, tissue_cols)
test_edges, test_attr = process_graph(test_graph, test_node_to_idx, tissue_cols)

def mask_undirected_edges(edge_index, masking_ratio, device):
    # edge_index: [2, E] where both directions are present
    src, dst = edge_index
    undirected = torch.stack([torch.min(src, dst), torch.max(src, dst)], dim=0)
    # unique unordered pairs and mapping back to original indices
    und_edges, inv = torch.unique(undirected, dim=1, return_inverse=True)
    num_und = und_edges.size(1)

    und_mask = (torch.rand(num_und, device=device) < masking_ratio)
    edge_mask = und_mask[inv]  # broadcast to both directions

    if edge_mask.sum() == 0:  # guarantee at least one masked edge
        edge_mask[torch.randint(0, edge_mask.numel(), (1,), device=device)] = True
    return edge_mask

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

    def training_step(self, batch, batch_idx):
        ei = batch.edge_index.to(self.device)
        ea = batch.edge_attr.to(self.device)
        x  = batch.x.to(self.device)

        edge_mask = mask_undirected_edges(ei, self.hparams.masking_ratio, self.device)
        ei_keep, ea_keep = ei[:, ~edge_mask], ea[~edge_mask]
        z = self._stack_forward(x, ei_keep, ea_keep)

        src, dst = ei[:, edge_mask]
        logits = self.edge_predictor(torch.cat([z[src], z[dst]], dim=-1))
        labels = ea[edge_mask]
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        ei = batch.edge_index.to(self.device)
        ea = batch.edge_attr.to(self.device)
        x  = batch.x.to(self.device)

        edge_mask = mask_undirected_edges(ei, self.hparams.masking_ratio, self.device)
        z = self._stack_forward(x, ei[:, ~edge_mask], ea[~edge_mask])

        src, dst = ei[:, edge_mask]
        logits = self.edge_predictor(torch.cat([z[src], z[dst]], dim=-1))
        labels = ea[edge_mask]
        val_loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.val_outputs.append({"logits": logits.detach().cpu(), "labels": labels.detach().cpu()})

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        logits = torch.cat([o["logits"] for o in self.val_outputs], dim=0)
        labels = torch.cat([o["labels"] for o in self.val_outputs], dim=0)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = labels.cpu().numpy()
        try:
            auc_macro = roc_auc_score(labels, probs, average="macro")
            auc_micro = roc_auc_score(labels, probs, average="micro")
            pr_auc_macro = average_precision_score(labels, probs, average="macro")
            pr_auc_micro = average_precision_score(labels, probs, average="micro")
        except ValueError:
            auc_macro = auc_micro = pr_auc_macro = pr_auc_micro = 0.0
        self.log("val_auc_macro", auc_macro, prog_bar=True)
        self.log("val_auc_micro", auc_micro)
        self.log("val_pr_auc_macro", pr_auc_macro)
        self.log("val_pr_auc_micro", pr_auc_micro)
        self.val_outputs.clear()

    def test_step(self, batch, batch_idx):
        ei = batch.edge_index.to(self.device)
        ea = batch.edge_attr.to(self.device)
        x  = batch.x.to(self.device)

        edge_mask = mask_undirected_edges(ei, self.hparams.masking_ratio, self.device)
        z = self._stack_forward(x, ei[:, ~edge_mask], ea[~edge_mask])

        src, dst = ei[:, edge_mask]
        logits = self.edge_predictor(torch.cat([z[src], z[dst]], dim=-1))
        labels = ea[edge_mask]
        test_loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log("test_loss", test_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.test_outputs.append({"logits": logits.detach().cpu(), "labels": labels.detach().cpu()})

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
        logits = torch.cat([o["logits"] for o in self.test_outputs], dim=0)
        labels = torch.cat([o["labels"] for o in self.test_outputs], dim=0)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = labels.cpu().numpy()
        try:
            auc_macro = roc_auc_score(labels, probs, average="macro")
            auc_micro = roc_auc_score(labels, probs, average="micro")
            pr_auc_macro = average_precision_score(labels, probs, average="macro")
            pr_auc_micro = average_precision_score(labels, probs, average="micro")
        except ValueError:
            auc_macro = auc_micro = pr_auc_macro = pr_auc_micro = 0.0
        self.log("test_auc_macro", auc_macro, prog_bar=True)
        self.log("test_auc_micro", auc_micro)
        self.log("test_pr_auc_macro", pr_auc_macro)
        self.log("test_pr_auc_micro", pr_auc_micro)
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        steps_per_epoch = len(self.train_dataloader())
        total_steps = steps_per_epoch * self.hparams.epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def train_dataloader(self):
        return NeighborLoader(self.train_data, input_nodes=torch.arange(self.train_data.num_nodes), num_neighbors=self.hparams.num_neighbors, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return NeighborLoader(self.val_data, input_nodes=torch.arange(self.val_data.num_nodes), num_neighbors=self.hparams.num_neighbors, batch_size=self.hparams.batch_size, shuffle=False)

    def test_dataloader(self):
        return NeighborLoader(self.test_data, input_nodes=torch.arange(self.test_data.num_nodes), num_neighbors=self.hparams.num_neighbors, batch_size=self.hparams.batch_size, shuffle=False)

def train_model(project_name, model_name):
    wandb_logger = WandbLogger(
        project=project_name,
        name=model_name,
        save_dir=os.environ["WANDB_DIR"],
        log_model="all"
    )

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

    def get_embeddings(node_to_idx, embeddings_dict):
        embedding_dim = next(iter(embeddings_dict.values())).shape[0]
        x = torch.zeros((len(node_to_idx), embedding_dim), dtype=torch.float)
    
        for node, idx in node_to_idx.items():
            if node in embeddings_dict:
                x[idx] = torch.tensor(embeddings_dict[node], dtype=torch.float).clone().detach()
            else:
                print(f'warning: node {node} missing from embeddings_dict')
    
        return x

    x_train = get_embeddings(train_node_to_idx, embeddings_dict)
    x_val   = get_embeddings(val_node_to_idx,   embeddings_dict)
    x_test  = get_embeddings(test_node_to_idx,  embeddings_dict)

    train_data = Data(x=x_train, edge_index=train_edges, edge_attr=train_attr)
    val_data   = Data(x=x_val,   edge_index=val_edges,   edge_attr=val_attr)
    test_data  = Data(x=x_test,  edge_index=test_edges,  edge_attr=test_attr)
    train_data.num_nodes = train_data.x.size(0)
    val_data.num_nodes   = val_data.x.size(0)
    test_data.num_nodes  = test_data.x.size(0)

    model = TransformerMaskedModel(
            emb_dim=emb_dim_trial,               # use sweep value
            edge_dim=edge_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_neighbors=[neighbors] * num_layers,
            epochs=epochs,
            masking_ratio=masking_ratio,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            norm_type=norm_type,
            global_skip=global_skip,
            use_residual=use_residual
        )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True
    )

    checkpoint_root = "/mnt/home/aaggarwal/ceph/gates_proj/mage_networks/wandb/xgboost_idea"
    checkpoint_dir = os.path.join(checkpoint_root, model_name)

    os.makedirs(checkpoint_dir, exist_ok=True)

    hyperparams_path = os.path.join(checkpoint_dir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(model.hparams, f, indent=4)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        filename="best-checkpoint"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint_callback],
        default_root_dir=os.environ["WANDB_DIR"],
        accumulate_grad_batches=4
    )

    trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    train_model(project_name, model_name)