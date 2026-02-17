<p align="center">
  <img src="mahi_logo.png" alt="MAHI Logo" width="250"/>
</p>

**Mahi** is a deep learning framework integrating chromatin features and protein structure with tissue-specific networks for context-dependent gene representation.

---
## Installation

### **Recommended Installation (using 'environment.yaml')**
```bash
# clone GitHub repository
git clone https://github.com/FunctionLab/mahi.git
cd mahi

# create Conda environment from YAML
conda env create -f environment.yaml
conda activate mahi

# install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### **Manual Installation**
```bash
# create new Conda environment
conda create --name mahi python=3.10 pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate mahi

# install dependencies
pip install "numpy<2"
pip install torch-geometric wandb pytorch-lightning ipykernel umap-learn biopython pyfaidx seaborn xgboost
conda install scikit-learn matplotlib pandas -c conda-forge

# install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# install transformers package
pip install "transformers[torch]"
```

## Demo: gene essentiality prediction
Please start by downloading the data from the following link & unzip the data: https://drive.google.com/drive/folders/1xWfPkC8bs3aQCsI6YMqYpXnSn6f6E1-B?usp=share_link
```bash
unzip <data>.zip
```

This demo runs gene essentiality prediction on **one cell line** to verify your set up (takes 15-20 minutes depending on your setup):
```bash
# attach gene essentiality labels to Mahi demo embeddings for lung tissue
python scripts/gene_essentiality/add_labels.py \
  --mahi_root data/demo/mahi_embeddings \
  --data_dir data

# evaluate gene essentiality (5-fold CV + test eval)
python scripts/gene_essentiality/evaluate_mahi_gene_essentiality.py \
  --out_dir outputs/demo \
  --mahi_root data/demo/mahi_embeddings \
  --mapping_file resources/cell_lines.txt \
  --cell_line ACH-000012                         # comment out this flag to run on all 1,183 cell lines
```

### Optional (HPC/SLURM)
For **much faster runtime on CPUs (2 minutes)**, you can also submit the demo as a SLURM job:
```bash
sbatch demo.slurm
```

### **Outputs**
```bash
outputs/demo/mahi_gene_essentiality_eval/
  ├── mahi.metrics_by_cellline_and_tissue.csv    # summary metrics on training set
  ├── cv_preds/                                  # per-gene out-of-fold predictions
  └── test_preds/                                # per-gene test predictions
```

## Mahi: End-to-end
Mahi can be run entirely on CPU (unless you are re-training the multigraph GNN). Please download the functional networks using the links from the manuscript before running Mahi.
### **Generate Mahi embeddings**
#### **Single tissue**
```bash
python wt_mahi.py \
  --dir data \
  --tissue lung \
  --checkpoint checkpoints/best-checkpoint.ckpt
```

#### **Multiple tissues**
```bash
python wt_mahi.py \
  --dir data \
  --tissues lung heart kidney \
  --checkpoint checkpoints/best-checkpoint.ckpt
```

#### **Multiple tissues from a file**
`tissues.txt`
```txt
# tissues.txt
lung
heart
colon
```

```bash
python wt_mahi.py \
  --dir data \
  --tissues_txt tissues.txt \
  --checkpoint checkpoints/best-checkpoint.ckpt
```

### **Perturbation (gene KO) analysis**
You can specify a single tissue (`--tissue`), multiple tissues (`--tissues`), or provide a tissue list file (`--tissues_txt`).
```bash
python perturb_mahi.py \
  --dir data \
  --gene <Entrez ID> \
  --tissue lung \
  --checkpoint checkpoints/best-checkpoint.ckpt
```

### **Rank perturbation effects**
You can specify a single tissue (`--tissue`), multiple tissues (`--tissues`), or provide a tissue list file (`--tissues_txt`).
```bash
python get_top_genes.py \
  --dir data \
  --gene <Entrez ID> \
  --tissue lung \
  --avg resources/averaged_distances.csv \
  --top 1000
```

