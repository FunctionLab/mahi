<p align="center">
  <img src="mahi_logo.png" alt="MAHI Logo" width="250"/>
</p>

**Mahi** is a deep learning framework integrating genomic sequence and tissue-specific gene interactions for functional genomics.

---
## Installation

### **Recommended Installation (using 'environment.yaml')**
```bash
# clone GitHub repository
git clone https://github.com/anushaggs/target_genes.git
cd target_genes

# create Conda environment from YAML
conda env create -f environment.yaml
conda activate target-genes

# install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### **Manual Installation**
```bash
# create new Conda environment
conda create --name target-genes python=3.10 pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate target-genes

# install dependencies
pip install "numpy<2"
pip install torch-geometric wandb pytorch-lightning ipykernel umap-learn biopython pyfaidx seaborn xgboost
conda install scikit-learn matplotlib pandas -c conda-forge

# install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## Data Used for Mahi
The model is trained using the following datasets:
- **Reference Genome:** hg38 (https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/)
- **ESM-C (600M parameters):** Pretrained protein embeddings.
- **GTEx v10:** Tissue-specific gene expression data.
- **MAGE_2025_scaled Functional Networks:** Gene interaction and functional association data.
- **DepMap Public 24Q4:** CRISPR gene dependency data (`CRISPRGeneDependency.csv`).
- **COSMIC release v101, 19th Nov 2024:** Cancer genes & type data.
- **CCLE & CCLE-v2 Public data:** Unfiltered VCF files for 777 cell lines (https://app.terra.bio/#workspaces/broad-firecloud-ccle/CCLE-public)
- **Housekeeping genes:** https://housekeeping.unicamp.br/?download
- **The Human Protein Atlas:**

## NOTES FOR ME
- esm env has to be separate
- pyfaidx needs to be installed as well
- seaborn needs to be installed as well
- transformers[torch] needs to be installed as well
- xgboost will need to be installed as well
- joypy need to be pip installed

