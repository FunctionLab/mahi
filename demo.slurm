#!/bin/bash
#SBATCH --job-name=demo         # create a short name for your job
#SBATCH --output=slurm-%A.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=1:00:00         # set the maximum runtime
#SBATCH --mem=512G                 # request memory (adjust as needed)

conda activate mahi

# attach gene essentiality labels to Mahi demo embeddings for lung tissue
python scripts/gene_essentiality/add_labels.py \
  --mahi_root data/demo/mahi_embeddings_lung \
  --data_dir data/demo

# evaluate gene essentiality (5-fold CV + test eval)
python scripts/gene_essentiality/evaluate_mahi_gene_essentiality.py \
  --out_dir outputs/demo \
  --mahi_root data/demo/mahi_embeddings_lung \
  --mapping_file resources/cell_lines.txt \
  --cell_line ACH-000012 # cell line associated with lung tissue