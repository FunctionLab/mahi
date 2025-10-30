#!/bin/bash
#SBATCH --job-name=combine_embeddings         # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/target_genes/slurm_outputs/slurm-%A_%a.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=30:00         # set the maximum runtime (1 day)
#SBATCH --mem=512G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

module purge

conda activate target-genes

python 3_combine_embeddings.py \
  --esm_embeddings_path /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/raw_embeddings/esm_embeddings.pkl \
  --beluga_embeddings_path /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/raw_embeddings/beluga_embeddings_single_exp.pkl \
  --output_embeddings_path /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/combined_embeddings/esm_beluga_embeddings_single_exp.pkl \
  --output_nodes_path /mnt/home/aaggarwal/gates_proj/target_genes/MAGE/17_xgboost_idea/resources/common_nodes.txt \
  --which_embeddings esm beluga