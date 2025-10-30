#!/bin/bash
#SBATCH --job-name=get_genomic_embed    # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/target_genes/slurm_outputs/slurm-%A_%a.out
#SBATCH --gres=gpu:1  # request 4 GPUs
#SBATCH --constraint=a100-80gb&sxm4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  # increase number of CPUs per task
#SBATCH --time=2:15:00
#SBATCH --mem=512G
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

module purge

conda activate target-genes

#python 2_generate_beluga_embeddings.py \
#  --input_fasta /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/fasta_files/beluga_sequences.fasta \
#  --output_pkl /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/raw_embeddings/beluga_embeddings.pkl \
#  --beluga_model_path /mnt/home/aaggarwal/ceph/gates_proj/github_clones/ExPecto/resources/deepsea.beluga.pth \
#  --pca_model_out /mnt/home/aaggarwal/gates_proj/target_genes/MAGE/17_xgboost_idea/1_generate_embeddings/models/beluga_pca_model.pkl \
#  --pca_dim 1152

python 2_generate_beluga_embeddings.py \
  --input_fasta /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/fasta_files/beluga_sequences.fasta \
  --output_pkl /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/raw_embeddings/beluga_embeddings_single_exp.pkl \
  --beluga_model_path /mnt/home/aaggarwal/ceph/gates_proj/github_clones/ExPecto/resources/deepsea.beluga.pth \