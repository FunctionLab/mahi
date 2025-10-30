#!/bin/bash
#SBATCH --job-name=wt_mahi         # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/mahi/slurm_outputs/slurm-%A.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=1-00:00         # set the maximum runtime (1 day)
#SBATCH --mem=1200G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

module purge

conda activate target-genes

OUTDIR=/mnt/home/aaggarwal/ceph/gates_proj/wt_mahi/
WT_MULTIGRAPH=/mnt/home/aaggarwal/ceph/gates_proj/MAGE_networks/xgboost_mage_2025_networks/all_filtered_edges_across_tissues_top3.csv

mkdir -p ${OUTDIR}

# get multi-graph embeddings
python 2_get_multigraph_embed.py \
  --graph_csv ${WT_MULTIGRAPH} \
  --output_pkl ${OUTDIR}/multigraph_wt.pkl

# combine embeddings
python 3_combine_embeddings.py \
  --esm_embeddings_path /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/raw_embeddings/esm_embeddings.pkl \
  --beluga_embeddings_path /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/raw_embeddings/beluga_embeddings_single_exp.pkl \
  --graph_embeddings_path ${OUTDIR}/multigraph_wt.pkl \
  --output_embeddings_path ${OUTDIR}/esm_beluga_graph_wt.pkl \
  --which_embeddings esm beluga graph

# APPNP
python 4_generate_mahi.py \
  --perturb_gene none \
  --embeddings ${OUTDIR}/esm_beluga_graph_wt.pkl \
  --outdir ${OUTDIR} \