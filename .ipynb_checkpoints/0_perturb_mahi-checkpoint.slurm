#!/bin/bash
#SBATCH --job-name=perturb_mahi         # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/mahi/slurm_outputs/slurm-%A.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=15:00         # set the maximum runtime (1 day)
#SBATCH --mem=1200G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

module purge

conda activate target-genes

ENTREZ=122319436
GENE_NAME=ALPK3
OUTDIR=/mnt/home/aaggarwal/ceph/gates_proj/perturb_mahi/${GENE_NAME}_${ENTREZ}

mkdir -p ${OUTDIR}

# do gene KO on graph
python 1_do_gene_KO.py \
  --perturb_gene ${ENTREZ} \
  --output_csv ${OUTDIR}/multigraph_perturb.csv

# get multi-graph embeddings
python 2_get_multigraph_embed.py \
  --graph_csv ${OUTDIR}/multigraph_perturb.csv \
  --output_pkl ${OUTDIR}/multigraph_perturb.pkl

# combine embeddings
python 3_combine_embeddings.py \
  --esm_embeddings_path /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/raw_embeddings/esm_embeddings.pkl \
  --beluga_embeddings_path /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/raw_embeddings/beluga_embeddings_single_exp.pkl \
  --graph_embeddings_path ${OUTDIR}/multigraph_perturb.pkl \
  --output_embeddings_path ${OUTDIR}/esm_beluga_graph_perturb.pkl \
  --which_embeddings esm beluga graph

# APPNP
python 4_generate_mahi.py \
  --perturb_gene ${ENTREZ} \
  --embeddings ${OUTDIR}/esm_beluga_graph_perturb.pkl \
  --outdir ${OUTDIR} \
  --tissue liver

