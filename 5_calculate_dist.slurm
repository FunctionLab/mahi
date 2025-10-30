#!/bin/bash
#SBATCH --job-name=perturb_mahi         # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/target_genes/slurm_outputs/slurm-%A_%a.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=5:00         # set the maximum runtime (1 day)
#SBATCH --mem=1200G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

module purge

conda activate target-genes

ENTREZ=672
GENE_NAME=BRCA1
TISSUE=mammary_gland
OUTDIR=/mnt/home/aaggarwal/ceph/gates_proj/perturb_mahi/${GENE_NAME}_${ENTREZ}

# calculate distances
python 5_calculate_dist.py \
  --wt /mnt/home/aaggarwal/ceph/gates_proj/wt_mahi/mahi_embeddings/${TISSUE}.pkl \
  --ko ${OUTDIR}/mahi_embeddings/${TISSUE}.pkl \
  --out ${OUTDIR}/distances.csv

