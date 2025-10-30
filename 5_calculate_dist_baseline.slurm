#!/bin/bash
#SBATCH --job-name=perturb_mahi         # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/target_genes/slurm_outputs/slurm-%A_%a.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=30:00         # set the maximum runtime (1 day)
#SBATCH --mem=1200G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

module purge

conda activate target-genes

BASE_DIR=/mnt/home/aaggarwal/ceph/gates_proj/perturb_mahi/baseline
TISSUE=global
WT_PATH=/mnt/home/aaggarwal/ceph/gates_proj/wt_mahi/mahi_embeddings/${TISSUE}.pkl

for ENTREZ_DIR in ${BASE_DIR}/*; do
    if [ -d "${ENTREZ_DIR}" ]; then
        ENTREZ=$(basename "${ENTREZ_DIR}")
        OUTDIR=${ENTREZ_DIR}
        KO_PATH=${OUTDIR}/mahi_embeddings/${TISSUE}.pkl
        OUT_FILE=${OUTDIR}/distances.csv

        # skip if already processed
        if [ -f "${OUT_FILE}" ]; then
            echo "Skipping ${ENTREZ} (already done)"
            continue
        fi

        echo "processing ${ENTREZ}..."
        python 5_calculate_dist.py \
            --wt "${WT_PATH}" \
            --ko "${KO_PATH}" \
            --out "${OUT_FILE}"
    fi
done

echo "done!"

