#!/bin/bash
#SBATCH --job-name=convert_and_copy_mage          # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/target_genes/slurm_outputs/slurm-%A_%a.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=2-00:00         # set the maximum runtime (1 day)
#SBATCH --mem=64G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

while IFS= read -r tissue; do
    echo "processing $tissue..."
    python /mnt/home/zpan1/ceph/for_Anusha/dab.py \
        -i "/mnt/home/zpan1/ceph/for_Anusha/MAGE_2025_scaled/${tissue}.dab" \
        -o "/mnt/home/aaggarwal/ceph/gates_proj/MAGE_networks/xgboost_mage_2025_networks/${tissue}.dat"
done < /mnt/home/aaggarwal/gates_proj/target_genes/MAGE/17_xgboost_idea/resources/mage_tissues.txt