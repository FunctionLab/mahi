#!/bin/bash
#SBATCH --job-name=convert_and_copy          # create a short name for your job
#SBATCH --output=slurm-%A.%a.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=1-00:00:00         # set the maximum runtime (1 day)
#SBATCH --mem=64G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

tissues=(
"adipose_tissue"
"adrenal_gland"
"blood"
"blood_vessel"
"brain"
"colon"
"esophagus"
"fallopian_tube"
"heart"
"kidney"
"liver"
"lung"
"ovary"
"pancreas"
"pituitary_gland"
"prostate_gland"
"skin_of_body"
"small_intestine"
"spleen"
"stomach"
"testis"
"thyroid_gland"
"urinary_bladder"
"uterine_cervix"
"uterus"
)

# loop through each tissue and run dab.py
for tissue in "${tissues[@]}"
do
    echo "processing $tissue..."
    python /mnt/home/zpan1/ceph/for_Anusha/dab.py \
        -i "/mnt/home/zpan1/ceph/for_Anusha/MAGE_2025_scaled/${tissue}.dab" \
        -o "./${tissue}.dat"
done