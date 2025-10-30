#!/bin/bash
#SBATCH --job-name=merge          # create a short name for your job
#SBATCH --output=slurm-%A.%a.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=3-00:00         # set the maximum runtime (1 day)
#SBATCH --mem=512G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

python 4_merge_across_tissues.py