#!/bin/bash
#SBATCH --job-name=param_sweep    # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/target_genes/slurm_outputs/slurm-%A_%a.out
#SBATCH --gres=gpu:1  # request 4 GPUs
#SBATCH --constraint=a100-80gb&sxm4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # increase number of CPUs per task
#SBATCH --time=4:00:00
#SBATCH --mem=512G
#SBATCH --array=0-4
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

module purge
conda activate target-genes

export PROJECT_NAME="mahi_maked_edge_reconstruction_v7"

mkdir -p sweep_ids
mkdir -p skipped_configs

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "creating sweep..."
  python create_sweep.py
  echo "sweep created - waiting for sweep id to be saved..."
  sleep 10
fi

sleep 15

python run_sweep_loader.py