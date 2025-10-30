#!/bin/bash
#SBATCH --job-name=extract_sequences         # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/target_genes/slurm_outputs/slurm-%A_%a.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=15:00         # set the maximum runtime (1 day)
#SBATCH --mem=512G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

python 1_extract_sequences.py \
  --gtf_transcript_path /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/hg38.p14.ncbiRefSeq.transcript_final_gtf.csv \
  --gtf_cds_path /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/hg38.p14.ncbiRefSeq.CDS_final_gtf.csv \
  --genome_fasta_dir /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14 \
  --output_esm_fasta /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/fasta_files/esm_sequences.fasta \
  --output_beluga_fasta /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/fasta_files/beluga_sequences.fasta \
  --output_seqweaver_fasta /mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/fasta_files/seqweaver_sequences.fasta