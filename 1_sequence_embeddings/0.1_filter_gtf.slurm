#!/bin/bash
#SBATCH --job-name=filter_gtf          # create a short name for your job
#SBATCH --output=/mnt/home/aaggarwal/gates_proj/target_genes/slurm_outputs/slurm-%A_%a.out
#SBATCH --nodes=1                 # request one node
#SBATCH --time=30:00         # set the maximum runtime (1 day)
#SBATCH --mem=512G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

# -----------------------------------------------------------------------------------
# STEP 1: chromosome filtering and renaming
awk -F'\t' '
BEGIN {
  while ((getline line < "/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/chrom_map.txt") > 0) {
    split(line, a, " ")
    chrom_map[a[1]] = a[2]
  }
}
$1 in chrom_map {
  $1 = chrom_map[$1]
  OFS = "\t"
  print
}
' /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.gtf > /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.chroms.gtf

# -----------------------------------------------------------------------------------
# STEP 2: get the gene symbols for protein coding genes
awk -F'\t' '$3 == "gene" && $9 ~ /gene_biotype "protein_coding"/ {
    if (match($9, /gene_id "([^"]+)"/, a)) {
        print a[1]
    }
}' /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.chroms.gtf | sort -u \
> /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/protein_coding_gene_symbols.txt

# -----------------------------------------------------------------------------------
# Step 3: extract the MANE Select (canonical transcripts) that belong to the protein-coding genes
awk -F'\t' '
BEGIN {
    while ((getline id < "/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/protein_coding_gene_symbols.txt") > 0)
        gene[id] = 1
}
$3 == "transcript" && $9 ~ /tag "MANE Select"/ {
    if (match($9, /gene_id "([^"]+)"/, g) && g[1] in gene &&
        match($9, /transcript_id "([^"]+)"/, t)) {
        print t[1]
    }
}
' /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.chroms.gtf | sort -u \
> /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/mane_select_transcript_ids.txt

# -----------------------------------------------------------------------------------
# STEP 4: filter for full transcript entries (MANE + protein-coding only)
awk -F'\t' '
BEGIN {
  while ((getline id < "/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/mane_select_transcript_ids.txt") > 0)
    tx[id] = 1
}
$3 == "transcript" {
  if (match($9, /transcript_id "([^"]+)"/, arr) && arr[1] in tx) {
    print
  }
}
' /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.chroms.gtf \
> /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.final_transcript.gtf

# -----------------------------------------------------------------------------------
# STEP 5: Filter CDS/start_codon/stop_codon entries for those transcripts
awk -F'\t' '
BEGIN {
  while ((getline id < "/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/mane_select_transcript_ids.txt") > 0)
    tx[id] = 1
}
($3 == "CDS" || $3 == "start_codon" || $3 == "stop_codon") {
  if (match($9, /transcript_id "([^"]+)"/, arr) && arr[1] in tx) {
    print
  }
}
' /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.chroms.gtf \
> /mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.final_CDS.gtf

