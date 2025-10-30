import torch
import pickle
from typing import List, Optional
import pandas as pd
from Bio.Seq import Seq
from pyfaidx import Fasta
import os

def load_gtf(gtf_path):
    # load the GTF as pandas dataframe

    col_names = [
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attributes", "gene_id", "transcript_id", "entrez_id"
    ]
    
    gtf_df = pd.read_csv(gtf_path)

    return gtf_df

def extract_gene_regions(gtf_df):
    # filter to protein-coding genes and extract
    # - entrez id
    # - chrom
    # - tss
    # - gene start
    # - gene end

    entrez_id = gtf_df['entrez_id'].astype(str)
    chrom = gtf_df['seqname']
    strand = gtf_df['strand']
    tss = gtf_df['start'].where(strand == '+', gtf_df['end'])
    gene_start = gtf_df['start']
    gene_end = gtf_df['end']
    transcript_id = gtf_df['transcript_id']
    
    return zip(entrez_id, transcript_id, chrom, strand, tss, gene_start, gene_end)

def extract_tss_sequences(entrez_id, chrom, tss, strand, fasta_index, beluga_window=41_800, seqweaver_window=1_000):
    # given the chromosome, strand, and the TSS, extract sequence +/- window size
    # return None if invalid

    # if the chromosome in the gtf is not found in the fasta files, skip it
    # make sure the chromosome files are aligned with those in the gtf file
    if chrom not in fasta_index:
            print(f"Skipping gene {entrez_id} — no FASTA for {chrom}", flush=True)
            return None, None
    
    genome = fasta_index[chrom]
    chrom_len = len(genome[chrom])

    # get the sequence start and end site from the genome and extract the sequence
    # adjust boundaries if close to the edge of the chromosome
    def get_window_sequence(center, length):
        half = length // 2
        seq_start = center - half
        seq_end = center + half
    
        if seq_start < 0:
                seq_end += abs(seq_start)
                seq_start = 0
        if seq_end > chrom_len:
            shift = seq_end - chrom_len
            seq_start -= shift
            seq_end = chrom_len
            if seq_start < 0:
                print(f"Skipping {entrez_id} — cannot fit full window on chromosome", flush=True)
                return None
    
        # make the sequence uppercase for consistency
        seq = genome[chrom][seq_start:seq_end].seq.upper()
    
        # make sure the length lines up
        if len(seq) != length:
                print(f"Skipping gene {entrez_id} — final length {len(seq)}", flush=True)
                return None

        return seq

    # get beluga and seweaver embeddings
    seq_beluga = get_window_sequence(tss, beluga_window)
    seq_seqweaver = get_window_sequence(tss, seqweaver_window)

    # get the reverse complement of the sequence if the strand is negative
    if strand == '-' and seq_beluga is not None:
        seq_beluga = str(Seq(seq_beluga).reverse_complement())
    if strand == '-' and seq_seqweaver is not None:
        seq_seqweaver = str(Seq(seq_seqweaver).reverse_complement())
    
    return seq_beluga, seq_seqweaver

def extract_cds_sequence(transcript_id, chrom, strand, cds_df, fasta_index):
    entries = cds_df[cds_df['transcript_id'] == transcript_id]
    if entries.empty or chrom not in fasta_index:
        return None

    genome = fasta_index[chrom]
    chrom_len = len(genome[chrom])

    # sort CDS entries in transcript (5' to 3') order
    if strand == '+':
        entries = entries.sort_values(by='start')
    else:
        entries = entries.sort_values(by='end', ascending=False)

    seq_parts = []
    for _, row in entries.iterrows():
        start = row['start']
        end = row['end']
        frame = str(row['frame']) if pd.notna(row['frame']) else '0'

        if start < 1 or end > chrom_len:
            print(f"Skipping transcript {transcript_id} — invalid CDS coordinates", flush=True)
            return None

        exon_seq = genome[chrom][start - 1:end].seq.upper()

        # apply frame (phase)
        if frame in {'0', '1', '2'}:
            exon_seq = exon_seq[int(frame):]
        elif pd.notna(frame):
            print(f"Invalid frame value: {frame} for transcript {transcript_id}")
            return None

        seq_parts.append(exon_seq)

    full_seq = ''.join(seq_parts)

    # reverse complement only once at the end, for minus strand
    if strand == '-':
        full_seq = str(Seq(full_seq).reverse_complement())

    return full_seq

# TO DO -- figure out why we are getting (*) in the amino acid sequence
def translate_to_protein(entrez_id, genic_sequence):
    # translate the coding sequence into protein
    # return None if the start codon is not found or sequence is not a multiple of 3

    seq = genic_sequence.replace('\n', '').replace(' ', '').upper()

    # find the first start codon
    start_idx = seq.find('ATG')
    if start_idx == -1:
        print(f"Skipping gene {entrez_id} — no ATG start codon found", flush=True)
        return None

    trimmed_seq = seq[start_idx:]

    protein = str(Seq(trimmed_seq).translate(to_stop=False))

    return protein

def main(
    gtf_transcript_path: str,
    gtf_cds_path: str,
    genome_fasta_dir: str,
    output_esm_fasta: str,
    output_beluga_fasta: str,
    output_seqweaver_fasta: str,
):
    # load the genome
    fasta_files = {
        f.replace(".fa", ""): os.path.join(genome_fasta_dir, f)
        for f in os.listdir(genome_fasta_dir) if f.endswith(".fa")
    }
    fasta_index = {k: Fasta(v) for k, v in fasta_files.items()}

    # load GTF
    gtf_transcripts = load_gtf(gtf_transcript_path)
    gtf_cds = load_gtf(gtf_cds_path)

    # get gene regions for transcripts
    gene_regions = extract_gene_regions(gtf_transcripts)

    # output file handles
    esm_out = open(output_esm_fasta, "w")
    beluga_out = open(output_beluga_fasta, "w")
    seqweaver_out = open(output_seqweaver_fasta, "w")

    # iterate through genes and get genomic & protein sequences
    for i, (entrez_id, transcript_id, chrom, strand, tss, gene_start, gene_end) in enumerate(gene_regions):
        
        # progress update
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} genes...", flush=True)
            
        # protein sequence (for ESM)
        genic_sequence = extract_cds_sequence(transcript_id, chrom, strand, gtf_cds, fasta_index)
        protein_sequence = translate_to_protein(entrez_id, genic_sequence)
        
        if protein_sequence:
            esm_out.write(f">{entrez_id}\n{protein_sequence}\n")

        # TSS-centered sequence (for Beluga & Seqweaver)
        seq_beluga, seq_seqweaver = extract_tss_sequences(
            entrez_id, chrom, tss, strand, fasta_index
        )
        
        if seq_beluga:
            beluga_out.write(f">{entrez_id}\n{seq_beluga}\n")

        if seq_seqweaver:
            seqweaver_out.write(f">{entrez_id}\n{seq_seqweaver}\n")

    esm_out.close()
    beluga_out.close()
    seqweaver_out.close()
    print("Fasta files written for sequence input into esm, beluga, and seqweaver!")

if __name__ == "__main__":
    gtf_transcript_path = "/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/hg38.p14.ncbiRefSeq.transcript_final_gtf.csv"
    gtf_cds_path = "/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/hg38.p14.ncbiRefSeq.CDS_final_gtf.csv"
    genome_fasta_dir = "/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14"
    output_esm_fasta = "/mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/fasta_files/esm_sequences.fasta"
    output_beluga_fasta = "/mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/fasta_files/beluga_sequences.fasta"
    output_seqweaver_fasta = "/mnt/home/aaggarwal/ceph/gates_proj/GNN_model_files/fasta_files/seqweaver_sequences.fasta"

    main(
        gtf_transcript_path,
        gtf_cds_path,
        genome_fasta_dir,
        output_esm_fasta,
        output_beluga_fasta,
        output_seqweaver_fasta,
    )