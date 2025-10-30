import torch
import pickle
from typing import List, Optional
import pandas as pd
from Bio.Seq import Seq
from pyfaidx import Fasta
import os
import re
import argparse

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

        if not re.fullmatch(r'[ACGTN]+', seq):
            print(f"Skipping gene {entrez_id} — invalid base found in sequence", flush=True)
            return None
    
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
    transcript_df = cds_df[cds_df['transcript_id'] == transcript_id]
    cds_df = transcript_df[transcript_df['feature'] == 'CDS']
    start_df = transcript_df[transcript_df['feature'] == 'start_codon']
    stop_df = transcript_df[transcript_df['feature'] == 'stop_codon']

    if cds_df.empty or start_df.empty or stop_df.empty:
        print(f"[{transcript_id}] Missing CDS or codon annotations", flush=True)
        return None 

    if chrom not in fasta_index:
        print(f"[{transcript_id}] Chromosome {chrom} not found in FASTA", flush=True)
        return None

    genome = fasta_index[chrom]
    chrom_len = len(genome[chrom])

    start_codon_start = start_df.iloc[0]['start']
    start_codon_end = start_df.iloc[0]['end']
    stop_codon_start = stop_df.iloc[0]['start']
    stop_codon_end = stop_df.iloc[0]['end']

    if strand == '+':
        cds_df = cds_df[
            (cds_df['end'] >= start_codon_start) &
            (cds_df['start'] <= stop_codon_end)
        ].sort_values(by='start')
    else:
        cds_df = cds_df[
            (cds_df['end'] >= stop_codon_start) &
            (cds_df['start'] <= start_codon_end)
        ].sort_values(by='start', ascending=False)

    if cds_df.empty:
        print(f"[{transcript_id}] No CDSs after trimming by start/stop codons", flush=True)
        return None

    cds_rows = cds_df.to_dict('records') # to preserve the order
    seq_parts = []

    for i, row in enumerate(cds_rows):
        start = row['start']
        end = row['end']
        frame = int(row['frame']) if pd.notna(row['frame']) and str(row['frame']).isdigit() else 0

        if strand == '+':
            if start <= start_codon_start <= end:
                start = start_codon_start
            if start <= stop_codon_end <= end:
                end = stop_codon_end
        else:
            if start <= start_codon_end <= end:
                end = start_codon_end
            if start <= stop_codon_start <= end:
                start = stop_codon_start

        if end <= start:
            print(f"[{transcript_id}] Skipping — invalid CDS coordinates ({start}, {end})", flush=True)
            return None

        exon_seq = genome[chrom][start - 1:end].seq.upper()

        if i == 0:
            exon_seq = exon_seq[frame:]
        
        #exon_seq = exon_seq[frame:]
        seq_parts.append(exon_seq)
    
    if strand == '-':
        seq_parts = seq_parts[::-1]
        full_seq = ''.join(seq_parts)
        full_seq = str(Seq(full_seq).reverse_complement())
    else:
        full_seq = ''.join(seq_parts)

    if not full_seq:
        print(f"[{transcript_id}] Sequence empty after CDS extraction", flush=True)
        return None

    return full_seq

# TO DO -- figure out why we are getting (*) in the amino acid sequence
def translate_to_protein(entrez_id, genic_sequence):
    seq = genic_sequence.replace('\n', '').replace(' ', '').upper()

    if len(seq) % 3 != 0:
        print(f"[{entrez_id}] Sequence length not divisible by 3 — skipping.")
        return None

    protein = str(Seq(seq).translate(to_stop=False))

    if not protein.startswith('M'):
        print(f"[{entrez_id}] Protein does not start with 'M' — skipping.")
        return None

    if '*' in protein[:-1]:  # allow stop only at very end
        print(f"[{entrez_id}] Internal stop codon found — skipping.")
        return None

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
        if genic_sequence:
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
    parser = argparse.ArgumentParser(description="Extract sequences for ESM, Beluga, and Seqweaver models.")
    parser.add_argument("--gtf_transcript_path", required=True, help="Path to transcript-level GTF file (CSV format)")
    parser.add_argument("--gtf_cds_path", required=True, help="Path to CDS-level GTF file (CSV format)")
    parser.add_argument("--genome_fasta_dir", required=True, help="Directory containing chromosome FASTA files")
    parser.add_argument("--output_esm_fasta", required=True, help="Output FASTA file path for ESM protein sequences")
    parser.add_argument("--output_beluga_fasta", required=True, help="Output FASTA file path for Beluga input sequences")
    parser.add_argument("--output_seqweaver_fasta", required=True, help="Output FASTA file path for Seqweaver input sequences")

    args = parser.parse_args()

    main(
        args.gtf_transcript_path,
        args.gtf_cds_path,
        args.genome_fasta_dir,
        args.output_esm_fasta,
        args.output_beluga_fasta,
        args.output_seqweaver_fasta,
    )