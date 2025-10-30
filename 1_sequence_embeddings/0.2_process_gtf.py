import pandas as pd

# processing the GTF file with CDS, start codons, and stop codon regions
gtf_cds_path = '/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.final_CDS.gtf'

col_names = [
    "seqname", "source", "feature", "start", "end",
    "score", "strand", "frame", "attributes"
]

gtf_cds = pd.read_csv(
    gtf_cds_path,
    sep='\t',
    comment='#',
    names=col_names,
    dtype=str
)

gtf_cds['gene_id'] = gtf_cds['attributes'].str.extract(r'gene_id "([^"]+)"')
gtf_cds['transcript_id'] = gtf_cds['attributes'].str.extract(r'transcript_id "([^"]+)"')
gtf_cds['entrez_id'] = gtf_cds['attributes'].str.extract(r'GeneID:(\d+)')

gtf_cds["start"] = gtf_cds["start"].astype(int)
gtf_cds["end"] = gtf_cds["end"].astype(int)
gtf_cds["length"] = gtf_cds["end"] - gtf_cds["start"] + 1

# identify valid start/stop codons
valid_codons = gtf_cds[
    (gtf_cds["feature"].isin(["start_codon", "stop_codon"])) &
    (gtf_cds["frame"] == "0") &
    (gtf_cds["length"] == 3)
]

# count how many of each valid codon per transcript
valid_counts = valid_codons.groupby(["transcript_id", "feature"]).size().unstack(fill_value=0)

# keep only transcripts that have both good start and stop codons
transcripts_to_keep = valid_counts[
    (valid_counts.get("start_codon", 0) > 0) & 
    (valid_counts.get("stop_codon", 0) > 0)
].index

# filter the CDS GTF to only include rows for those transcripts
gtf_cds_filtered = gtf_cds[gtf_cds["transcript_id"].isin(transcripts_to_keep)].copy()

gtf_cds_filtered.to_csv('/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/hg38.p14.ncbiRefSeq.CDS_final_gtf.csv', index=False)

# -----------------------------------------------------------------------------------------------------------------------------

# processing the GTF file with transcripts
gtf_tr_path = '/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/GCF_000001405.40_GRCh38.p14_genomic.final_transcript.gtf'

col_names = [
    "seqname", "source", "feature", "start", "end",
    "score", "strand", "frame", "attributes"
]

gtf_tr = pd.read_csv(
    gtf_tr_path,
    sep='\t',
    comment='#',
    names=col_names,
    dtype=str
)

gtf_tr['gene_id'] = gtf_tr['attributes'].str.extract(r'gene_id "([^"]+)"')
gtf_tr['transcript_id'] = gtf_tr['attributes'].str.extract(r'transcript_id "([^"]+)"')
gtf_tr['entrez_id'] = gtf_tr['attributes'].str.extract(r'GeneID:(\d+)')

gtf_tr_filtered = gtf_tr[gtf_tr["transcript_id"].isin(transcripts_to_keep)].copy()

gtf_tr_filtered.to_csv('/mnt/home/aaggarwal/ceph/gates_proj/ncbi_genome_hg38.p14/hg38.p14.ncbiRefSeq.transcript_final_gtf.csv', index=False)


