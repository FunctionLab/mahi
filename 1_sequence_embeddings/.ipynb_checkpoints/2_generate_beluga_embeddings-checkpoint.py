import argparse
import pickle
from models.beluga_model import load_beluga_model, get_beluga_embeddings
from Bio import SeqIO
import torch
import numpy as np
#from sklearn.decomposition import PCA

def main(
    input_fasta: str,
    output_pkl: str,
    beluga_model_path: str,
    #pca_model_out: str,
    #pca_components: int,
    device: str = "cuda"
):
    # load the Beluga model
    beluga_model = load_beluga_model(beluga_model_path, device=device)

    # parse FASTA and store sequences
    seqs = []
    ids = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        ids.append(record.id)
        seqs.append(str(record.seq))

    # generate embeddings
    embeddings = {}
    #raw_embeddings = []
    for idx, (entrez_id, seq) in enumerate(zip(ids, seqs), 1):
        if idx <= 10 or idx % 1000 == 0 or idx == len(seqs):
            print(f"Processing {idx}/{len(seqs)}: gene {entrez_id}", flush=True)

        try:
            emb = get_beluga_embeddings(beluga_model, seq).view(-1)
            embeddings[entrez_id] = emb
        except KeyError as e:
            print(f"[SKIP] Gene {entrez_id} contains unsupported base: {e}. Skipping...", flush=True)
            continue
        #raw_embeddings.append(emb.numpy())

    with open(output_pkl, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved Beluga embeddings to {output_pkl}", flush=True)

    '''
    # stack and fit PCA
    X = np.stack(raw_embeddings)
    print(f"Fitting PCA on shape {X.shape}...", flush=True)

    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)

    # save PCA model
    with open(pca_model_out, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to {pca_model_out}", flush=True)

    # save PCA-transformed embeddings
    reduced_embeddings = {
        gene_id: torch.tensor(vec, dtype=torch.float32)
        for gene_id, vec in zip(ids, X_pca)
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(reduced_embeddings, f)
    print(f"Saved PCA-transformed embeddings to {output_pkl}", flush=True)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Beluga embeddings from DNA FASTA and reduce dimensionality using PCA"
    )
    parser.add_argument("--input_fasta", required=True, help="Input FASTA file of DNA sequences")
    parser.add_argument("--output_pkl", required=True, help="Output pickle file for PCA-reduced embeddings")
    #parser.add_argument("--pca_model_out", required=True, help="Output file to save trained PCA model")
    #parser.add_argument("--pca_dim", type=int, default=512, help="Target dimension after PCA")
    parser.add_argument("--beluga_model_path", required=True, help="Path to trained Beluga model (.pth)")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    args = parser.parse_args()

    main(
        input_fasta=args.input_fasta,
        output_pkl=args.output_pkl,
        beluga_model_path=args.beluga_model_path,
        #pca_model_out=args.pca_model_out,
        #pca_components=args.pca_dim,
        device=args.device
    )