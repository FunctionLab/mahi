import argparse
import pickle
from models.esm_model import load_esm_model, get_esm_embeddings
from Bio import SeqIO
import torch

def main(
    input_fasta: str,
    output_pkl: str,
    model_name: str = "esmc_600m",
    batch_size: int = 32,
    device: str = "cuda"
):
    # load ESM model
    esm_model = load_esm_model(model_name=model_name, device=device)

    # read protein sequences from FASTA
    proteins = []
    ids = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        ids.append(record.id)
        proteins.append(str(record.seq))

    # get embeddings one sequence at a time
    embeddings = {}
    for idx, (entrez_id, protein) in enumerate(zip(ids, proteins)):
        if idx <= 10 or idx % 1000 == 0 or idx == len(proteins):
            print(f"Processing {idx}/{len(proteins)}: gene {entrez_id}", flush=True)

        # deal with protein sequences larger than 2048
        if len(protein) <= 2048:
            full_embedding = get_esm_embeddings(esm_model, protein)
        else:
            chunks = [protein[i:i+2048] for i in range(0, len(protein), 2048)]
            all_embeddings = []

            for chunk in chunks:
                emb = get_esm_embeddings(esm_model, chunk)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach()
                all_embeddings.append(emb)

            full_embedding = torch.stack(all_embeddings, dim=0).mean(dim=0)

        embeddings[entrez_id] = full_embedding

    # save to pickle
    with open(output_pkl, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"saved ESM embeddings to {output_pkl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ESM embeddings from protein FASTA"
    )
    parser.add_argument("--input_fasta", required=True, help="Input FASTA file of protein sequences")
    parser.add_argument("--output_pkl", required=True, help="Output pickle file for embeddings")
    parser.add_argument("--model_name", default="esmc_600m", help="ESM model variant")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    args = parser.parse_args()

    main(
        input_fasta=args.input_fasta,
        output_pkl=args.output_pkl,
        model_name=args.model_name,
        device=args.device
    )