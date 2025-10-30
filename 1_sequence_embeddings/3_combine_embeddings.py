import argparse
import pickle
import torch
import os

def main(
    esm_embeddings_path,
    beluga_embeddings_path,
    seqweaver_embeddings_path,
    output_embeddings_path,
    output_nodes_path,
    which_embeddings
):
    # load embeddings if requested
    esm_embeddings = beluga_embeddings = seqweaver_embeddings = None

    if "esm" in which_embeddings:
        with open(esm_embeddings_path, "rb") as f:
            esm_embeddings = pickle.load(f)
    if "beluga" in which_embeddings:
        with open(beluga_embeddings_path, "rb") as f:
            beluga_embeddings = pickle.load(f)
    if "seqweaver" in which_embeddings:
        with open(seqweaver_embeddings_path, "rb") as f:
            seqweaver_embeddings = pickle.load(f)

    # gather gene sets for all requested embeddings
    available_gene_sets = []
    if esm_embeddings is not None:
        available_gene_sets.append(set(esm_embeddings.keys()))
    if beluga_embeddings is not None:
        available_gene_sets.append(set(beluga_embeddings.keys()))
    if seqweaver_embeddings is not None:
        available_gene_sets.append(set(seqweaver_embeddings.keys()))

    # intersection of all requested embeddings
    common_genes = set.intersection(*available_gene_sets)

    output_embeddings = {}
    valid_nodes = []

    for gene in common_genes:
        embs = []
        if esm_embeddings is not None:
            embs.append(esm_embeddings[gene].cpu())
        if beluga_embeddings is not None:
            embs.append(beluga_embeddings[gene].cpu())
        if seqweaver_embeddings is not None:
            embs.append(seqweaver_embeddings[gene].cpu())

        concat_emb = torch.cat(embs, dim=-1)
        output_embeddings[gene] = concat_emb
        valid_nodes.append(gene)

    # save the valid nodes and concatenated embeddings
    with open(output_embeddings_path, "wb") as f:
        pickle.dump(output_embeddings, f)

    with open(output_nodes_path, "w") as f:
        for node in valid_nodes:
            f.write(f"{node}\n")

    print(f"saved {len(valid_nodes)} gene/protein embeddings to {output_embeddings_path}")
    print(f"saved {len(valid_nodes)} node names to {output_nodes_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate requested embeddings (esm, beluga, seqweaver) and save."
    )
    parser.add_argument("--esm_embeddings_path", type=str, help="Path to esm_embeddings.pkl")
    parser.add_argument("--beluga_embeddings_path", type=str, help="Path to beluga_embeddings.pkl")
    parser.add_argument("--seqweaver_embeddings_path", type=str, help="Path to seqweaver_embeddings.pkl")
    parser.add_argument("--output_embeddings_path", type=str, required=True, help="Output path for concatenated embeddings.pkl")
    parser.add_argument("--output_nodes_path", type=str, required=True, help="Output path for list of valid nodes.pkl")
    parser.add_argument("--which_embeddings", nargs='+', choices=['esm','beluga','seqweaver'], required=True, help="Models to use (intersection of valid genes)")

    args = parser.parse_args()

    if (
        "esm" in args.which_embeddings and args.esm_embeddings_path is None or
        "beluga" in args.which_embeddings and args.beluga_embeddings_path is None or
        "seqweaver" in args.which_embeddings and args.seqweaver_embeddings_path is None
    ):
        print("Error: You must provide a path for at least one model's embeddings listed in --which_embeddings")
        sys.exit(1)

    main(
        esm_embeddings_path=args.esm_embeddings_path,
        beluga_embeddings_path=args.beluga_embeddings_path,
        seqweaver_embeddings_path=args.seqweaver_embeddings_path,
        output_embeddings_path=args.output_embeddings_path,
        output_nodes_path=args.output_nodes_path,
        which_embeddings=args.which_embeddings,
    )