import argparse
import pickle
import torch
import os

def main(
    esm_embeddings_path,
    beluga_embeddings_path,
    graph_embeddings_path,
    output_embeddings_path,
    which_embeddings
):
    # load embeddings if requested
    esm_embeddings = beluga_embeddings = graph_embeddings = None

    if "esm" in which_embeddings:
        with open(esm_embeddings_path, "rb") as f:
            esm_embeddings = pickle.load(f)
    if "beluga" in which_embeddings:
        with open(beluga_embeddings_path, "rb") as f:
            beluga_embeddings = pickle.load(f)
    if "graph" in which_embeddings:
        with open(graph_embeddings_path, "rb") as f:
            graph_embeddings = pickle.load(f)

    # gather gene sets for all requested embeddings
    available_gene_sets = []
    if esm_embeddings is not None:
        available_gene_sets.append(set(esm_embeddings.keys()))
    if beluga_embeddings is not None:
        available_gene_sets.append(set(beluga_embeddings.keys()))
    if graph_embeddings is not None:
        available_gene_sets.append(set(graph_embeddings.keys()))

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
        if graph_embeddings is not None:
            embs.append(graph_embeddings[gene].cpu())

        concat_emb = torch.cat(embs, dim=-1)
        output_embeddings[gene] = concat_emb
        valid_nodes.append(gene)

    # save the concatenated embeddings
    with open(output_embeddings_path, "wb") as f:
        pickle.dump(output_embeddings, f)

    print(f"{len(valid_nodes)} overlapping nodes/genes/proteins")

    print(f"saved {len(valid_nodes)} gene/protein/graph embeddings to {output_embeddings_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate requested embeddings (esm, beluga, graph) and save."
    )
    parser.add_argument("--esm_embeddings_path", type=str, help="Path to esm_embeddings.pkl")
    parser.add_argument("--beluga_embeddings_path", type=str, help="Path to beluga_embeddings.pkl")
    parser.add_argument("--graph_embeddings_path", type=str, help="Path to graph_embeddings.pkl")
    parser.add_argument("--output_embeddings_path", type=str, required=True, help="Output path for concatenated embeddings.pkl")
    parser.add_argument("--which_embeddings", nargs='+', choices=['esm','beluga','graph'], required=True, help="Models to use (intersection of valid genes)")

    args = parser.parse_args()

    if (
        "esm" in args.which_embeddings and args.esm_embeddings_path is None or
        "beluga" in args.which_embeddings and args.beluga_embeddings_path is None or
        "graph" in args.which_embeddings and args.graph_embeddings_path is None
    ):
        print("Error: You must provide a path for at least one model's embeddings listed in --which_embeddings")
        sys.exit(1)

    main(
        esm_embeddings_path=args.esm_embeddings_path,
        beluga_embeddings_path=args.beluga_embeddings_path,
        graph_embeddings_path=args.graph_embeddings_path,
        output_embeddings_path=args.output_embeddings_path,
        which_embeddings=args.which_embeddings,
    )