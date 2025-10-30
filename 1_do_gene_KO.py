import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from trained MAGE model")
    parser.add_argument("--perturb_gene", type=str, required=True,
                        help="Entrez ID of the gene to KO from the graph")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save the perturbed graph CSV file")
    args = parser.parse_args()
    
    top = 'top3'
    
    # process the graph and the associated embeddings
    graph_path = f'/mnt/home/aaggarwal/ceph/gates_proj/MAGE_networks/xgboost_mage_2025_networks/all_filtered_edges_across_tissues_{top}.csv'
    
    graph = pd.read_csv(graph_path)
    print(f"Loaded initial graph from: {graph_path} ({len(graph)} edges)", flush=True)
    
    perturb_gene = args.perturb_gene
    perturb_gene = str(perturb_gene)
    
    src_str = graph["source"].astype(str)
    dst_str = graph["target"].astype(str)
    
    involved_mask = (src_str == perturb_gene) | (dst_str == perturb_gene)
    
    num_involved = int(involved_mask.sum())
    if num_involved > 0:
        perturbed_graph = graph.loc[~involved_mask].copy()
        print(f"KO node {perturb_gene} FOUND. Removed {num_involved} edges. New edge count: {len(perturbed_graph)}", flush=True)
    else:
        perturbed_graph = graph.copy()
        print(f"KO node {perturb_gene} NOT found. No edges removed. Saving unchanged graph.", flush=True)
    
    perturbed_graph.to_csv(args.output_csv, index=False)
    print(f"Saved perturbed graph to: {args.output_csv}", flush=True)

if __name__ == "__main__":
    main()