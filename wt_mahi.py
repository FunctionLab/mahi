import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run full Mahi pipeline")
    parser.add_argument("--dir", required=True, help="Base directory for inputs/outputs")
    parser.add_argument("--tissue", required=True, help="Tissue name (for Mahi embedding generation)")
    parser.add_argument("--checkpoint", required=True, help="Path to multigraph GNN checkpoint file")
    args = parser.parse_args()

    DIR = Path(args.dir)

    # 1. get multigraph embeddings
    run_cmd(
        f"python scripts/graph/2_get_multigraph_embed.py "
        f"--graph_csv {DIR}/multigraph_top3.csv "
        f"--nodes_txt resources/common_nodes.txt "
        f"--tissues_txt resources/35_mage_tissues.txt "
        f"--checkpoint {args.checkpoint} "
        f"--output_pkl {DIR}/multigraph_wt.pkl"
    )

    # 2. combine embeddings
    run_cmd(
        f"python scripts/mahi/3_combine_embeddings.py "
        f"--esm_embeddings_path {DIR}/esm-c_mean_embeddings.pkl "
        f"--deepsea_embeddings_path {DIR}/beluga_embeddings_single_exp.pkl "
        f"--graph_embeddings_path {DIR}/multigraph_wt.pkl "
        f"--output_embeddings_path {DIR}/esm_deepsea_graph_wt.pkl "
        f"--which_embeddings esm deepsea graph"
    )

    # 3. generate mahi embeddings
    run_cmd(
        f"python scripts/mahi/4_generate_mahi.py "
        f"--perturb_gene none "
        f"--embeddings {DIR}/esm_deepsea_graph_wt.pkl "
        f"--graph_dir {DIR}/networks "
        f"--tissues_list resources/all_mage_tissues.txt "
        f"--tissue {args.tissue} "
        f"--overwrite "
        f"--outdir {DIR}"
    )

    print("\n Mahi embeddings generated!\n")

if __name__ == "__main__":
    main()
