import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)

# format the tissue input accordingly
def resolve_tissues(args) -> list[str]:
    """
    Priority:
      1) --tissues_txt (one tissue per line; supports '#' comments)
      2) --tissues (space-separated list)
      3) --tissue (single)
    """
    if args.tissues_txt:
        p = Path(args.tissues_txt)
        lines = p.read_text().splitlines()
        tissues = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tissues.append(line)
        if not tissues:
            raise ValueError(f"No tissues found in {args.tissues_txt}")
        return tissues

    if args.tissues and len(args.tissues) > 0:
        return [t.strip() for t in args.tissues if t.strip()]

    if not args.tissue or not args.tissue.strip():
        raise ValueError("Provide --tissue, --tissues, or --tissues_txt.")
    return [args.tissue.strip()]

def main():
    parser = argparse.ArgumentParser(description="Run WT Mahi embedding generation")
    parser.add_argument("--dir", required=True, help="Base directory for inputs/outputs")
    parser.add_argument("--checkpoint", required=True, help="Path to multigraph GNN checkpoint file")

    # multi-tissue support
    parser.add_argument("--tissue", help="Single tissue name (backward-compatible)")
    parser.add_argument("--tissues", nargs="+", help="One or more tissues (space-separated)")
    parser.add_argument("--tissues_txt", help="Path to txt file with one tissue per line")

    args = parser.parse_args()
    DIR = Path(args.dir)

    tissues = resolve_tissues(args)

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
    for tissue in tissues:
        run_cmd(
            f"python scripts/mahi/4_generate_mahi.py "
            f"--perturb_gene none "
            f"--embeddings {DIR}/esm_deepsea_graph_wt.pkl "
            f"--graph_dir {DIR}/dat_networks "
            f"--tissues_list resources/all_mage_tissues.txt "
            f"--tissue {tissue} "
            f"--overwrite "
            f"--outdir {DIR}"
        )

    print(f"\nMahi WT embeddings generated for tissues: {', '.join(tissues)}\n")

if __name__ == "__main__":
    main()
